from __future__ import annotations

import json
from typing import Iterable, Iterator, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from paperagent.config import Settings
from paperagent.schemas.models import AgentEvent, RetrievalResult
from paperagent.storage.repositories import PaperRepository


class AgentState(TypedDict, total=False):
    paper_id: str
    paper_title: str
    question: str
    style: str
    should_search: bool
    search_query: str
    evidence: list[RetrievalResult]
    answer: str


class PaperChatAgent:
    def __init__(self, settings: Settings, paper_repository: PaperRepository, retrieval_service) -> None:
        self.settings = settings
        self.paper_repository = paper_repository
        self.retrieval_service = retrieval_service
        self.graph = self._build_graph()

    def ask(self, paper_id: str, question: str, style: str = "beginner") -> Iterator[AgentEvent]:
        paper = self.paper_repository.get_paper(paper_id)
        if not paper:
            yield AgentEvent("error", f"Paper {paper_id} not found.")
            return

        yield AgentEvent("agent_started", f"Preparing answer for {paper.title}")
        state: AgentState = {
            "paper_id": paper_id,
            "paper_title": paper.title,
            "question": question,
            "style": style,
        }
        result = self.graph.invoke(state)
        if result.get("should_search"):
            evidence = result.get("evidence", [])
            yield AgentEvent(
                "rag_hit",
                f"Retrieved {len(evidence)} relevant chunks.",
                {
                    "chunks": [
                        {
                            "chunk_id": item.chunk_id,
                            "section_title": item.section_title,
                            "page_number": item.page_number,
                            "preview": item.content[:160],
                        }
                        for item in evidence
                    ]
                },
            )
        yield from self._stream_answer(result.get("answer", ""))

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("decide", self._decide)
        graph.add_node("search", self._search)
        graph.add_node("answer", self._answer)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges(
            "decide",
            lambda state: "search" if state.get("should_search") else "answer",
            {"search": "search", "answer": "answer"},
        )
        graph.add_edge("search", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    def _decide(self, state: AgentState) -> AgentState:
        if self.settings.llm_backend == "mock":
            should_search = any(
                word in state["question"].lower()
                for word in ["method", "experiment", "result", "contribution", "limitation", "detail"]
            )
            return {
                "should_search": should_search,
                "search_query": state["question"],
            }

        llm = self._chat_model(temperature=0)
        prompt = (
            "You are a paper explanation assistant. Decide whether the user question needs document retrieval. "
            "Return compact JSON with keys should_search (boolean) and search_query (string). "
            "Set should_search to false for general explanations or reformulations."
        )
        response = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=json.dumps(
                        {
                            "paper_title": state["paper_title"],
                            "question": state["question"],
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        payload = self._safe_json_loads(response.content)
        return {
            "should_search": bool(payload.get("should_search", True)),
            "search_query": str(payload.get("search_query") or state["question"]),
        }

    def _search(self, state: AgentState) -> AgentState:
        results = self.retrieval_service.retrieve(
            query=state.get("search_query") or state["question"],
            paper_id=state["paper_id"],
        )
        return {"evidence": results}

    def _answer(self, state: AgentState) -> AgentState:
        evidence = state.get("evidence", [])
        if self.settings.llm_backend == "mock":
            answer = self._mock_answer(state, evidence)
            return {"answer": answer}

        llm = self._chat_model(temperature=0.2)
        evidence_text = "\n\n".join(
            f"[{item.chunk_id}] section={item.section_title} page={item.page_number}\n{item.content}"
            for item in evidence
        )
        if not evidence_text:
            evidence_text = "No retrieval evidence used."
        system_prompt = (
            "You explain papers clearly for beginners. "
            "Use evidence when provided, cite chunk ids inline, and say you are unsure when evidence is missing."
        )
        user_prompt = json.dumps(
            {
                "paper_title": state["paper_title"],
                "style": state["style"],
                "question": state["question"],
                "evidence": evidence_text,
            },
            ensure_ascii=False,
        )
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        return {"answer": str(response.content)}

    def _mock_answer(self, state: AgentState, evidence: list[RetrievalResult]) -> str:
        if evidence:
            bullet_lines = [
                f"- 来自 {item.section_title} (p.{item.page_number}, {item.chunk_id})：{item.content[:160]}"
                for item in evidence[:3]
            ]
            return (
                f"问题：{state['question']}\n"
                f"这是一份面向初学者的解释。系统检索到了以下证据：\n"
                + "\n".join(bullet_lines)
                + "\n\n简要总结：论文的核心信息已经从这些片段中提取出来，你可以继续追问方法细节、实验设置或局限性。"
            )
        return (
            f"问题：{state['question']}\n"
            "当前没有调用检索工具，因此我先给出通用解释。"
            "如果你需要论文中的具体实验数字、方法流程或结论依据，可以继续追问更细的问题。"
        )

    def _stream_answer(self, answer: str) -> Iterable[AgentEvent]:
        yield AgentEvent("status", "Generating final answer")
        for token in answer.split():
            yield AgentEvent("final_answer_stream", token + " ")
        yield AgentEvent("final_answer_done", "")

    def _chat_model(self, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.settings.llm_model,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            temperature=temperature,
        )

    def _safe_json_loads(self, payload: str) -> dict:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"should_search": True, "search_query": payload}
