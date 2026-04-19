from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Iterator, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from paperagent.agent.prompts import PromptLoader
from paperagent.config import Settings
from paperagent.schemas.models import AgentEvent, ChatSessionRecord, RetrievalResult
from paperagent.storage.repositories import ChatMessageRepository, ChatSessionRepository, PaperRepository


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    paper_id: str | None
    paper_title: str
    style: str
    chat_mode: str
    latest_retrieval: list[RetrievalResult]
    tool_iterations: int


class PaperChatAgent:
    def __init__(
        self,
        settings: Settings,
        paper_repository: PaperRepository,
        chat_session_repository: ChatSessionRepository,
        chat_message_repository: ChatMessageRepository,
        retrieval_service,
        paper_catalog_service=None,
    ) -> None:
        self.settings = settings
        self.paper_repository = paper_repository
        self.chat_session_repository = chat_session_repository
        self.chat_message_repository = chat_message_repository
        self.retrieval_service = retrieval_service
        self.paper_catalog_service = paper_catalog_service
        self.prompt_loader = PromptLoader()
        self.max_tool_iterations = 4
        self.tools = self._build_tools()
        self.graph = self._build_graph()

    def ask(
        self,
        paper_id: str | None,
        question: str,
        style: str = "beginner",
        session_id: str | None = None,
    ) -> Iterator[AgentEvent]:
        session, created = self._prepare_session(
            requested_session_id=session_id,
            requested_paper_id=paper_id,
            question=question,
            style=style,
        )
        effective_paper_id = session.paper_id
        paper = self.paper_repository.get_paper(effective_paper_id) if effective_paper_id else None
        if effective_paper_id and not paper:
            yield AgentEvent("error", f"Paper {effective_paper_id} not found.")
            return

        event_type = "session_created" if created else "session_loaded"
        yield AgentEvent(
            event_type,
            f"Using session {session.session_id}",
            {
                "session_id": session.session_id,
                "paper_id": session.paper_id,
                "style": session.style,
            },
        )

        chat_mode = "paper" if paper else "general"
        title = paper.title if paper else "general chat"
        yield AgentEvent("agent_started", f"Preparing answer for {title}")

        history = self.chat_message_repository.list_messages(session.session_id)
        initial_messages = history + [HumanMessage(content=question)]
        state: AgentState = {
            "session_id": session.session_id,
            "paper_id": effective_paper_id,
            "paper_title": paper.title if paper else "Indexed paper database",
            "style": session.style,
            "chat_mode": chat_mode,
            "messages": initial_messages,
            "latest_retrieval": [],
            "tool_iterations": 0,
        }
        final_state: AgentState | None = None
        emitted_tool_call_ids: set[str] = set()
        streamed_answer = False
        status_emitted = False

        for mode, data in self.graph.stream(
            state,
            stream_mode=["messages", "updates", "values"],
        ):
            if mode == "messages":
                chunk, metadata = data
                if metadata.get("langgraph_node") != "agent":
                    continue
                text = self._chunk_text(chunk)
                if not text:
                    continue
                if not status_emitted:
                    yield AgentEvent("status", "Generating final answer", {"session_id": session.session_id})
                    status_emitted = True
                streamed_answer = True
                yield AgentEvent("final_answer_stream", text, {"session_id": session.session_id})
                continue

            if mode == "updates":
                for event in self._yield_update_events(
                    data=data,
                    session_id=session.session_id,
                    emitted_tool_call_ids=emitted_tool_call_ids,
                ):
                    yield event
                continue

            if mode == "values":
                final_state = data

        result = final_state or state
        new_messages = result["messages"][len(history):]
        self.chat_message_repository.append_messages(session.session_id, new_messages)
        self._touch_session(session, paper_id=effective_paper_id, style=session.style)

        final_message = self._find_last_ai_without_tool_calls(result["messages"])
        answer = final_message.content if final_message else ""
        if streamed_answer:
            yield AgentEvent("final_answer_done", "", {"session_id": session.session_id})
            return
        yield from self._stream_answer(answer, session.session_id)

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_step)
        graph.add_node("tools", self._run_tools)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")
        return graph.compile()

    def _build_tools(self):
        @tool
        def search_paper_context(query: str) -> str:
            """Search imported paper chunks and return the most relevant evidence."""
            return "This tool is executed by the LangGraph tools node."

        return [search_paper_context]

    def _agent_step(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        if self.settings.llm_backend == "mock":
            return {"messages": [self._mock_agent_message(state)]}

        llm = self._chat_model(temperature=0.2).bind_tools(self.tools)
        system_prompt = self._load_system_prompt(state)
        model_input = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(model_input)

        if response.tool_calls and state.get("tool_iterations", 0) >= self.max_tool_iterations:
            response = AIMessage(
                content=(
                    "I have already used the search tool several times and still do not have enough certainty. "
                    "Please narrow the question or specify the paper you want me to focus on."
                )
            )
        return {"messages": [response]}

    def _run_tools(self, state: AgentState) -> AgentState:
        tool_messages: list[ToolMessage] = []
        latest_retrieval: list[RetrievalResult] = []
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return {
                "messages": [],
                "latest_retrieval": [],
                "tool_iterations": state.get("tool_iterations", 0) + 1,
            }

        for tool_call in last_message.tool_calls:
            tool_name = str(tool_call["name"])
            if tool_name != "search_paper_context":
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "error": f"Unsupported tool: {tool_name}",
                                "results": [],
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id=str(tool_call["id"]),
                    )
                )
                continue

            query = str(tool_call.get("args", {}).get("query", ""))
            content, retrievals = self._search_paper_context(
                query=query,
                paper_id=state.get("paper_id"),
            )
            latest_retrieval.extend(retrievals)
            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=str(tool_call["id"]),
                )
            )
        return {
            "messages": tool_messages,
            "latest_retrieval": latest_retrieval,
            "tool_iterations": state.get("tool_iterations", 0) + 1,
        }

    def _route_after_agent(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    def _prepare_session(
        self,
        requested_session_id: str | None,
        requested_paper_id: str | None,
        question: str,
        style: str,
    ) -> tuple[ChatSessionRecord, bool]:
        session_id = requested_session_id or uuid4().hex
        existing = self.chat_session_repository.get_session(session_id)
        now = datetime.utcnow()

        if existing:
            if existing.paper_id and requested_paper_id and existing.paper_id != requested_paper_id:
                raise ValueError(
                    f"Session {session_id} is already bound to paper {existing.paper_id}, "
                    f"not {requested_paper_id}."
                )
            paper_id = existing.paper_id or requested_paper_id
            resolved_style = existing.style if style == "beginner" else style
            title = existing.title or question[:80]
            updated = ChatSessionRecord(
                session_id=session_id,
                paper_id=paper_id,
                title=title,
                style=resolved_style,
                created_at=existing.created_at,
                updated_at=now,
            )
            self.chat_session_repository.upsert_session(updated)
            return updated, False

        created = ChatSessionRecord(
            session_id=session_id,
            paper_id=requested_paper_id,
            title=question[:80],
            style=style,
            created_at=now,
            updated_at=now,
        )
        self.chat_session_repository.upsert_session(created)
        return created, True

    def _touch_session(self, session: ChatSessionRecord, paper_id: str | None, style: str) -> None:
        updated = ChatSessionRecord(
            session_id=session.session_id,
            paper_id=paper_id,
            title=session.title,
            style=style,
            created_at=session.created_at,
            updated_at=datetime.utcnow(),
        )
        self.chat_session_repository.upsert_session(updated)

    def _load_system_prompt(self, state: AgentState) -> str:
        prompt_name = "agent_paper_system.txt" if state.get("chat_mode") == "paper" else "agent_general_system.txt"
        return self.prompt_loader.load(
            prompt_name,
            style=state.get("style", "beginner"),
            paper_title=state.get("paper_title", "Indexed paper database"),
            paper_scope=state.get("paper_id") or "all indexed papers",
        )

    def _mock_agent_message(self, state: AgentState) -> AIMessage:
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            normalized = last_message.content.strip().lower()
            greetings = {"hello", "hi", "hey", "你好", "您好"}
            if normalized in greetings:
                return AIMessage(
                    content=(
                        "Hello. I can chat normally, and if you ask about imported papers I can decide whether "
                        "to search the paper database."
                    )
                )
            should_search = any(
                word in normalized
                for word in ["method", "experiment", "result", "contribution", "limitation", "detail", "paper"]
            )
            if should_search:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_paper_context",
                            "args": {"query": last_message.content},
                            "id": f"call_{uuid4().hex[:8]}",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(
                content=(
                    "This looks like a general chat question, so I answered directly without searching the database. "
                    "If you want evidence from imported papers, ask a more paper-specific question."
                )
            )

        if isinstance(last_message, ToolMessage):
            retrievals = self._extract_retrieval_results([last_message])
            if retrievals:
                bullet_lines = [
                    f"- 来自论文 {item.paper_id} / {item.section_title} (p.{item.page_number}, {item.chunk_id})：{item.content[:160]}"
                    for item in retrievals[:3]
                ]
                return AIMessage(
                    content=(
                        "我已经结合检索结果整理出答案：\n"
                        + "\n".join(bullet_lines)
                        + "\n\n如果你想继续深入，我可以继续围绕这些证据展开。"
                    )
                )
            return AIMessage(
                content=(
                    "我尝试搜索了数据库，但没有找到足够相关的证据。"
                    "你可以换一种问法，或者指定某篇论文再继续问。"
                )
            )

        return AIMessage(content="I am ready to continue this session.")

    def _extract_retrieval_results(self, tool_messages: list[BaseMessage]) -> list[RetrievalResult]:
        retrievals: list[RetrievalResult] = []
        for message in tool_messages:
            if not isinstance(message, ToolMessage):
                continue
            try:
                payload = json.loads(message.content)
            except json.JSONDecodeError:
                continue
            for item in payload.get("results", []):
                retrievals.append(
                    RetrievalResult(
                        paper_id=str(item["paper_id"]),
                        chunk_id=str(item["chunk_id"]),
                        content=str(item["content"]),
                        section_title=str(item["section_title"]),
                        page_number=int(item["page_number"]),
                        score=float(item.get("score", 0.0)),
                        source="hybrid",
                    )
                )
        return retrievals

    def _search_paper_context(
        self,
        query: str,
        paper_id: str | None,
    ) -> tuple[str, list[RetrievalResult]]:
        results = self.retrieval_service.retrieve(
            query=query,
            paper_id=paper_id,
            top_k=self.settings.default_top_k,
        )
        payload = {
            "query": query,
            "results": [
                {
                    "paper_id": item.paper_id,
                    "chunk_id": item.chunk_id,
                    "section_title": item.section_title,
                    "page_number": item.page_number,
                    "score": item.score,
                    "content": item.content,
                }
                for item in results
            ],
        }
        return json.dumps(payload, ensure_ascii=False), results

    def _find_last_ai_without_tool_calls(self, messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                return message
        return None

    def _stream_answer(self, answer: str, session_id: str) -> Iterator[AgentEvent]:
        yield AgentEvent("status", "Generating final answer", {"session_id": session_id})
        for char in answer:
            yield AgentEvent("final_answer_stream", char, {"session_id": session_id})
        yield AgentEvent("final_answer_done", "", {"session_id": session_id})

    def _chat_model(self, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.settings.llm_model,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            temperature=temperature,
            streaming=True,
        )

    def _yield_update_events(
        self,
        data: dict,
        session_id: str,
        emitted_tool_call_ids: set[str],
    ) -> Iterator[AgentEvent]:
        for node_name, update in data.items():
            if node_name == "agent":
                for message in update.get("messages", []):
                    if not isinstance(message, AIMessage) or not message.tool_calls:
                        continue
                    for tool_call in message.tool_calls:
                        tool_call_id = str(tool_call.get("id") or "")
                        if tool_call_id and tool_call_id in emitted_tool_call_ids:
                            continue
                        if tool_call_id:
                            emitted_tool_call_ids.add(tool_call_id)
                        yield AgentEvent(
                            "tool_called",
                            f"Calling tool {tool_call['name']}",
                            {
                                "tool_name": tool_call["name"],
                                "args": tool_call.get("args", {}),
                                "session_id": session_id,
                            },
                        )

            if node_name == "tools":
                retrievals = update.get("latest_retrieval", [])
                if not retrievals:
                    continue
                yield AgentEvent(
                    "rag_hit",
                    f"Retrieved {len(retrievals)} relevant chunks.",
                    {
                        "chunks": [
                            {
                                "paper_id": item.paper_id,
                                "chunk_id": item.chunk_id,
                                "section_title": item.section_title,
                                "page_number": item.page_number,
                                "preview": item.content[:160],
                            }
                            for item in retrievals
                        ],
                        "session_id": session_id,
                    },
                )

    def _chunk_text(self, chunk) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        return ""
