from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Any, Iterator, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from paperagent.agent.paper_resolution import is_ppt_request, resolve_ppt_target
from paperagent.agent.prompts import PromptLoader
from paperagent.config import Settings
from paperagent.schemas.models import AgentEvent, ChatSessionRecord, DeckContent, PaperCatalogResult, RetrievalResult, SlideContent
from paperagent.storage.repositories import ChatMessageRepository, ChatSessionRepository, PaperRepository


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    paper_id: str | None
    paper_title: str
    style: str
    chat_mode: str
    ppt_intent: bool
    ppt_target_paper_id: str | None
    ppt_target_paper_title: str | None
    latest_retrieval: list[RetrievalResult]
    latest_paper_catalog: list[PaperCatalogResult]
    latest_ppt_result: dict[str, Any] | None
    tool_iterations: int


class GeneratePPTInput(BaseModel):
    paper_id: str
    title: str
    audience: str = "beginner"
    slides: list[dict[str, Any]] = Field(default_factory=list)


class PaperChatAgent:
    def __init__(
        self,
        settings: Settings,
        paper_repository: PaperRepository,
        chat_session_repository: ChatSessionRepository,
        chat_message_repository: ChatMessageRepository,
        retrieval_service,
        paper_catalog_service=None,
        ppt_service=None,
    ) -> None:
        self.settings = settings
        self.paper_repository = paper_repository
        self.chat_session_repository = chat_session_repository
        self.chat_message_repository = chat_message_repository
        self.retrieval_service = retrieval_service
        self.paper_catalog_service = paper_catalog_service
        self.ppt_service = ppt_service
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

        ppt_intent = is_ppt_request(question)
        ppt_target = resolve_ppt_target(
            prompt=question,
            scoped_paper_id=effective_paper_id,
            paper_repository=self.paper_repository,
            paper_catalog_service=self.paper_catalog_service,
        ) if ppt_intent else None

        history = self.chat_message_repository.list_messages(session.session_id)
        human_message = HumanMessage(content=question)

        if ppt_intent and (not ppt_target or not ppt_target.paper_id):
            failure_message = (
                "I couldn't determine which indexed paper to use for this PPT request. "
                "Please mention the paper title or start the chat with a scoped paper."
            )
            self.chat_message_repository.append_messages(
                session.session_id,
                [human_message, AIMessage(content=failure_message)],
            )
            self._touch_session(session, paper_id=effective_paper_id, style=session.style)
            yield from self._stream_answer(failure_message, session.session_id)
            return

        initial_messages = history + [human_message]
        state: AgentState = {
            "session_id": session.session_id,
            "paper_id": effective_paper_id,
            "paper_title": paper.title if paper else "Indexed paper database",
            "style": session.style,
            "chat_mode": chat_mode,
            "ppt_intent": ppt_intent,
            "ppt_target_paper_id": ppt_target.paper_id if ppt_target else None,
            "ppt_target_paper_title": ppt_target.paper_title if ppt_target else None,
            "messages": initial_messages,
            "latest_retrieval": [],
            "latest_paper_catalog": [],
            "latest_ppt_result": None,
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
        def search_paper_context(query: str, paper_id: str | None = None) -> str:
            """Search imported paper chunks and return the most relevant evidence."""
            return "This tool is executed by the LangGraph tools node."

        @tool
        def search_papers(query: str) -> str:
            """Search the indexed paper catalog and return relevant papers."""
            return "This tool is executed by the LangGraph tools node."

        @tool
        def get_paper_profile(paper_id: str) -> str:
            """Return the cached abstract summary and keywords for one indexed paper."""
            return "This tool is executed by the LangGraph tools node."

        @tool(args_schema=GeneratePPTInput)
        def generate_ppt(
            paper_id: str,
            title: str,
            audience: str = "beginner",
            slides: list[dict[str, Any]] | None = None,
        ) -> str:
            """Render a structured PPT deck for one indexed paper."""
            return "This tool is executed by the LangGraph tools node."

        return [search_paper_context, search_papers, get_paper_profile, generate_ppt]

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
        latest_paper_catalog: list[PaperCatalogResult] = []
        latest_ppt_result: dict[str, Any] | None = None
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return {
                "messages": [],
                "latest_retrieval": [],
                "latest_paper_catalog": [],
                "latest_ppt_result": None,
                "tool_iterations": state.get("tool_iterations", 0) + 1,
            }

        for tool_call in last_message.tool_calls:
            tool_name = str(tool_call["name"])
            if tool_name == "search_paper_context":
                query = str(tool_call.get("args", {}).get("query", ""))
                requested_paper_id = tool_call.get("args", {}).get("paper_id") or state.get("paper_id")
                content, retrievals = self._search_paper_context(
                    query=query,
                    paper_id=str(requested_paper_id) if requested_paper_id else None,
                )
                latest_retrieval.extend(retrievals)
                tool_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=str(tool_call["id"]),
                    )
                )
                continue

            if tool_name == "search_papers":
                query = str(tool_call.get("args", {}).get("query", ""))
                content, catalog_hits = self._search_papers(query=query)
                latest_paper_catalog.extend(catalog_hits)
                tool_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=str(tool_call["id"]),
                    )
                )
                continue

            if tool_name == "get_paper_profile":
                requested_paper_id = str(tool_call.get("args", {}).get("paper_id", "")).strip()
                content = self._get_paper_profile(requested_paper_id)
                tool_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=str(tool_call["id"]),
                    )
                )
                continue

            if tool_name == "generate_ppt":
                content, latest_ppt_result = self._generate_ppt(tool_call.get("args", {}))
                tool_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=str(tool_call["id"]),
                    )
                )
                continue

            tool_messages.append(
                ToolMessage(
                    content=json.dumps(
                        {
                            "tool_name": tool_name,
                            "error": f"Unsupported tool: {tool_name}",
                            "results": [],
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id=str(tool_call["id"]),
                )
            )
        return {
            "messages": tool_messages,
            "latest_retrieval": latest_retrieval,
            "latest_paper_catalog": latest_paper_catalog,
            "latest_ppt_result": latest_ppt_result,
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
        prompt = self.prompt_loader.load(
            prompt_name,
            style=state.get("style", "beginner"),
            paper_title=state.get("paper_title", "Indexed paper database"),
            paper_scope=state.get("paper_id") or "all indexed papers",
        )
        if not state.get("ppt_intent"):
            return prompt
        skill_prompt = self.prompt_loader.load(
            "ppt_generation_skill.txt",
            ppt_target_paper_id=state.get("ppt_target_paper_id") or "",
            ppt_target_paper_title=state.get("ppt_target_paper_title") or "",
        )
        return f"{prompt}\n\n{skill_prompt}"

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
            if state.get("ppt_intent"):
                target_paper_id = state.get("ppt_target_paper_id")
                if not target_paper_id:
                    return AIMessage(
                        content=(
                            "I couldn't determine which indexed paper to use for this PPT request. "
                            "Please mention the paper title or use a scoped paper chat."
                        )
                    )
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_paper_context",
                            "args": {
                                "query": "Summarize the paper problem, method, experiments, and conclusion for a PPT deck.",
                                "paper_id": target_paper_id,
                            },
                            "id": f"call_{uuid4().hex[:8]}",
                            "type": "tool_call",
                        }
                    ],
                )
            should_search_catalog = (
                state.get("chat_mode") == "general"
                and any(
                    phrase in normalized
                    for phrase in [
                        "which paper",
                        "what papers",
                        "papers about",
                        "related papers",
                        "有哪些论文",
                        "哪些论文",
                        "什么论文",
                    ]
                )
            )
            if state.get("chat_mode") == "general" and not should_search_catalog:
                should_search_catalog = (
                    "论文" in normalized
                    and any(token in normalized for token in ["哪些", "哪篇", "什么", "watermark", "related"])
                )
            if should_search_catalog:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_papers",
                            "args": {"query": last_message.content},
                            "id": f"call_{uuid4().hex[:8]}",
                            "type": "tool_call",
                        }
                    ],
                )

            should_search_context = any(
                word in normalized
                for word in [
                    "method",
                    "experiment",
                    "result",
                    "contribution",
                    "limitation",
                    "detail",
                    "paper",
                    "方法",
                    "实验",
                    "结果",
                    "贡献",
                    "局限",
                    "细节",
                ]
            )
            if should_search_context:
                if state.get("chat_mode") == "general" and any(token in normalized for token in ["tag-wm", "roar", "seal"]):
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_papers",
                                "args": {"query": last_message.content},
                                "id": f"call_{uuid4().hex[:8]}",
                                "type": "tool_call",
                            }
                        ],
                    )
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_paper_context",
                            "args": {"query": last_message.content, "paper_id": state.get("paper_id")},
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
            payload = self._safe_tool_payload(last_message.content)
            tool_name = str(payload.get("tool_name", ""))
            if tool_name == "generate_ppt":
                if payload.get("error"):
                    return AIMessage(
                        content=(
                            "我尝试生成 PPT，但结构化内容没有通过校验或渲染失败："
                            + str(payload["error"])
                        )
                    )
                return AIMessage(
                    content=(
                        f"我已经为论文 {payload.get('paper_id', '')} 生成 PPT，"
                        f"输出路径是 {payload.get('ppt_path', '')}。"
                    )
                )
            if tool_name == "search_papers":
                papers = payload.get("results", [])
                if len(papers) == 1:
                    original_question = ""
                    for message in reversed(state["messages"]):
                        if isinstance(message, HumanMessage):
                            original_question = str(message.content)
                            break
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_paper_context",
                                "args": {
                                    "query": original_question,
                                    "paper_id": papers[0]["paper_id"],
                                },
                                "id": f"call_{uuid4().hex[:8]}",
                                "type": "tool_call",
                            }
                        ],
                    )
                if papers:
                    bullet_lines = [
                        f"- {item['paper_id']}: {item['title']} | {item.get('short_summary', '')}".strip()
                        for item in papers[:5]
                    ]
                    return AIMessage(
                        content=(
                            "我先在论文目录里找到了这些相关论文：\n"
                            + "\n".join(bullet_lines)
                            + "\n\n如果你想继续，我可以进一步讲其中某一篇的方法、实验或结论。"
                        )
                    )

            retrievals = self._extract_retrieval_results([last_message])
            if state.get("ppt_intent") and state.get("ppt_target_paper_id"):
                deck_content = self._build_mock_ppt_deck(
                    paper_id=str(state["ppt_target_paper_id"]),
                    paper_title=state.get("ppt_target_paper_title") or "Paper Deck",
                    retrievals=retrievals,
                    audience=state.get("style", "beginner"),
                )
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "generate_ppt",
                            "args": self._deck_content_to_tool_args(deck_content),
                            "id": f"call_{uuid4().hex[:8]}",
                            "type": "tool_call",
                        }
                    ],
                )
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
            payload = self._safe_tool_payload(message.content)
            for item in payload.get("results", []):
                if "chunk_id" not in item:
                    continue
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
            "tool_name": "search_paper_context",
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

    def _search_papers(self, query: str) -> tuple[str, list[PaperCatalogResult]]:
        results = self.paper_catalog_service.search_papers(query, top_k=5) if self.paper_catalog_service else []
        payload = {
            "tool_name": "search_papers",
            "query": query,
            "results": [
                {
                    "paper_id": item.paper_id,
                    "title": item.title,
                    "short_summary": item.short_summary,
                    "keywords": item.keywords,
                    "score": item.score,
                }
                for item in results
            ],
        }
        return json.dumps(payload, ensure_ascii=False), results

    def _get_paper_profile(self, paper_id: str) -> str:
        profile = self.paper_repository.get_profile(paper_id)
        paper = self.paper_repository.get_paper(paper_id)
        payload = {
            "tool_name": "get_paper_profile",
            "paper_id": paper_id,
            "title": paper.title if paper else "",
            "abstract_text": profile.abstract_text if profile else "",
            "short_summary": profile.short_summary if profile else "",
            "keywords": profile.keywords if profile else [],
            "profile_status": profile.profile_status if profile else "missing",
        }
        return json.dumps(payload, ensure_ascii=False)

    def _generate_ppt(self, raw_args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        if not self.ppt_service:
            payload = {
                "tool_name": "generate_ppt",
                "error": "PPT service is unavailable.",
            }
            return json.dumps(payload, ensure_ascii=False), None

        try:
            deck_content = self._tool_args_to_deck_content(raw_args)
            result = self.ppt_service.generate_from_content(deck_content)
        except Exception as exc:  # noqa: BLE001
            payload = {
                "tool_name": "generate_ppt",
                "paper_id": str(raw_args.get("paper_id", "")).strip(),
                "error": str(exc),
            }
            return json.dumps(payload, ensure_ascii=False), None

        payload = {
            "tool_name": "generate_ppt",
            **result,
        }
        return json.dumps(payload, ensure_ascii=False), result

    def _tool_args_to_deck_content(self, raw_args: dict[str, Any]) -> DeckContent:
        slides = raw_args.get("slides", [])
        if not isinstance(slides, list):
            slides = []

        return DeckContent(
            paper_id=str(raw_args.get("paper_id", "")).strip(),
            title=str(raw_args.get("title", "")).strip(),
            audience=str(raw_args.get("audience", "beginner")).strip() or "beginner",
            slides=[
                SlideContent(
                    slide_type=str(item.get("slide_type", item.get("type", ""))).strip(),
                    title=str(item.get("title", "")).strip(),
                    bullets=[str(bullet).strip() for bullet in item.get("bullets", []) if str(bullet).strip()],
                    notes=str(item.get("notes", "")).strip(),
                    citations=[
                        str(citation).strip()
                        for citation in item.get("citations", [])
                        if str(citation).strip()
                    ],
                    layout_hint=str(item.get("layout_hint", "")).strip(),
                    visual_intent=str(item.get("visual_intent", "")).strip(),
                )
                for item in slides
                if isinstance(item, dict)
            ],
        )

    def _deck_content_to_tool_args(self, deck_content: DeckContent) -> dict[str, Any]:
        return {
            "paper_id": deck_content.paper_id,
            "title": deck_content.title,
            "audience": deck_content.audience,
            "slides": [
                {
                    "slide_type": slide.slide_type,
                    "title": slide.title,
                    "bullets": slide.bullets,
                    "notes": slide.notes,
                    "citations": slide.citations,
                    "layout_hint": slide.layout_hint,
                    "visual_intent": slide.visual_intent,
                }
                for slide in deck_content.slides
            ],
        }

    def _build_mock_ppt_deck(
        self,
        paper_id: str,
        paper_title: str,
        retrievals: list[RetrievalResult],
        audience: str,
    ) -> DeckContent:
        bullets_from_retrievals = [
            item.content.replace("\n", " ").strip()[:120]
            for item in retrievals[:4]
            if item.content.strip()
        ]
        citations = [item.chunk_id for item in retrievals[:4]]
        if not bullets_from_retrievals:
            bullets_from_retrievals = [
                "Explain the paper problem and motivation.",
                "Summarize the method at a beginner-friendly level.",
            ]

        slides = [
            SlideContent(
                slide_type="title",
                title=paper_title,
                bullets=["Problem, method, and results overview"],
                notes="Open with the paper framing.",
                citations=[],
                layout_hint="title",
                visual_intent="hero",
            ),
            SlideContent(
                slide_type="background",
                title="Problem & Motivation",
                bullets=bullets_from_retrievals[:2],
                notes="Explain why this paper matters.",
                citations=citations[:2],
                layout_hint="content",
                visual_intent="problem framing",
            ),
            SlideContent(
                slide_type="method",
                title="Core Method",
                bullets=bullets_from_retrievals[:3],
                notes="Walk through the main pipeline.",
                citations=citations[:3],
                layout_hint="content",
                visual_intent="method summary",
            ),
            SlideContent(
                slide_type="experiments",
                title="Experiments & Results",
                bullets=(bullets_from_retrievals[1:4] or bullets_from_retrievals[:2]),
                notes="Highlight the most important evidence.",
                citations=citations[:3],
                layout_hint="content",
                visual_intent="results summary",
            ),
            SlideContent(
                slide_type="conclusion",
                title="Takeaways",
                bullets=[
                    "Summarize the main contribution.",
                    "Mention one limitation or next step.",
                ],
                notes="Close with a balanced summary.",
                citations=citations[:1],
                layout_hint="content",
                visual_intent="closing summary",
            ),
        ]
        return DeckContent(
            paper_id=paper_id,
            title=f"{paper_title} Presentation",
            audience=audience,
            slides=slides,
        )

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
                ppt_result = update.get("latest_ppt_result")
                if ppt_result:
                    yield AgentEvent(
                        "ppt_generated",
                        f"Generated PPT for {ppt_result['title']}.",
                        {
                            "paper_id": ppt_result["paper_id"],
                            "title": ppt_result["title"],
                            "content_path": ppt_result["content_path"],
                            "ppt_path": ppt_result["ppt_path"],
                            "slide_count": ppt_result["slide_count"],
                            "renderer": ppt_result["renderer"],
                            "session_id": session_id,
                        },
                    )
                catalog_hits = update.get("latest_paper_catalog", [])
                if catalog_hits:
                    yield AgentEvent(
                        "catalog_hit",
                        f"Retrieved {len(catalog_hits)} relevant papers.",
                        {
                            "papers": [
                                {
                                    "paper_id": item.paper_id,
                                    "title": item.title,
                                    "short_summary": item.short_summary,
                                    "keywords": item.keywords,
                                }
                                for item in catalog_hits
                            ],
                            "session_id": session_id,
                        },
                    )
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

    def _safe_tool_payload(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
