# 开发进度表

| 阶段 | 阶段目标 | 当前状态 | 产出物 | 验证命令 | 验证结果 | 遗留问题 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 项目脚手架、CLI、配置、文档、测试基线 | done | `pyproject.toml`, `src/`, `docs/architecture.md`, `README.md` | `python -m pytest -q`, `python -m paperagent.cli.app --help` | 通过 | 以后可再补 lint/type check |
| A2 | PDF/URL 导入、Markdown、SQLite 落库 | done | `ingest/`, SQLite 表结构, PDF/MD 文件落盘, 可切换 `pymupdf` / `datalab` PDF->MD backend | `python -m pytest -q` | 通过 | Markdown 提取质量仍可继续优化 |
| B1 | Chroma + BM25 混合检索 | done | `retrieval/`, Chroma 持久化, BM25 JSON 索引 | `python -m pytest -q` | 通过 | 目前未加入更强 reranker |
| B2 | ReAct 风格 Agent CLI 与流式事件 | done | `agent/`, `chat ask`, `chat interactive`, 事件流 | `python -m pytest -q` | 通过 | 真实 OpenAI 工具调用轮次仍可升级 |
| C1 | PPT 生成基础版 | done | `ppt/`, `deck.json`, `.pptx` 输出 | `python -m pytest -q` | 通过 | 视觉样式与图表抽取可继续增强 |
| C2 | 健壮性、缓存、doctor、回归测试 | done | `doctor`, 缓存导入, URL 稳定性修复, API 测试 | `python -m pytest -q`, `python -m paperagent.cli.app doctor` | 通过 | 可继续补更多异常分级与日志 |
| D1 | FastAPI + React 可视化 | done | `web/api.py`, `frontend/` React + Vite 页面 | `npm run build`, `python -m pytest -q` | 通过 | 目前未接入文件预览和任务队列 |
| E1 | Chat 驱动 PPT agent/tool 重构 | done | `agent/` PPT 意图识别, 项目内 `ppt_generation_skill.txt`, 结构化 `DeckContent`, `ppt_generated` 事件 | `python -m pytest -q` | 通过 | 真实模型下的 deck 质量仍可继续优化 |
| E2 | 旧 PPT 入口清理与 chat-only 收口 | done | 删除 `pptgen` / `/ppt`, React 聊天触发, README 与测试重构 | `python -m pytest -q`, `npm run build` | 通过 | 前端目前仍是单会话原型页面 |
