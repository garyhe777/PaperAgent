# PaperAgent

PaperAgent is a CLI-first assistant for importing academic papers, building hybrid retrieval indexes, chatting about a paper, and generating presentation decks.

## Quick Start

```powershell
paperagent doctor
paperagent ingest --pdf path\to\paper.pdf
paperagent chat --paper-id <paper_id> --question "What is the main contribution?"
paperagent ppt generate --paper-id <paper_id>
paperagent serve-api
```

The first implementation favors a clear code structure and testability so it can later power a FastAPI + React interface.

## Environment

The project is designed to run inside the existing `conda activate agent` environment.

- For local tests and offline development, the default embedding backend is `hash`.
- For real LLM answers, set `PAPERAGENT_LLM_API_KEY` and optionally `PAPERAGENT_LLM_BASE_URL`.
- For deterministic test runs, set `PAPERAGENT_LLM_BACKEND=mock`.

## Development Commands

```powershell
$env:PYTHONPATH="d:\Code\PaperAgent0419\src"
python -m pytest -q
python -m paperagent.cli.app doctor
python -m paperagent.cli.app ingest --pdf path\to\paper.pdf
python -m paperagent.cli.app chat ask --paper-id <paper_id> --question "Explain the method"
python -m paperagent.cli.app ppt generate --paper-id <paper_id>
python -m paperagent.cli.app serve-api
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```
