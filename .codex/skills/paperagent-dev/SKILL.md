---
name: paperagent-dev
description: Use when developing, debugging, testing, or committing changes inside the PaperAgent repository. Captures the default conda environment, Python path setup, progress-table update rule, one-stage-one-commit workflow, and key module navigation.
---

# PaperAgent Dev Workflow

Use this skill when working inside the `PaperAgent0419` repository.

## Environment

- Start from the project root: `D:\Code\PaperAgent0419`
- Activate the shared environment before development commands:
  - `conda activate agent`
- When running Python commands without relying on editable install, set:
  - `$env:PYTHONPATH="D:\Code\PaperAgent0419\src"`

## Development Rules

- Treat each development stage as one small, complete unit of work.
- After finishing a stage, update `PROGRESS.md`.
- Keep one stage to one Git commit when practical.
- Run tests before commit. Default command:
  - `python -m pytest -q`
- Prefer non-destructive, incremental changes.
- When changing behavior, update related docs or tests in the same stage if needed.

## Project Map

- CLI entrypoint:
  - `src/paperagent/cli/app.py`
- Configuration:
  - `src/paperagent/config.py`
- Storage and repositories:
  - `src/paperagent/storage/`
- Ingest and PDF/Markdown conversion:
  - `src/paperagent/ingest/`
- Retrieval and catalog search:
  - `src/paperagent/retrieval/`
- Chat agent and tool-calling logic:
  - `src/paperagent/agent/`
- PPT planning, enrichment, and rendering:
  - `src/paperagent/ppt/`
- Prompts:
  - `src/paperagent/prompts/`
- Tests:
  - `tests/`

## Common Commands

- Run tests:
  - `python -m pytest -q`
- Show CLI help:
  - `python -m paperagent.cli.app --help`
- Environment and dependency check:
  - `python -m paperagent.cli.app doctor`
- Ingest a paper:
  - `python -m paperagent.cli.app ingest --pdf path\to\paper.pdf`
- Ask a question:
  - `python -m paperagent.cli.app chat ask --question "Explain the method"`
- Continue a session:
  - `python -m paperagent.cli.app chat ask --session-id <session_id> --question "..."`

## Data and Output Paths

- Main local data root:
  - `.paperagent_data/`
- SQLite database:
  - `.paperagent_data/paperagent.db`
- Stored source files:
  - `.paperagent_data/storage/`
- Chroma index:
  - `.paperagent_data/chroma/`
- BM25 index:
  - `.paperagent_data/bm25/`
- PPT outputs and intermediate artifacts:
  - `.paperagent_data/decks/`

## Stage Completion Checklist

- Code changes are scoped to one stage.
- `PROGRESS.md` is updated for that stage.
- Relevant tests pass.
- Changes are committed as one stage-level commit.
