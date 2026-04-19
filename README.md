# PaperAgent

PaperAgent is a CLI-first assistant for importing academic papers, building hybrid retrieval indexes, chatting about a paper, and generating presentation decks.

## Quick Start

### 1. Prerequisites

PaperAgent currently assumes you already have the `agent` Conda environment mentioned in the project plan.

- Python environment: `conda activate agent`
- Recommended OS for this repo right now: Windows + PowerShell
- Node.js is only required if you want to run the React frontend

### 2. Install Python dependencies

From the project root:

```powershell
conda activate agent
$env:PYTHONPATH="d:\Code\PaperAgent0419\src"
python -m pip install -e .
```

If you only want to run tests and local development commands without editable install, you can keep using `PYTHONPATH`, but `pip install -e .` is the easier long-term setup.

### 3. Create your local config

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

PaperAgent reads configuration from `.env` automatically through `pydantic-settings`.

PDF parsing backend is also configurable:

```env
PAPERAGENT_PDF_BACKEND=datalab
PAPERAGENT_DATALAB_API_KEY=
PAPERAGENT_DATALAB_MODE=balanced
```

Supported `PAPERAGENT_PDF_BACKEND` values:

- `datalab`: hosted PDF-to-Markdown conversion through Datalab, now the default backend and requires `PAPERAGENT_DATALAB_API_KEY`
- `pymupdf`: the built-in local parser, useful as a local fallback when you do not want a hosted parser

If you choose `datalab`, make sure the active Python environment also has the optional `datalab_sdk` package installed.

### 4. Configure your provider and API

PaperAgent does not currently launch an interactive setup wizard on first run.
Instead, the project guides you through configuration with:

- `.env.example`: shows the available config keys
- `README.md`: explains common provider setups
- `paperagent doctor`: checks whether required modules and basic config are present

In other words: the user is guided by docs and the `doctor` command, not by a CLI questionnaire yet.

#### Option A: Fast local testing with no real model API

Use this when you want to verify the whole workflow first.

```env
PAPERAGENT_LLM_BACKEND=mock
PAPERAGENT_EMBEDDING_BACKEND=hash
```

This mode is best for:

- running tests
- validating ingest / indexing / CLI / PPT generation
- avoiding provider setup on day one

Limit:

- the chat answer is mocked, so it is useful for workflow validation, not final answer quality

#### Option B: OpenAI-compatible provider

This is the main intended runtime path for real answers.

You can use any provider that exposes an OpenAI-compatible chat API, for example:

- OpenAI
- DeepSeek
- Moonshot / Kimi
- a self-hosted OpenAI-compatible gateway

Typical `.env` example:

```env
PAPERAGENT_LLM_BACKEND=openai
PAPERAGENT_LLM_MODEL=gpt-4.1-mini
PAPERAGENT_LLM_BASE_URL=https://api.openai.com/v1
PAPERAGENT_LLM_API_KEY=your_api_key_here

PAPERAGENT_EMBEDDING_BACKEND=openai
PAPERAGENT_EMBEDDING_MODEL=text-embedding-3-small
PAPERAGENT_EMBEDDING_BASE_URL=https://api.openai.com/v1
PAPERAGENT_EMBEDDING_API_KEY=your_api_key_here
```

If your provider uses the same key and same base URL for chat and embeddings, you can usually keep both sections aligned.

If you want a simpler first run, you can also start with:

```env
PAPERAGENT_LLM_BACKEND=openai
PAPERAGENT_LLM_MODEL=your_chat_model
PAPERAGENT_LLM_BASE_URL=your_provider_base_url
PAPERAGENT_LLM_API_KEY=your_api_key_here

PAPERAGENT_EMBEDDING_BACKEND=hash
```

That means:

- real LLM for answering
- local deterministic embedding for indexing
- lower setup complexity for the first run

### 5. Verify the environment

Run:

```powershell
python -m paperagent.cli.app doctor
```

What `doctor` helps you confirm:

- required Python modules are installed
- the local data directories can be created
- the configured LLM backend/model values are loaded
- the embedding backend/model values are loaded
- the selected PDF backend is loaded
- whether the optional `datalab_sdk` module and Datalab API key are available

Important:

- `doctor` currently validates configuration presence and runtime modules
- it does not yet make a live network call to verify your API key against the provider

### 6. Import your first paper

From a local PDF:

```powershell
python -m paperagent.cli.app ingest --pdf path\to\paper.pdf
```

If you want to override the configured PDF backend for one run:

```powershell
python -m paperagent.cli.app ingest --pdf path\to\paper.pdf --pdf-backend datalab
```

Or from a URL:

```powershell
python -m paperagent.cli.app ingest --url https://example.com/paper.pdf
```

What happens internally:

- the original PDF is saved under `.paperagent_data/storage/<paper_id>/`
- the PDF is converted into Markdown
- paper metadata is written into SQLite
- chunks are written into SQLite
- Chroma and BM25 indexes are built for retrieval

The Markdown conversion backend is chosen in this order:

1. `ingest --pdf-backend ...` CLI option, if provided
2. `PAPERAGENT_PDF_BACKEND` from `.env`
3. default fallback: `datalab`

### 7. Ask questions about the paper

```powershell
python -m paperagent.cli.app chat ask --paper-id <paper_id> --question "What is the main contribution?"
```

For an interactive loop:

```powershell
python -m paperagent.cli.app chat interactive --paper-id <paper_id>
```

If you do not provide `--paper-id`, PaperAgent switches to general chat mode:

```powershell
python -m paperagent.cli.app chat ask --question "hello"
python -m paperagent.cli.app chat ask --question "Explain the method used in the indexed papers"
```

General chat mode behavior:

- simple greetings like `hello` do not trigger database search
- if the question appears to need indexed paper evidence, the agent can search across imported papers
- if you want to limit retrieval to one paper, pass `--paper-id`

### 8. Generate a PPT

```powershell
python -m paperagent.cli.app ppt generate --paper-id <paper_id>
```

This creates:

- a `.pptx` file
- a `deck.json` intermediate outline for inspection and later frontend reuse

### 9. Run the API and frontend

Start the FastAPI backend:

```powershell
python -m paperagent.cli.app serve-api
```

Then start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

The first implementation favors a clear code structure and testability so it can later power a FastAPI + React interface.

## Environment

The project is designed to run inside the existing `conda activate agent` environment.

- For local tests and offline development, the default embedding backend is `hash`.
- For real LLM answers, set `PAPERAGENT_LLM_API_KEY` and optionally `PAPERAGENT_LLM_BASE_URL`.
- For deterministic test runs, set `PAPERAGENT_LLM_BACKEND=mock`.

## Provider Notes

### Which provider should a new user choose?

Recommended starting path:

1. First run: `PAPERAGENT_LLM_BACKEND=mock` and `PAPERAGENT_EMBEDDING_BACKEND=hash`
2. After the workflow is confirmed: switch the LLM to an OpenAI-compatible provider
3. Finally, if needed, switch embeddings from `hash` to provider-backed embeddings

This reduces the number of things that can fail at once.

### What config keys matter most?

- `PAPERAGENT_LLM_BACKEND`: currently `mock` or `openai`
- `PAPERAGENT_LLM_MODEL`: chat model name
- `PAPERAGENT_LLM_BASE_URL`: provider base URL for OpenAI-compatible APIs
- `PAPERAGENT_LLM_API_KEY`: provider API key
- `PAPERAGENT_EMBEDDING_BACKEND`: `hash` or `openai`
- `PAPERAGENT_EMBEDDING_MODEL`: embedding model name
- `PAPERAGENT_EMBEDDING_BASE_URL`: embedding endpoint base URL
- `PAPERAGENT_EMBEDDING_API_KEY`: embedding API key

### Does the project guide the user to configure the provider?

Yes, but currently in a documentation-driven way.

Current guidance path:

- copy `.env.example` to `.env`
- fill provider values based on the examples above
- run `paperagent doctor`

Not implemented yet:

- an interactive first-run setup wizard
- automatic browser-based provider login
- live API credential validation inside `doctor`

## Development Commands

```powershell
$env:PYTHONPATH="d:\Code\PaperAgent0419\src"
python -m pytest -q
python -m paperagent.cli.app doctor
python -m paperagent.cli.app ingest --pdf path\to\paper.pdf
python -m paperagent.cli.app chat ask --paper-id <paper_id> --question "Explain the method"
python -m paperagent.cli.app chat ask --question "hello"
python -m paperagent.cli.app ppt generate --paper-id <paper_id>
python -m paperagent.cli.app serve-api
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

## Troubleshooting

### `doctor` shows config but chat still fails

That usually means one of these:

- the API key is invalid
- the base URL is wrong
- the model name is not supported by your provider
- the provider supports chat but not the embedding model you configured

For the fastest recovery:

1. set `PAPERAGENT_LLM_BACKEND=mock`
2. set `PAPERAGENT_EMBEDDING_BACKEND=hash`
3. confirm the local workflow works
4. switch one provider setting at a time back to the real API

### I only want to test ingest, retrieval, and PPT locally

Use:

```env
PAPERAGENT_LLM_BACKEND=mock
PAPERAGENT_EMBEDDING_BACKEND=hash
```

That is the simplest zero-to-running configuration today.
