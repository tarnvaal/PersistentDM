# <img src="frontend/src/icon.webp" alt="PersistentDM icon" width="28" style="vertical-align:middle;margin-right:8px;" /> PersistentDM

A text-based interactive fiction system that uses large language models to generate persistent world states and narrative responses.

Supports reusable ingest shards: persist world knowledge (locations + memories) across sessions and load them on demand.

## Overview

PersistentDM implements an AI dungeon master for text-based role-playing games. The system maintains persistent world state through a vector-based memory system and provides conversational interfaces for game interactions.

### Key Features

- Persistent world memory with vector similarity search
- World-aware replies: injects relevant World Facts and NPC Cards into the LLM prompt
- Automatic memory extraction from each conversation turn (stores significant facts/NPC snapshots)
- FastAPI backend with automatic CORS handling
- React frontend with Tailwind CSS
- Llama.cpp integration for local LLM inference
- Automated development environment setup
 - Optional debug fields in responses for the dev UI (context snippets and relevance scores)
 - Location graph (locations + exits) injected into prompts for spatial grounding
 - LLM-guided movement detection and dynamic graph growth (confidence-gated)
 - Ingest shards: persist reusable world subgraphs and memories on disk; list/name/load/delete via API/UI

## Architecture

### Backend

- **Framework**: FastAPI with automatic API documentation
- **LLM Integration**: llama-cpp-python with CUDA acceleration
- **World Memory**: Vector-based memory system using embeddings for similarity search
 - **World Graph**: In-memory location graph (locations + exits) used to ground navigation; included in prompt context
- **Conversation Orchestration**: `ConversationService` composes `Chatter` and `WorldMemory`, retrieves relevant memories and NPC snapshots, formats World Facts + NPC Cards, injects them into the LLM call, and persists new memories
- **Model**: Harbinger-24B-GGUF (quantized for ~23GB VRAM usage)
 - **Ingest Layer (Shards)**: Persistent, reusable world shards (locations + memories) stored under `data/ingests` (configurable via `INGESTS_DIR`). Loaded at startup and included in retrieval without mutating session history.

### Frontend

- **Framework**: React 18 with Vite build system
- **Styling**: Tailwind CSS 4.x
- **Development**: Hot reload with concurrent backend proxying

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- **NVIDIA GPU with 24GB+ VRAM (required for model inference)**
- **CUDA toolkit (required for GPU acceleration)**

### Quick Setup

Run the automated setup script:

```bash
./scripts/setup.sh
```

This creates a Python virtual environment, installs dependencies, and sets up the frontend.

### Manual Setup

1. Create Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Install frontend dependencies:
   ```bash
   cd frontend && npm install
   ```

### Model Configuration

Download the required model file: [Harbinger-24B-GGUF](https://huggingface.co/LatitudeGames/Harbinger-24B-GGUF)

Set the model path:
```bash
export MODEL_PATH=/absolute/path/to/Harbinger-24B-Q5_K_M.gguf
```
Optionally set the maximum model context window (default 16384 tokens):
```bash
export MAX_CONTEXT_TOKENS=16384
```


Default path: `~/dev/llm/Harbinger-24B-Q5_K_M.gguf`

### CUDA Installation

**CRITICAL**: The llama-cpp-python package must be built from source with CUDA support enabled. Do NOT install the CPU-only wheel from PyPI.

```bash
CMAKE_ARGS="-DGGML_CUDA=on" \
pip install --no-binary=:all: --no-cache-dir llama-cpp-python==0.3.16
```

GPU acceleration is required for practical inference speeds with the 24B parameter model.

## Running the Application

### Development Mode

Start both backend and frontend simultaneously:

```bash
npm run dev
```

This runs:
- Backend API on `http://localhost:8000`
- Frontend dev server on `http://localhost:5173`

### Individual Services

Start backend only:
```bash
npm run api
# or
uvicorn backend.app.main:app --reload
```

Start frontend only:
```bash
npm run web
# or
cd frontend && npm run dev
```

### Production Build

```bash
cd frontend && npm run build
cd frontend && npm run preview
```

## API Endpoints

- `GET /health` - Health check
- `POST /chat` - Send chat message
- `POST /chat/clear` - Clear conversation history
- `POST /ingest/upload` - Upload raw text to be processed
- `GET /ingest/stream?id=...` - Server-Sent Events stream for chunked ingestion
 - `GET /ingest/list` - List persisted ingest shards (id, name, counts, size)
 - `PUT /ingest/shard/{ingest_id}/name` - Rename a shard
 - `POST /ingest/shard/{ingest_id}/load` - Load a shard (locations + memories) into memory
 - `DELETE /ingest/shard/{ingest_id}` - Delete a shard from disk

See `requests.rest` for example API calls.

### Request Flow (Chat)
- Frontend POSTs `{ "message": string }` to `/chat`
- Backend router delegates to `ConversationService.handle_user_message`
- Service retrieves relevant memories (weighted by similarity/recency/type) and NPC snapshots
- Service formats World Facts, NPC Cards, and the current Location Context (location + exits) and injects them as a transient system message
- `Chatter.chat` generates the DM reply; the service analyzes the turn and stores new durable memories when confidence is high; the service also uses the LLM to conservatively infer player movement and propose new locations/exits to grow the graph
 - Response returns `{ "reply": string, "context"?: string | null, "relevance"?: object | null }`
  - `context` is the exact world context injected into the model for that turn
  - `relevance` contains lightweight memory/NPC relevance info
  - `relevance.saved` (when present) summarizes what memory was persisted this turn (type, summary, entities, confidence)
  - These fields are intended for development/debugging and are shown in the dev UI

### Request Flow (Ingest pasted text)
- Frontend uploads the pasted text via `POST /ingest/upload` and receives `{ id, totalWords, totalLines }`.
- Frontend opens an `EventSource` to `GET /ingest/stream?id=<id>`.
- The backend iterates over the text in sliding windows (window/stride by words) and emits SSE events:
  - `info` — initial run parameters: approx tokens, window/stride (words), total steps, checkpoint interval
  - `progress` — `{ step, totalSteps, consumedWords, consumedLines, progress }`
  - `saved` — when a durable memory is persisted; includes summary, type, entities, optional NPC payload, confidence
  - `checkpoint` — lightweight summaries at periodic checkpoints (not persisted)
  - `hygiene` — results of location-graph hygiene (merged/pruned counts)
  - `done` — final stats: words, lines, steps
- Cancellation: closing the EventSource (e.g., clicking the trashcan/clear button) stops processing at the start of the next chunk; `POST /chat/clear` also resets memories, NPC index, and the location graph.

On completion, the current ingest run is persisted as a shard on disk so it survives restarts. Persisted shards can be listed, renamed, loaded, and deleted via the endpoints above; when loaded, their locations and memories are merged into the retrieval context without altering session history.

Example (terminal):
```bash
# 1) Upload text
curl -sX POST http://localhost:8000/ingest/upload \
  -H 'Content-Type: application/json' \
  -d '{"text":"Your pasted text here"}'

# 2) Stream SSE events (using curl --no-buffer)
curl --no-buffer "http://localhost:8000/ingest/stream?id=<id-from-step-1>"
```

## Testing

Run the test suite:

```bash
npm test
# or
pytest
```

Tests are located in `backend/tests/` and cover:
- API endpoints
- Message handling
- History management

## Project Structure

```
PersistentDM/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── dependencies.py      # Dependency injection
│   │   ├── routers/
│   │   │   └── chat.py          # Chat API endpoints
│   │   ├── utility/
│   │   │   ├── embeddings.py    # Vector embeddings
│   │   │   └── llama.py         # LLM integration
│   │   └── world/
│   │       ├── memory.py              # World state memory (memories + NPC snapshots)
│   │       ├── context_builder.py     # Weighted retrieval and formatting (World Facts, NPC Cards)
│   │       ├── conversation_service.py# Orchestrates context injection + memory extraction
│   │       ├── memory_utils.py        # Helpers for sanitizing entities
│   │       ├── queries.py             # Memory/planner prompts
│   │       └── summarizer.py          # Memory summarization
│   └── tests/                   # Test suite
├── frontend/
│   ├── src/                     # React application
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── scripts/
│   ├── setup.sh                 # Development setup
│   └── run.sh                   # Development runner
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Python project config
└── package.json                 # Root package scripts
```

## Code Quality

### Python

- **Formatting**: Black (88 character line length)
- **Linting**: Ruff
- **Type checking**: Pyright

Format code:
```bash
black .
ruff check . --fix
```

### JavaScript/TypeScript

- **Formatting**: Prettier (via Vite)
- **Linting**: ESLint (via Vite)

### Development Workflow

- Use the setup script for initial environment configuration
- Run tests before committing
- Follow existing code style and patterns
- Update documentation for API changes

## Dependencies

### Python

Key dependencies in `requirements.txt`:
- fastapi: Web framework
- llama-cpp-python: LLM inference
- numpy: Numerical operations
- sentence-transformers: Text embeddings

### Node.js

Key dependencies in `frontend/package.json`:
- react: UI framework
- vite: Build tool
- tailwindcss: CSS framework

## Environment Variables

- `MODEL_PATH`: Path to GGUF model file
- `FRONTEND_PORT`: Frontend development port (default: 5173)
- `VITE_API_BASE_URL`: Frontend override for the backend API URL (e.g., `http://localhost:8000`). If not set, the frontend infers the host and uses port 8000.
- `MIN_FREE_VRAM_MIB`: Minimum free VRAM (MiB) required before attempting model load (default: 23400). Prevents accidental CPU fallback; all layers remain on GPU.
- `LLAMA_INIT_WAIT_SECS`: How long other requests will wait for the singleton model to finish initializing (default: 300).
- `PDM_DEBUG_ERRORS`: If "1" (default), API 500 responses include error details; set to "0" in production to hide internal messages.
- `INGESTS_DIR`: Absolute or relative directory where ingest shards are persisted and loaded from (default: `data/ingests` under the project root).

## Troubleshooting

### CUDA Issues

Ensure CUDA is properly installed and the llama-cpp-python package is built with CUDA support. Check GPU memory usage with `nvidia-smi`.

### Port Conflicts

The development scripts automatically handle common Vite port ranges (5173-5180). Adjust `FRONTEND_PORT` if needed.

### Model Loading

Model loading can take several minutes on first startup. The app preloads the model asynchronously at startup; the `/health` endpoint responds immediately. The first chat request may take longer while the model finishes loading. The model is a singleton in-process; concurrent requests will wait for initialization up to `LLAMA_INIT_WAIT_SECS`.

### State & Persistence
- World memory and NPC snapshots live in-process only (not persisted). They are cleared by `POST /chat/clear`.
- This project assumes a single-process, single-thread server (default Uvicorn). Running multiple threads/workers requires external persistence and coordination; current in-memory structures are not thread-safe.

### Security/Operations Notes
- The `POST /chat/clear` endpoint resets all in-memory state and is meant for development only. Do not expose it publicly without authentication.
