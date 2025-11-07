# <img src="frontend/src/icon.webp" alt="PersistentDM icon" width="28" style="vertical-align:middle;margin-right:8px;" /> PersistentDM

<img src="screenshots/ShardDemo.webp" alt="ShardDemo" style="max-width:100%;height:auto;margin:16px 0;" />

PersistentDM is a text-based adventure roleplay system that uses an AI "Dungeon Master" to create an interactive story. Unlike simple chatbots, it remembers what happens in the world and uses that knowledge to generate its responses.

It also supports "ingest shards," which let you load pre-built world knowledge (like locations and memories) to start a new game in an established setting.

Key Features

    Persistent Memory: The AI remembers key facts, events, and characters from your story using a vector database.

    World-Aware Replies: The AI checks its memory for relevant facts and NPC details before responding, making the story more consistent.

    Automatic Memory: The system automatically saves important new facts after each turn.

    Spatial Awareness: A built-in location graph helps the AI (and you) keep track of where you are and where you can go.

    Reusable Worlds: You can save and load "shards" of world data (locations, memories) to reuse across different game sessions.

    Memory Search: Query the AI's knowledge base with semantic similarity, literal matching, and recency boosting for debugging and exploration.

Tech Stack

    Backend: FastAPI (Python)

    Frontend: React (with Vite) and Tailwind CSS

    LLM: Runs a local model (Harbinger-24B-GGUF) using llama-cpp-python.

Development Setup

1. Prerequisites

    Python 3.11+

    Node.js 18+

    NVIDIA GPU with 24GB+ VRAM. This is essential for running the 24B parameter model.

    CUDA Toolkit (required for the GPU to work)

2. Steps

    Python

    `python3 -m venv .venv`

    `source .venv/bin/activate`
    `pip install -r requirements.txt`

    Frontend:
    Bash

    `cd frontend && npm install`

3. Model & CUDA

    Build with CUDA: The Python LLM library must be built from source to use your GPU.
    Bash

    `CMAKE_ARGS="-DGGML_CUDA=on"`
    `pip install --no-binary=:all: --no-cache-dir llama-cpp-python==0.3.16`

4. Model installations

    This is currently configurable in llama.py.
    MODEL_PATH = "~/dev/llm/Harbinger-24B-Q5_K_M.gguf"

5. Run the script

   `./scripts/run.sh`

6. Connect via http://localhost:5174/


## Search API

Query the AI's memory database with hybrid ranking that combines semantic similarity, literal substring matching, and recency boosting.

### Endpoint

```
GET /search?q={query}&mode={mode}&k={limit}&types={types}&since={timestamp}
```

### Parameters

- `q` (required): Search query string (1-512 characters)
- `mode` (optional): Search mode - `literal`, `semantic`, or `hybrid` (default: `hybrid`)
- `k` (optional): Maximum results to return (1-100, default: 10)
- `types` (optional): Comma-separated memory types to filter by (e.g., `npc,location`)
- `since` (optional): ISO 8601 timestamp with timezone (e.g., `2025-01-01T00:00:00Z`)

### Response

```json
{
  "query": "blacksmith",
  "mode": "hybrid",
  "k": 5,
  "results": [
    {
      "item_id": "mem_123",
      "type": "npc",
      "text": "The town blacksmith, Rinna...",
      "score": 0.912,
      "explanation": {
        "similarity": 0.83,
        "literal_boost": 0.2,
        "recency_bonus": 0.04,
        "type_bonus": 0.01
      },
      "updated_at": "2025-10-01T12:34:56Z",
      "source": {"shard": "session", "origin": "memory"}
    }
  ]
}
```

### Configuration

Search behavior can be tuned via environment variables:

```bash
# Scoring weights
SEARCH_W_SIM=1.0          # Semantic similarity weight
SEARCH_W_LITERAL=0.2      # Literal substring boost
SEARCH_W_REC=0.15         # Recency bonus weight
SEARCH_W_TYPE=0.05        # Memory type bonus weight

# Recency parameters
SEARCH_HALF_LIFE_HOURS=72 # Recency half-life in hours

# Type-specific bonuses
SEARCH_TYPE_BONUS='{"npc": 0.02, "location": 0.01}'
```

### Examples

```bash
# Hybrid search (default)
curl "http://localhost:8000/search?q=blacksmith"

# Literal substring search
curl "http://localhost:8000/search?q=blacksmith&mode=literal"

# Semantic search only
curl "http://localhost:8000/search?q=blacksmith&mode=semantic"

# Filter by type and recency
curl "http://localhost:8000/search?q=smith&types=npc,location&since=2025-01-01T00:00:00Z"
```
