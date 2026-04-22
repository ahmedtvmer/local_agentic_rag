# Agentic RAG — Fully Local, Self-Correcting LLM Pipeline

A production-grade, fully local Retrieval-Augmented Generation system engineered from scratch to run on **entry-level GPUs** (as low as 4 GB VRAM). The entire stack — inference, embeddings, reranking, vector search, and a chat UI — runs on your machine with zero cloud dependencies.

---

## Why This Exists

Standard RAG tutorials assume unlimited GPU memory and cloud API keys. Real workstations — especially laptops with mobile Quadro or GTX-class GPUs — don't have that luxury. This project solves three hard problems simultaneously:

1. **Conversational Amnesia** — Most RAG pipelines treat every question in isolation. Ask "tell me about ChatGPT" then "when did it launch?" and the system has no idea what "it" refers to. Solved with a dedicated **Reformulate** node.
2. **Silent Retrieval Failures** — If the vector database returns irrelevant chunks, a naive pipeline hallucinates an answer anyway. Solved with a **Cross-Encoder Reranker** + **Self-Correcting Retry Loop**.
3. **GPU Out-of-Memory Crashes** — Loading a 8B parameter model alongside embedding models and a cross-encoder on a 4 GB GPU is impossible with standard frameworks. Solved with **Ollama's split-execution** architecture.

---

## Architecture

```
┌───────────┐     POST /ask      ┌─────────────────────────────────────────────────┐
│ Streamlit │───────────────────▶│                  FastAPI (api.py)                │
│   Chat UI │◀───────────────────│  Pydantic schemas · Health check · Error guard   │
│ :8501     │   JSON response    └────────────────────┬────────────────────────────┘
└───────────┘                                         │
                                                      ▼
                              ┌─────────────────────────────────────────────────────┐
                              │           LangGraph State Machine (run_agent.py)     │
                              │                                                     │
                              │  ┌─────────────┐   ┌──────────┐   ┌──────────────┐  │
                              │  │ Reformulate  │──▶│ Retrieve │──▶│   Rerank     │  │
                              │  │ (chat hist.) │   │ (Chroma) │   │(CrossEncoder)│  │
                              │  └─────────────┘   └──────────┘   └──────┬───────┘  │
                              │                         ▲                │           │
                              │                         │          grade │           │
                              │                    ┌────┴────┐    ┌──────▼───────┐   │
                              │                    │ Rewrite │◀───│  Route       │   │
                              │                    │ (query) │ bad│  Evaluation  │   │
                              │                    └─────────┘    └──────┬───────┘   │
                              │                                    good │           │
                              │                                  ┌──────▼───────┐   │
                              │                                  │  Generate    │   │
                              │                                  │  (Llama 3.1) │   │
                              │                                  └──────────────┘   │
                              └─────────────────────────────────────────────────────┘
                                                      │
                              ┌────────────────────────┼────────────────────────┐
                              │                        ▼                        │
                              │           Ollama (host, port 11434)             │
                              │           Llama 3.1 8B · llama.cpp             │
                              │           GPU layers ↔ RAM offload             │
                              └─────────────────────────────────────────────────┘
```

---

## Feature Breakdown

### 1. Conversational Memory via Query Reformulation
**File:** `src/nodes.py → reformulate()`

The first node in the graph. Before any retrieval happens, the user's question and the full chat history are passed to Llama 3.1 with a strict instruction: *"Rewrite the latest question to be a standalone query. Do not answer the question."* This converts ambiguous follow-ups like "when did it launch?" into "When was ChatGPT officially released?" — the reformulated query is what hits the embedding model and vector database.

If there is no chat history (first message), the LLM call is skipped entirely to save latency.

### 2. Semantic Data Ingestion Pipeline
**File:** `scripts/ingest_data.py`

A dual-source ingestion system:
- **Web Scraping:** Fetches Wikipedia articles using BeautifulSoup, targeting the `mw-content-text` container. Strips citation superscripts (`<sup class="reference">`) and reference lists to eliminate noise before chunking.
- **Local Documents:** Reads `.txt` files from the `raw_documents/` directory.

Both sources are split using `RecursiveCharacterTextSplitter` (1000-char chunks, 100-char overlap, paragraph-aware separators) and stored in ChromaDB with source metadata.

### 3. Vector Retrieval (ChromaDB)
**File:** `src/nodes.py → retrieve()`

Queries ChromaDB with the reformulated question using the `all-MiniLM-L6-v2` embedding model (pre-exported to `local_models/embedder/`). Retrieves the top 10 candidate documents for maximum recall before precision filtering.

### 4. Cross-Encoder Reranking
**File:** `src/nodes.py → rerank()`

The top-10 retrieved documents are passed through an `ms-marco-MiniLM-L-12-v2` Cross-Encoder, which scores each query-document pair jointly rather than relying on cosine similarity alone. Only the top 3 documents with a relevance score above 0.0 survive. If no document passes the threshold, the grade is set to `"bad"`, triggering the self-correction loop.

### 5. Self-Correcting Retry Loop (Circuit Breaker)
**File:** `src/run_agent.py → route_evaluation()`

When the reranker grades documents as `"bad"`, the graph routes to a **Rewrite** node that asks Llama 3.1 to reformulate the search query for better semantic vector search. The rewritten query re-enters the retrieve → rerank cycle. A hard **circuit breaker** at 3 retries forces generation with whatever context is available, preventing infinite loops.

### 6. Grounded Generation
**File:** `src/nodes.py → generate()`

The final generation step concatenates the surviving documents into a context block and prompts Llama 3.1 with strict instructions: *"Answer the following question based strictly on the provided context."* This grounds the model's output in retrieved facts rather than parametric knowledge.

### 7. FastAPI Backend with Pydantic Schemas
**File:** `api.py`

- **`POST /ask`** — Accepts `{ question, chat_history }`, invokes the LangGraph state machine, returns `{ answer, retry_count, execution_time_ms }`. Wrapped in `try/except` with `HTTPException(500)` on any graph failure.
- **`GET /health`** — Liveness probe for Docker healthchecks. Returns `{ "status": "healthy" }`.
- All request/response models are strict Pydantic schemas with field validation.

### 8. Streamlit Chat Frontend
**File:** `frontend.py`

A conversational chat interface that maintains `st.session_state.messages` across turns. Sends the full chat history with every request so the reformulate node can resolve pronouns and references. Displays the generated answer with execution time. The `API_URL` is read from an environment variable, resolving correctly both locally (`localhost:8080`) and inside Docker (`http://api:8080/ask`).

### 9. LLM-as-a-Judge Evaluation Suite
**File:** `scripts/evaluate_pipeline.py`

Automated testing pipeline that:
1. Sends predefined test cases to the running FastAPI server via `httpx`.
2. Passes each answer + expected fact to Llama 3.1 acting as a factual-accuracy judge (1–5 scale).
3. Reports average score and average latency.
4. Exits with code 1 if the average score falls below 3.0 (CI/CD-friendly).

### 10. Dockerized Microservices
**Files:** `Dockerfile`, `docker-compose.yml`

Two-service architecture:
- **`api`** — FastAPI backend with a Python-based healthcheck (no `curl` dependency in slim images). Volumes mount `local_models/` and `db/` from the host so model weights and vector data are never baked into the image (~200 MB image).
- **`frontend`** — Streamlit UI that only starts after the API passes its healthcheck (`depends_on: condition: service_healthy`).

Both containers reach the host's Ollama instance via `host.docker.internal:host-gateway`.

### 11. Local Model Export
**File:** `scripts/export_models.py`

One-time setup script that downloads `all-MiniLM-L6-v2` (embedder) and `ms-marco-MiniLM-L-12-v2` (cross-encoder) from HuggingFace and saves them to `local_models/`. This ensures fully offline operation after initial setup — no network calls at inference time.

---

## Handling Entry-Level GPUs

This pipeline was engineered to run on an **NVIDIA Quadro T2000 with 4 GB VRAM** — a mobile workstation GPU that most ML frameworks consider insufficient for 8B parameter models.

### The Problem
Standard serving frameworks (vLLM, TGI, vanilla Transformers) attempt to load the entire model into GPU memory. An 8B model in FP16 requires ~16 GB VRAM minimum. Even with 4-bit quantization (~4.5 GB), the KV cache, embedding models, and cross-encoder compete for the remaining memory, causing OOM crashes.

### The Solution: Ollama + llama.cpp Split-Execution

Instead of a Python-native serving framework, this project uses **Ollama**, which wraps `llama.cpp` under the hood:

1. **Automatic Layer Splitting** — Ollama profiles the available VRAM and loads as many transformer layers as will fit onto the GPU. The remaining layers execute on system RAM via CPU. On a 4 GB GPU, roughly 50–60% of Llama 3.1 8B's layers run on the GPU, with the rest offloaded transparently.

2. **Isolated Model Memory** — The embedding model (`all-MiniLM-L6-v2`, ~90 MB) and cross-encoder (`ms-marco-MiniLM-L-12-v2`, ~130 MB) are loaded via `sentence-transformers` in the FastAPI container's Python process, completely isolated from Ollama's VRAM allocation. No fragmentation, no contention.

3. **Host-Level Serving** — Ollama runs on the host machine (not inside Docker), giving it direct access to GPU drivers and VRAM. The Docker containers communicate with it over HTTP (`host.docker.internal:11434`), keeping the container images lightweight.

### Performance Characteristics on 4 GB VRAM

| Metric | Typical Value |
|---|---|
| First-token latency | ~2–4 seconds |
| Token generation speed | ~8–15 tokens/sec |
| Full pipeline (retrieve + rerank + generate) | ~10–30 seconds |
| Embedding model memory | ~90 MB (Python process) |
| Cross-encoder memory | ~130 MB (Python process) |
| Docker image size | ~200 MB (no model weights) |

> **Note:** If you have more VRAM (8 GB+), Ollama will automatically load more layers onto the GPU, proportionally increasing inference speed with zero configuration changes.

---

## Project Structure

```
RAG/
├── api.py                          # FastAPI backend (POST /ask, GET /health)
├── frontend.py                     # Streamlit chat UI
├── Dockerfile                      # Multi-stage build with uv + healthcheck
├── docker-compose.yml              # API + Frontend services
├── pyproject.toml                  # Python dependencies (uv-managed)
├── uv.lock                         # Locked dependency graph
├── .env                            # Runtime config (OLLAMA_HOST, API_URL)
├── .env example                    # Template for environment variables
├── .dockerignore                   # Keeps build context lean
├── .gitignore                      # Excludes db/, models/, secrets
├── src/
│   ├── __init__.py
│   ├── state.py                    # GraphState TypedDict definition
│   ├── nodes.py                    # All LangGraph nodes (reformulate, retrieve, rerank, generate, rewrite)
│   └── run_agent.py                # Graph wiring + circuit breaker routing
├── scripts/
│   ├── ingest_data.py              # Web scraping + local doc ingestion → ChromaDB
│   ├── export_models.py            # Downloads embedder + reranker to local_models/
│   └── evaluate_pipeline.py        # LLM-as-a-Judge automated evaluation
├── local_models/                   # Pre-exported HuggingFace models (git-ignored)
│   ├── embedder/                   # all-MiniLM-L6-v2
│   └── reranker/                   # ms-marco-MiniLM-L-12-v2
├── db/                             # ChromaDB persistent storage (git-ignored)
└── raw_documents/                  # Drop .txt files here for ingestion (git-ignored)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Inference | Llama 3.1 8B via Ollama (llama.cpp) |
| Orchestration | LangGraph (deterministic state machine) |
| Backend API | FastAPI + Pydantic + Uvicorn |
| Vector Database | ChromaDB (persistent, local) |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Reranking | `ms-marco-MiniLM-L-12-v2` Cross-Encoder |
| Frontend | Streamlit |
| Data Ingestion | BeautifulSoup4 + RecursiveCharacterTextSplitter |
| Containerization | Docker + Docker Compose |
| Package Management | uv (fast Python package manager) |
| Evaluation | LLM-as-a-Judge via Ollama |

---

## Getting Started

### Prerequisites

1. **Docker & Docker Compose** installed.
2. **[Ollama](https://ollama.com/)** installed on the host machine.
3. **[uv](https://docs.astral.sh/uv/)** installed (for running scripts outside Docker).
4. Pull the required model:
   ```bash
   ollama pull llama3.1:8b
   ```

### One-Time Setup

```bash
# 1. Export embedding + reranking models locally
uv run python scripts/export_models.py

# 2. Ingest your data (edit URLs/documents in the script first)
uv run python scripts/ingest_data.py
```

### Run the Stack

```bash
# Start Ollama on the host (if not already running as a service)
ollama serve

# Build and launch the API + Frontend
docker compose up --build -d
```

- **Chat UI:** [http://localhost:8501](http://localhost:8501)
- **API docs:** [http://localhost:8080/docs](http://localhost:8080/docs)
- **Health check:** [http://localhost:8080/health](http://localhost:8080/health)

### Run Evaluation

```bash
# Requires both Ollama and the API to be running
OLLAMA_HOST=http://localhost:11434 API_URL=http://localhost:8080/ask uv run python scripts/evaluate_pipeline.py
```

---

## Environment Variables

| Variable | Description | Example |
|---|---|---|
| `OLLAMA_HOST` | URL of the Ollama server | `http://localhost:11434` |
| `API_URL` | URL of the FastAPI `/ask` endpoint | `http://localhost:8080/ask` |

Inside Docker, these are set automatically via `docker-compose.yml`. For local development, copy `.env example` to `.env` and fill in the values.