# Agentic RAG for High-Performance Local Workstations

An edge-optimized, local-first Retrieval-Augmented Generation (RAG) pipeline built for environments with strict VRAM constraints (e.g., 4GB mobile GPUs). 

This system uses a LangGraph state machine to intelligently route, retrieve, and evaluate queries against a local ChromaDB vector store, powered by an Ollama-managed Llama 3.1 8B engine.

## 🧠 Architecture Overview
* **Orchestration:** LangGraph (Agentic state machine with fallback/rewrite logic)
* **Backend:** FastAPI (Async API wrapper)
* **LLM Engine:** Llama 3.1 8B via Ollama (Host-managed for optimal VRAM offloading)
* **Embedding Model:** `all-MiniLM-L6-v2` (Local HuggingFace pipeline)
* **Reranking:** `ms-marco-MiniLM-L-12-v2` Cross-Encoder (Boosts top-K precision)
* **Frontend:** Streamlit (Chat UI)

## 🛠️ Hardware & Engineering Optimizations
This project was specifically engineered to run on an **NVIDIA Quadro T2000 (4GB VRAM)**. Standard vLLM/Transformer serving architectures fail under these constraints due to KV-cache memory allocation. 

**Solutions Implemented:**
1. **Ollama Split-Execution:** Leveraged `llama.cpp` under the hood to load critical model layers onto the 4GB GPU while seamlessly offloading the remainder to system RAM.
2. **Local-First Embedding:** Isolated the embedding and reranking models from the generative LLM to prevent VRAM fragmentation.
3. **Containerized Microservices:** Dockerized the FastAPI backend and Streamlit frontend while mapping host volumes, ensuring the container remains extremely lightweight (~200MB) without baking in model weights.

## 🚀 Getting Started

### Prerequisites
1. Docker & Docker Compose
2. [Ollama](https://ollama.com/) installed on the host machine.
3. Pull the required model: `ollama run llama3.1:8b`

### Run the Stack
```bash
# 1. Export local embedding models (one-time setup)
uv run python scripts/export_models.py

# 2. Spin up the API and UI
docker compose up --build -d