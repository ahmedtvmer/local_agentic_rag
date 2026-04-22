# Local Agentic RAG: Self-Correcting LLM Architecture on Constrained Hardware

A fully local, self-correcting Retrieval-Augmented Generation (RAG) pipeline built from scratch. This project abandons black-box agent wrappers in favor of a deterministic state machine, running a heavily optimized inference engine designed to fit entirely within a single 16GB T4 GPU.

## System Architecture

This pipeline is engineered to solve the three primary failure modes of standard RAG systems: silent hallucinations, wording mismatches, and hardware out-of-memory (OOM) crashes.

### Key Engineering Features

* **Self-Correcting Routing (State Machine):** Built with LangGraph, the agent evaluates its own answers using a strict `reflect` node. If the retrieved documents lack the answer, it triggers a fallback loop to a `rewrite_query` node, optimizing its own search terms and querying the vector database again until it succeeds or hits a hard circuit breaker.
* **Maximum Recall Retrieval (Ensemble):** Implements an advanced retrieval layer utilizing Multi-Query expansion and Hypothetical Document Embeddings (HyDE). The system queries ChromaDB simultaneously with the original question, three generated variants, and a hallucinated answer shape, deduplicating the results to cast a massive semantic net.
* **Precision Reranking (Cross-Encoder):** Passes the deduplicated, high-recall document batch through an `ms-marco-MiniLM-L-12-v2` Cross-Encoder to guarantee only the most mathematically relevant chunks are injected into the final LLM prompt.
* **Semantic ETL Pipeline:** Replaces naive web scraping with a robust `BeautifulSoup` data ingestion script that targets specific DOM containers, sanitizes HTML boilerplate and citation noise, and chunks documents strictly by paragraph boundaries to preserve semantic meaning.
* **Hardware-Optimized Inference:** Runs the ungated Qwen 2.5 3B Instruct model locally via vLLM. Engineered specifically for standard Colab T4 hardware by capping GPU memory utilization at 0.7 and disabling CUDA graphs (`--enforce-eager`) to leave exact VRAM clearance for the embedding models and KV cache.

## Tech Stack

* **Inference Engine:** vLLM (Local), Qwen 2.5 3B Instruct
* **Orchestration & State Management:** LangGraph, LangChain Core
* **Vector Database:** ChromaDB
* **Embeddings & Reranking:** HuggingFace (`all-MiniLM-L6-v2`), SentenceTransformers (Cross-Encoder)
* **Data Ingestion:** BeautifulSoup4, Requests

## Hardware Requirements

To run this pipeline locally, you must meet the following constraints:

* **GPU:** 1x NVIDIA T4 (or any equivalent GPU with at least 14.5 GB usable VRAM)
* **Memory:** 16 GB System RAM