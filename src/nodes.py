from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from langchain_community.llms import Ollama

import os

base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

vllm_llm = Ollama(
    model="llama3.1:8b",
    temperature=0,
    base_url=base_url 
)

embeddings = HuggingFaceEmbeddings(model_name='./local_models/embedder')
vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
cross_encoder = CrossEncoder('./local_models/reranker', local_files_only=True)

def retrieve(state: dict) -> dict:
    question = state["question"]
    docs = retriever.invoke(question)
    return {"raw_docs": [doc.page_content for doc in docs]}

def rerank(state: dict) -> dict:
    question = state["question"]
    raw_docs = state["raw_docs"]
    
    if not raw_docs:
        return {"docs": [], "grade": "bad"}

    pairs = [[question, doc] for doc in raw_docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = list(zip(scores, raw_docs))
    sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    best_docs = [doc for score, doc in sorted_docs[:3] if score > 0.0]
    grade = "good" if best_docs else "bad"
    
    return {"docs": best_docs, "grade": grade}

def generate(state: dict) -> dict:
    question = state["question"]
    docs = state["docs"]
    
    context = "\n\n".join(docs) if docs else "No relevant context found."
    prompt = f"Answer the following question based strictly on the provided context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    response = vllm_llm.invoke(prompt)
    
    return {"generation": response}

def rewrite(state: dict) -> dict:
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1
    
    prompt = f"Rewrite this search query to be more effective for a semantic vector database. Output only the rewritten query. Query: {question}"
    
    response = vllm_llm.invoke(prompt)
    
    return {"question": response.strip(), "retry_count": retry_count}


def reformulate(state: dict) -> dict:
    """Rewrites a follow-up question into a standalone query using chat history.

    If there is no chat history the original question is passed through
    unchanged so the LLM call is skipped entirely.
    """
    chat_history = state.get("chat_history", [])
    question = state["question"]

    if not chat_history:
        return {"question": question}

    history_block = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history
    )

    prompt = (
        "Given the following chat history and the user's latest question, "
        "rewrite the latest question to be a standalone query. "
        "Do not answer the question, just rewrite it.\n\n"
        f"Chat History:\n{history_block}\n\n"
        f"Latest Question: {question}\n\n"
        "Standalone Question:"
    )

    response = vllm_llm.invoke(prompt)
    return {"question": response.strip()}