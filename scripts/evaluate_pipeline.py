"""
Evaluation Suite – LLM-as-a-Judge
=================================
Automated evaluation of the RAG pipeline using the local vLLM engine
(Qwen 2.5 3B Instruct) as a factual-accuracy judge.

Prerequisites (both must be running):
    1. vLLM server on port 8000
    2. FastAPI server on port 8080
       uvicorn api:api --host 0.0.0.0 --port 8080

Usage:
    python -m scripts.evaluate_pipeline
"""

import re
import sys

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ─── Configuration ───────────────────────────────────────────────────────────

RAG_API_URL = "http://localhost:8080/ask"
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
REQUEST_TIMEOUT = 120.0

# ─── Test Cases ──────────────────────────────────────────────────────────────

TEST_CASES: list[dict[str, str]] = [
    {
        "question": "What is retrieval-augmented generation?",
        "expected_fact": (
            "Retrieval-augmented generation (RAG) is a technique that combines "
            "information retrieval with a generative language model to produce "
            "answers grounded in retrieved documents."
        ),
    },
    {
        "question": "What is a vector database used for in RAG systems?",
        "expected_fact": (
            "A vector database stores document embeddings and enables fast "
            "similarity search so a RAG system can retrieve the most "
            "semantically relevant passages for a given query."
        ),
    },
    {
        "question": "What is the purpose of a cross-encoder reranker?",
        "expected_fact": (
            "A cross-encoder reranker takes a query-document pair as joint "
            "input and produces a relevance score, allowing the system to "
            "reorder retrieved documents by true semantic relevance rather "
            "than relying solely on embedding similarity."
        ),
    },
]

# ─── Judge LLM ───────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are a strict factual-accuracy judge.

You will receive:
- QUESTION: the user's original question
- EXPECTED FACT: the ground-truth fact
- GENERATED ANSWER: the answer produced by an AI system

Your task: rate the GENERATED ANSWER on factual accuracy compared to the \
EXPECTED FACT.

Use this scale:
1 = completely wrong or irrelevant
2 = mostly wrong with a minor correct detail
3 = partially correct but missing key facts
4 = mostly correct with minor inaccuracies
5 = fully correct and consistent with the expected fact

Rules:
- Output ONLY a single integer (1, 2, 3, 4, or 5).
- Do NOT output any other text, explanation, or punctuation.
"""


def build_judge_prompt(question: str, expected_fact: str, answer: str) -> list:
    """Construct the message list for the judge LLM."""
    human_content = (
        f"QUESTION: {question}\n\n"
        f"EXPECTED FACT: {expected_fact}\n\n"
        f"GENERATED ANSWER: {answer}"
    )
    return [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]


def parse_score(raw: str) -> int | None:
    """Extract the first integer 1-5 from the judge's response."""
    match = re.search(r"[1-5]", raw.strip())
    return int(match.group()) if match else None


# ─── Main Evaluation Loop ───────────────────────────────────────────────────

def main() -> None:
    judge = ChatOpenAI(
        base_url=VLLM_BASE_URL,
        model=VLLM_MODEL,
        api_key="empty",
        temperature=0.0,
        max_tokens=4,
    )

    scores: list[int] = []
    latencies: list[float] = []
    results: list[dict] = []

    print("=" * 70)
    print("RAG Pipeline Evaluation Suite")
    print("=" * 70)

    for idx, case in enumerate(TEST_CASES, start=1):
        question = case["question"]
        expected_fact = case["expected_fact"]

        print(f"\n[Test {idx}/{len(TEST_CASES)}] {question}")
        print("-" * 60)

        # --- Hit the RAG API --------------------------------------------------
        try:
            response = httpx.post(
                RAG_API_URL,
                json={"question": question},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"  ✗ API request failed: {exc}")
            results.append({"question": question, "error": str(exc)})
            continue

        payload = response.json()
        answer = payload.get("answer", "")
        exec_time = payload.get("execution_time_ms", 0.0)
        latencies.append(exec_time)

        print(f"  Answer  : {answer[:120]}{'…' if len(answer) > 120 else ''}")
        print(f"  Latency : {exec_time:.1f} ms")

        # --- Judge the answer -------------------------------------------------
        try:
            messages = build_judge_prompt(question, expected_fact, answer)
            judge_response = judge.invoke(messages)
            raw_score = judge_response.content
            score = parse_score(raw_score)
        except Exception as exc:
            print(f"  ✗ Judge LLM failed: {exc}")
            results.append({"question": question, "error": str(exc)})
            continue

        if score is None:
            print(f"  ✗ Could not parse score from judge output: '{raw_score}'")
            results.append({"question": question, "raw_judge": raw_score})
            continue

        scores.append(score)
        print(f"  Score   : {score}/5")

        results.append({
            "question": question,
            "answer": answer,
            "expected_fact": expected_fact,
            "score": score,
            "latency_ms": exec_time,
        })

    # --- Summary Report -------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"  Average Score   : {avg_score:.2f} / 5")
    else:
        avg_score = 0.0
        print("  Average Score   : N/A (no successful evaluations)")

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"  Average Latency : {avg_latency:.1f} ms")
    else:
        avg_latency = 0.0
        print("  Average Latency : N/A (no successful API calls)")

    total = len(TEST_CASES)
    passed = len(scores)
    failed = total - passed
    print(f"  Tests Run       : {total}")
    print(f"  Scored          : {passed}")
    print(f"  Failed/Skipped  : {failed}")
    print("=" * 70)

    # Exit with non-zero if average score is below threshold
    if avg_score < 3.0:
        print("\n⚠  Average score below 3.0 — pipeline needs improvement.")
        sys.exit(1)


if __name__ == "__main__":
    main()
