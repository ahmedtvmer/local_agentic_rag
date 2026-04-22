import re
import sys
import httpx
from langchain_community.llms import Ollama


RAG_API_URL = "http://localhost:8080/ask"
REQUEST_TIMEOUT = 120.0

TEST_CASES: list[dict[str, str]] = [
    {
        "question": "Who developed ChatGPT?",
        "expected_fact": "ChatGPT was developed by OpenAI.",
    },
    {
        "question": "When was ChatGPT officially released?",
        "expected_fact": "ChatGPT was released in November 2022.",
    },
    {
        "question": "What kind of models power ChatGPT?",
        "expected_fact": "It uses large language models, specifically generative pre-trained transformers (GPTs).",
    },
]


JUDGE_SYSTEM_PROMPT = """\
You are a strict factual-accuracy judge.

QUESTION: {question}
EXPECTED FACT: {expected_fact}
GENERATED ANSWER: {answer}

Task: Rate the GENERATED ANSWER on factual accuracy compared to the EXPECTED FACT.

Scale:
1 = completely wrong or irrelevant
2 = mostly wrong with a minor correct detail
3 = partially correct but missing key facts
4 = mostly correct with minor inaccuracies
5 = fully correct and consistent with the expected fact

Rules:
- Output ONLY a single integer (1, 2, 3, 4, or 5).
- Do NOT output any other text or explanation.
Score:"""

def build_judge_prompt(question: str, expected_fact: str, answer: str) -> str:
    """Construct the prompt string for the Ollama LLM."""
    return JUDGE_SYSTEM_PROMPT.format(
        question=question,
        expected_fact=expected_fact,
        answer=answer
    )

def parse_score(raw: str) -> int | None:
    """Extract the first integer 1-5 from the judge's response."""
    match = re.search(r"[1-5]", str(raw).strip())
    return int(match.group()) if match else None


def main() -> None:
    judge = Ollama(
        model="llama3.1:8b",
        temperature=0,
        base_url="http://localhost:11434"
    )

    scores: list[int] = []
    latencies: list[float] = []

    print("=" * 70)
    print("RAG Pipeline Evaluation Suite")
    print("=" * 70)

    for idx, case in enumerate(TEST_CASES, start=1):
        question = case["question"]
        expected_fact = case["expected_fact"]

        print(f"\n[Test {idx}/{len(TEST_CASES)}] {question}")
        print("-" * 60)

        try:
            response = httpx.post(
                RAG_API_URL,
                json={"question": question},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"  ✗ API request failed: {exc}")
            continue

        payload = response.json()
        answer = payload.get("answer", "")
        exec_time = payload.get("execution_time_ms", 0.0)
        latencies.append(exec_time)

        print(f"  Answer  : {answer[:120]}{'…' if len(answer) > 120 else ''}")
        print(f"  Latency : {exec_time:.1f} ms")

        try:
            prompt = build_judge_prompt(question, expected_fact, answer)
            raw_score = judge.invoke(prompt)
            score = parse_score(raw_score)
        except Exception as exc:
            print(f"  ✗ Judge LLM failed: {exc}")
            continue

        if score is None:
            print(f"  ✗ Could not parse score from judge output: '{raw_score}'")
            continue

        scores.append(score)
        print(f"  Score   : {score}/5")


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
        print("  Average Latency : N/A")

    print(f"  Tests Run       : {len(TEST_CASES)}")
    print(f"  Scored          : {len(scores)}")
    print(f"  Failed/Skipped  : {len(TEST_CASES) - len(scores)}")
    print("=" * 70)

    if avg_score < 3.0:
        print("\nAverage score below 3.0 — pipeline needs improvement.")
        sys.exit(1)

if __name__ == "__main__":
    main()