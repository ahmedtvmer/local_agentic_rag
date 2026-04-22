from langgraph.graph import END, StateGraph
from src.state import GraphState
from src.nodes import retrieve, rerank, generate, rewrite, reformulate

MAX_RETRIES = 3

def route_evaluation(state: dict) -> str:
    """Routes to generation if docs are good, rewrites if bad, or forces generation if out of retries."""
    if state["grade"] == "bad":
        if state["retry_count"] >= MAX_RETRIES:
            print("Circuit breaker hit. Forcing generation with available context.")
            return "generate"
        return "rewrite"
    return "generate"

workflow = StateGraph(GraphState)

workflow.add_node("reformulate", reformulate)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rerank", rerank)
workflow.add_node("generate", generate)
workflow.add_node("rewrite", rewrite)

workflow.set_entry_point("reformulate")
workflow.add_edge("reformulate", "retrieve")
workflow.add_edge("retrieve", "rerank")

workflow.add_conditional_edges(
    "rerank",
    route_evaluation,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()