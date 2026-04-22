import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.run_agent import app as rag_graph

api = FastAPI(
    title="Local Agentic RAG API",
    description=(
        "Self-correcting Retrieval-Augmented Generation pipeline. "
        "Wraps a LangGraph state machine that retrieves, reflects, and "
        "rewrites queries until a satisfactory answer is produced."
    ),
    version="1.0.0",
)


@api.get("/health", summary="Liveness probe")
async def health() -> dict:
    """Lightweight health check for container orchestrators."""
    return {"status": "healthy"}


class AskRequest(BaseModel):
    """Incoming question payload."""

    question: str = Field(
        ...,
        min_length=1,
        description="The user's natural-language question.",
        json_schema_extra={"example": "What is retrieval-augmented generation?"},
    )


class AskResponse(BaseModel):
    """Structured response returned to the client."""

    answer: str = Field(
        ...,
        description="The final generated answer from the RAG pipeline.",
    )
    retry_count: int = Field(
        ...,
        description="Number of self-correction retries the agent performed.",
    )
    execution_time_ms: float = Field(
        ...,
        description="Wall-clock time for the full graph invocation, in milliseconds.",
    )



@api.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask the RAG pipeline a question",
    description=(
        "Accepts a question, runs it through the self-correcting LangGraph "
        "agent, and returns the answer along with retry metadata and timing."
    ),
)
async def ask(request: AskRequest) -> AskResponse:
    """
    1. Start a high-resolution timer.
    2. Build the initial state dict expected by the LangGraph state machine.
    3. Invoke the compiled graph.
    4. Extract the answer and retry count from the final state.
    5. Return the formatted response (or a 500 on any internal failure).
    """

    initial_state: dict = {
        "question": request.question,
        "generation": "",
        "docs": [],
        "raw_docs": [],
        "retry_count": 0,
        "grade": ""
    }

    start = time.perf_counter()

    try:
        final_state = rag_graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline failed: {exc}",
        ) from exc

    elapsed_ms = (time.perf_counter() - start) * 1_000

    return AskResponse(
        answer=final_state.get("generation", ""),
        retry_count=final_state.get("retry_count", 0),
        execution_time_ms=round(elapsed_ms, 2),
    )
