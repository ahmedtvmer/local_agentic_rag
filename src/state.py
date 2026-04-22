from typing import TypedDict, List, Dict

class GraphState(TypedDict):
    question: str
    generation: str
    docs: List[str]
    raw_docs: List[str]
    retry_count: int
    grade: str
    chat_history: List[Dict[str, str]]