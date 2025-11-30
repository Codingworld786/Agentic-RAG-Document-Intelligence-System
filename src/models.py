from typing import TypedDict, Literal, List

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    route: Literal["rag", "direct"]