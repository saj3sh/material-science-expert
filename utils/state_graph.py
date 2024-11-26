from typing_extensions import TypedDict


class GraphState(TypedDict):
    question: str
    generation: str
    search_query: str
    context: str
