from email import utils
from typing import Callable
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from typing_extensions import TypedDict
from utils.embeddings import CustomEmbeddings
from utils.callback_handlers import RagRetrievalHandler
from utils.prompts import *
from utils.embedding_models import get_matscibert

# region initializing for Qdrant retrieval
qdrant_client = QdrantClient(
    host="localhost", port=6333)
embedding_model = CustomEmbeddings(*get_matscibert())
vectorstore = QdrantVectorStore(
    embedding=embedding_model,
    collection_name="materials",
    client=qdrant_client
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
# endregion


class GraphState(TypedDict):
    query: str
    chat_history: str
    summary: str
    related_attributes: str
    search_query: str
    required_data_points: str
    contexts: str
    output: str


class MatSciStateGraph:
    _get_ollama_model: Callable[[float | None], ChatOllama]
    _get_ollama_json_model: Callable[[float | None], ChatOllama]

    def __init__(
        self,
        get_ollama_model: Callable[[float | None], ChatOllama],
        get_ollama_json_model: Callable[[float | None], ChatOllama]
    ):
        self._get_ollama_model = get_ollama_model
        self._get_ollama_json_model = get_ollama_json_model
        self.container = None
        pass

    def add_streamlit_container(self, contaienr):
        self.container = contaienr

    def summarize(self, state):
        print("Step: Summarizing conversation")
        query = state["query"]
        # inv:- state["chat_history"] is not None
        chat_history = [
            f"{message.type}: {message.content}" for message in state["chat_history"]
        ]
        chain_summarize_conversation = prompt_summarize_conversation | self._get_ollama_model(
            temperature=.3) | StrOutputParser()
        query_result = chain_summarize_conversation.invoke(
            {"query": query, "chat_history": "\n".join(chat_history)})
        return {"summary": query_result}

    def generate_related_attributes(self, state):
        print("Step: Generating related attributes")
        summary = state["summary"]
        chain_generate_related_attributes = prompt_generate_related_attributes | self._get_ollama_json_model(
            temperature=.1) | JsonOutputParser()
        query_result = chain_generate_related_attributes.invoke(
            {"query": summary})
        related_attributes = (
            query_result["related_attributes"]
            if query_result["is_context_available"]
            else []
        )
        return {"related_attributes": related_attributes}

    def has_sufficient_context(self, state):
        print("Step: Determining whether the knowledge base has sufficient context")
        related_attributes = state["related_attributes"]
        if not related_attributes:
            return False
        return True

    def generate_search_query(self, state):
        summary = state["summary"]
        attributes = state["related_attributes"]
        chain_generate_search_query = prompt_generate_search_query | self._get_ollama_model(
            temperature=.3) | StrOutputParser()
        query_result = chain_generate_search_query.invoke(
            {"query": summary, "related_attributes": "\n".join(attributes)})
        return {"search_query": query_result}

    def generate_results_limit(self, state):
        print("Step: Determining a limit on the number of data points to retrieve")
        query = state["summary"]
        chain_required_data_points = prompt_required_data_points | self._get_ollama_json_model(
            temperature=0) | JsonOutputParser()
        query_result = chain_required_data_points.invoke(
            {"query": query})
        # clamp value between 1 and 10
        required_data_points = max(
            min(int(query_result["required_data_points"]), 10),
            1
        )
        return {"required_data_points": required_data_points}

    def retrieve_context(self, state):
        print("Step: Retrieving contexts from the knowledge base")
        search_query = state["search_query"]
        required_data_points = state["required_data_points"]
        retrieval_handler = RagRetrievalHandler(self.container)
        docs = retriever.invoke(
            search_query, callbacks=[retrieval_handler], top_k=required_data_points)
        context = [doc.page_content for doc in docs]
        return {"contexts": context}

    def generate_final_response(self, state):
        print("Step: Generating final respone")
        summary = state["summary"]
        contexts = state["contexts"] if "contexts" in state else [""]
        chain_generate_final_respone = prompt_generate_final_response | self._get_ollama_model(
            temperature=.2) | StrOutputParser()
        query_result = chain_generate_final_respone.invoke(
            {"summary": summary, "contexts": "\n".join(contexts)})
        return {"output": query_result}
