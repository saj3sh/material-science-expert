from email import utils
from typing import Callable
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from typing_extensions import TypedDict
from utils.data_formatting import extract_material_ids
from utils.embeddings import CustomEmbeddings
from utils.prompts import *
from utils.embedding_models import get_matscibert
from langchain.prompts import PromptTemplate

MATERIAL_PROJECT_BASE_URL = "https://next-gen.materialsproject.org/materials"

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
        self._status = None
        self._output_container = None
        self.output_text = ""
        pass

    def add_streamlit_containers(self, status_container, output_container):
        self._status = status_container.status("**Analyzing user query**")
        self._output_container = output_container

    def summarize(self, state):
        self._status.update(
            label="**Step: Summarizing previous conversation**", expanded=True)
        query = state["query"]
        # inv:- state["chat_history"] is not None
        chat_history = [
            f"{message.type}: {message.content}" for message in state["chat_history"]
        ]
        chain_summarize_conversation = prompt_summarize_conversation | self._get_ollama_model(
            temperature=.3) | StrOutputParser()
        query_result = chain_summarize_conversation.invoke(
            {"query": query, "chat_history": "\n".join(chat_history)})
        self._status.write(
            f'**Step: Summarizing previous conversation:** Conversation summary "{query_result}"')
        return {"summary": query_result}

    def generate_related_attributes(self, state):
        self._status.update(
            label="**Step: Finding related material attributes**")
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
        self._status.write(
            f"**Step: Finding related material attributes:** {(related_attributes)} have been found to be relevant to the question")
        return {"related_attributes": related_attributes}

    def has_sufficient_context(self, state):
        self._status.update(
            label="**Step: Determining whether the knowledge base has sufficient context or not**")
        related_attributes = state["related_attributes"]
        if related_attributes:
            self._status.write(
                f"**Step: Determining whether the knowledge base has sufficient context or not:** Knowledge base may have some contexts: continuing the retrieval process")
        else:
            self._status.write(
                f"**Step: Determining whether the knowledge base has sufficient context or not:** Knowledge base does not have sufficient context to proceed: skipping the retrieval process")
        return bool(related_attributes)

    def generate_search_query(self, state):
        self._status.update(
            label="**Step: Crafting more context-rich search query**")
        summary = state["summary"]
        attributes = state["related_attributes"]
        chain_generate_search_query = prompt_generate_search_query | self._get_ollama_model(
            temperature=.3) | StrOutputParser()
        query_result = chain_generate_search_query.invoke(
            {"query": summary, "related_attributes": "\n".join(attributes)})
        self._status.write(
            f'**Step: Crafting more context-rich search query:** Refactored search query "{query_result}"')
        return {"search_query": query_result}

    def generate_results_limit(self, state):
        self._status.update(
            label="**Step: Determining a limit on the number of data points to retrieve**")
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
        self._status.write(
            f"**Step: Determining a limit on the number of data points to retrieve:** Decided to limit data points to k={required_data_points}")
        return {"required_data_points": required_data_points}

    @staticmethod
    def __get_document_source_md(page_content):
        ids = extract_material_ids(page_content)
        if ids:
            return f'[{MATERIAL_PROJECT_BASE_URL}/{ids[0]}]({MATERIAL_PROJECT_BASE_URL}/{ids[0]})'
        return 'source information missing'

    def retrieve_context(self, state):
        self._status.update(
            label="**Step: Retrieving contexts from the knowledge base**")
        search_query = state["search_query"]
        required_data_points = state["required_data_points"]
        docs = retriever.invoke(search_query, top_k=required_data_points)
        contexts = [doc.page_content for doc in docs]
        self._status.write(
            f"**Step: Retrieving contexts from the knowledge base:** Found {len(contexts)} reference documents:")
        for index, context in enumerate(contexts):
            doc_source_md = MatSciStateGraph.__get_document_source_md(context)
            self._status.markdown(
                f"[{index+1}] [{doc_source_md}]: {context}")
        return {"contexts": contexts}

    def generate_final_response(self, state):
        self._status.update(
            label="**Step: Generating final response**")
        summary = state["summary"]
        contexts = state["contexts"] if "contexts" in state else [""]
        chain_generate_final_respone: PromptTemplate = prompt_generate_final_response | self._get_ollama_model(
            temperature=.2) | StrOutputParser()
        stream_gen = chain_generate_final_respone.stream({
            "query": summary,
            "contexts": "\n".join(contexts)
        })
        for output_chunk in stream_gen:
            self.output_text += output_chunk
            self._output_container.markdown(self.output_text)
        self._status.write(
            f"**Step: Generating final response:** Generated final response based on retrieved context")
        self._status.update(state="complete")
        return {"output": self.output_text}
