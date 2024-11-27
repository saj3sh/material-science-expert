from email import utils
from typing import Callable, Literal, Optional
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from typing_extensions import TypedDict
from streamlit_components.session_state import AiThoughtProcess
from utils.data_formatting import extract_material_ids
from utils.embeddings import CustomEmbeddings
from utils.prompts import *
from utils.embedding_models import get_matscibert
from langchain.prompts import PromptTemplate
from utils.qdrant_client import get_qdrant_client

MATERIAL_PROJECT_BASE_URL = "https://next-gen.materialsproject.org/materials"

# region initializing for Qdrant retrieval
qdrant_client = get_qdrant_client()
embedding_model = CustomEmbeddings(*get_matscibert())
vectorstore = QdrantVectorStore(
    embedding=embedding_model,
    collection_name="materials",
    client=qdrant_client
)
# endregion


class GraphState(TypedDict):
    query: str
    chat_history: str
    summary: str
    related_attributes: str
    search_query: str
    required_data_points: str
    material_ids: list[str]
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

        self._ai_thought_label: str = ''
        self._ai_thought_markdowns: list[str] = []
        self._ai_thought_state: str = ''
        pass

    def add_streamlit_containers(self, analysis_container, output_container):
        self._status = analysis_container
        self._output_container = output_container

    def extract_and_clear_ai_thought(self):
        final_thought = AiThoughtProcess(
            label=self._ai_thought_label,
            markdowns=self._ai_thought_markdowns,
            state=self._ai_thought_state
        )
        return final_thought

    def summarize(self, state):
        self.__update_status(
            label="**Step: Summarizing previous conversation**")
        query = state["query"]
        # inv:- state["chat_history"] is not None
        chat_history = [
            f"{message.type}: {message.content}" for message in state["chat_history"]
        ]
        chain_summarize_conversation = prompt_summarize_conversation | self._get_ollama_model(
            temperature=.3) | StrOutputParser()
        # skipping chain invocation due to LLM hallucination - need to refactor prompt
        # query_result = chain_summarize_conversation.invoke(
        #     {"query": query, "chat_history": "\n".join(chat_history)})
        query_result = query
        self.__write_status(
            f'**Step: Summarizing previous conversation:** Conversation summary "{query_result}"')
        return {"summary": query_result}

    def generate_related_attributes(self, state):
        self.__update_status(
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
        self.__write_status(
            f"**Step: Finding related material attributes:** {(related_attributes)} have been found to be relevant to the question")
        return {"related_attributes": related_attributes}

    def has_sufficient_context(self, state):
        self.__update_status(
            label="**Step: Determining whether the knowledge base has sufficient context or not**")
        related_attributes = state["related_attributes"]
        if related_attributes:
            self.__write_status(
                f"**Step: Determining whether the knowledge base has sufficient context or not:** Knowledge base may have some contexts: continuing the retrieval process")
        else:
            self.__write_status(
                f"**Step: Determining whether the knowledge base has sufficient context or not:** Knowledge base does not have sufficient context to proceed: skipping the retrieval process")
        return bool(related_attributes)

    def generate_search_query(self, state):
        self.__update_status(
            label="**Step: Crafting more context-rich search query**")
        summary = state["summary"]
        attributes = state["related_attributes"]
        chain_generate_search_query = prompt_generate_search_query | self._get_ollama_model(
            temperature=.3) | StrOutputParser()
        # skipping chain invocation due to LLM hallucination - need to refactor prompt
        # query_result = chain_generate_search_query.invoke(
        #     {"query": summary, "related_attributes": "\n".join(attributes)})
        query_result = summary
        self.__write_status(
            f'**Step: Crafting more context-rich search query:** Refactored search query "{query_result}"')
        return {"search_query": query_result}

    def generate_results_limit(self, state):
        self.__update_status(
            label="**Step: Determining a limit on the number of data points to retrieve**")
        query = state["summary"]
        material_ids = extract_material_ids(query)
        if material_ids:
            self.__write_status(
                (f"**Step: Determining a limit on the number of data points to retrieve:** "
                 f"Extracted Material IDs {(material_ids)}. "
                 "Filtering will be based on these IDs rather than the k-value")
            )
            return {"material_ids": material_ids}
        chain_required_data_points = prompt_required_data_points | self._get_ollama_json_model(
            temperature=0) | JsonOutputParser()
        query_result = chain_required_data_points.invoke(
            {"query": query})
        # clamp value between 1 and 10
        required_data_points = max(
            min(int(query_result["required_data_points"]), 10),
            1
        )
        self.__write_status(
            (f"**Step: Determining a limit on the number of data points to retrieve:** "
             f"Decided to limit data points to k={required_data_points}")
        )
        return {"required_data_points": required_data_points}

    @ staticmethod
    def __get_document_source_md(page_content):
        ids = extract_material_ids(page_content)
        if ids:
            return f'[{MATERIAL_PROJECT_BASE_URL}/{ids[0]}]({MATERIAL_PROJECT_BASE_URL}/{ids[0]})'
        return 'source information missing'

    def __update_status(
        self,
        label: Optional[str] = None,
        expanded: Optional[bool] = None,
        state: Optional[Literal['running', 'complete', 'error']] = None
    ):
        update_args = {}
        if label is not None:
            update_args['label'] = label
            self._ai_thought_label = label
        if expanded is not None:
            update_args['expanded'] = expanded
        if state is not None:
            update_args['state'] = state
            self._ai_thought_state = state
        if update_args:
            self._status.update(**update_args)

    def __write_status(self, status: str):
        self._ai_thought_markdowns.append(status)
        self._status.write(status)

    def retrieve_context(self, state):
        self.__update_status(
            label="**Step: Retrieving contexts from the knowledge base**")
        search_query = state["search_query"]
        required_data_points = state["required_data_points"] if "required_data_points" in state else 0
        material_ids = state["material_ids"] if "material_ids" in state else []

        if material_ids:
            metadata_filter = {
                "should": [
                    {
                        "key": "material_id",
                        "match": {"value": material_id}
                    }
                    for material_id in material_ids
                ]
            }
            retriever = vectorstore.as_retriever(
                search_kwargs={"filter": metadata_filter, "k": 10}
            )
            docs = retriever.invoke("")
        else:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": required_data_points}
            )
            docs = retriever.invoke(search_query)

        contexts = [doc.page_content for doc in docs]
        self.__write_status(
            f"**Step: Retrieving contexts from the knowledge base:** Found {len(contexts)} reference documents:")
        for index, context in enumerate(contexts):
            doc_source_md = MatSciStateGraph.__get_document_source_md(context)
            self.__write_status(f"[{index+1}] [{doc_source_md}]: {context}")
        return {"contexts": contexts}

    def generate_final_response(self, state):
        self.__update_status(
            label="**Step: Generating final response**")
        summary = state["summary"]
        contexts = state["contexts"] if "contexts" in state else []
        generation_prompt = prompt_answer_based_on_context if contexts else prompt_request_refinement
        chain_generate_final_respone: PromptTemplate = generation_prompt | self._get_ollama_model(
            temperature=.2) | StrOutputParser()
        stream_gen = chain_generate_final_respone.stream({
            "query": summary,
            "contexts": "\n".join(contexts)
        })
        for output_chunk in stream_gen:
            self.output_text += output_chunk
            self._output_container.markdown(self.output_text)
        self.__write_status(
            f"**Step: Generating final response:** Generated final response based on available context")
        self.__update_status(state="complete")
        return {"output": self.output_text}
