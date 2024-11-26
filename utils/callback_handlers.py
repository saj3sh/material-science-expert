from langchain_core.callbacks import BaseCallbackHandler
from utils.data_formatting import extract_material_ids

MATERIAL_PROJECT_BASE_URL = "https://next-gen.materialsproject.org/materials"


class RagRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    @staticmethod
    def __get_document_source_md(page_content):
        ids = extract_material_ids(page_content)
        if not ids:
            return f'[{MATERIAL_PROJECT_BASE_URL}/{ids[0]}]({MATERIAL_PROJECT_BASE_URL}/{ids[0]})'
        return 'source information missing'

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            doc_source_md = RagRetrievalHandler.__get_document_source_md(
                doc.page_content)
            self.status.write(
                f"**Document {idx} from <{doc_source_md}>**")
        self.status.update(state="complete")


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)
