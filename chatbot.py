import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import traceback
from dotenv import load_dotenv
import langchain_core
import langchain_core.runnables
import langchain_core.runnables.base
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
import streamlit as st
from langchain_qdrant import QdrantVectorStore
from langchain import hub
from chatbot_components.sidebar import configure_llm
from utils.custom_embeddings import MatSciEmbeddings
import debugpy
from langchain_core.globals import set_verbose
import config
from prompts.utility_prompts import prompt_rephrase_user_query, prompt_required_data_points
set_verbose(True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = [("assistant", "How can I help you?")]
if 'remote_ollama_url_enabled' not in st.session_state:
    st.session_state.remote_ollama_url_enabled = True

get_llma_model, get_llma_model_json = configure_llm(st)

st.title("MatRAG")
st.caption("📚 A knowledge-based system built using Material Project data")


chain_get_required_data_points = prompt_required_data_points | get_llma_model_json(
    temperature=0) | JsonOutputParser()
chain_rephrase_user_query = prompt_rephrase_user_query | get_llma_model(
    temperature=0) | StrOutputParser()

qdrant_client = QdrantClient(
    url=config.QDRANT_URL, api_key=config.QDRANT_TOKEN, port=6333)
embeddings = MatSciEmbeddings()

vectorstore = QdrantVectorStore(
    embedding=embeddings,
    collection_name="materials",
    client=qdrant_client
)

retriever = vectorstore.as_retriever()
rag_prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    if not docs:
        return "No docs present"
    return "\n\n".join(doc.page_content for doc in docs)


# Display existing messages
for (role, content) in st.session_state["messages"]:
    st.chat_message(role).write(content)


# # Sample dataframe
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Age': [25, 30, 35],
#     'City': ['New York', 'Los Angeles', 'Chicago']
# }
# df = pd.DataFrame(data)

# # Display the table
# st.dataframe(df)


# # Create some data
# x = np.linspace(0, 10, 100)
# y = np.sin(x)

# # Create a plot
# plt.figure(figsize=(10, 6))
# plt.plot(x, y)
# plt.title("Sine Wave")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")

# # Display the plot in Streamlit
# st.pyplot(plt)
one_model = get_llma_model(temperature=0) | StrOutputParser()
if user_prompt := st.chat_input():
    st.session_state["messages"].append(("user", user_prompt))
    st.chat_message("user").write(user_prompt)
    # template = ChatPromptTemplate.from_messages(st.session_state["messages"])

    try:
        response = chain_rephrase_user_query.invoke({"query": user_prompt})
        st.session_state["messages"].append(("assistant", response))
        st.chat_message("assistant").write(response)
    except Exception as e:
        stack_trace = traceback.format_exc()
        st.error(f"Error communicating with the LLM: {e}")
        st.markdown(f"Stack Trace:```{stack_trace}```")
