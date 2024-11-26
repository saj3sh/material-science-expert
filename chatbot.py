import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import traceback
from dotenv import load_dotenv
import langchain_core
import langchain_core.runnables
import langchain_core.runnables.base
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from qdrant_client import QdrantClient
import streamlit as st
from langchain_qdrant import QdrantVectorStore
from langchain import hub
from streamlit_components.page_styles import format_page_styles
from streamlit_components.sidebar import configure_llm
from custom_callbacks.rag_retrieval_handler import RagRetrievalHandler
from custom_callbacks.stream_handler import StreamHandler
from utils.embeddings import MatSciEmbeddings
import debugpy
from langchain_core.globals import set_verbose
import config
from utils.prompts import *
set_verbose(True)
format_page_styles(st)
if 'remote_ollama_url_enabled' not in st.session_state:
    st.session_state.remote_ollama_url_enabled = True

get_llma_model, get_llma_model_json = configure_llm(st)

st.title("MatSci Expert")
st.caption(
    "A knowledge-based system built using data from [Material Project](https://next-gen.materialsproject.org/)")


chain_get_required_data_points = prompt_required_data_points | get_llma_model_json(
    temperature=0) | JsonOutputParser()
chain_rephrase_user_query = prompt_rephrase_user_query | get_llma_model_json(
    temperature=.3) | JsonOutputParser()

qdrant_client = QdrantClient(
    host="localhost", port=6333)
embeddings = MatSciEmbeddings()

vectorstore = QdrantVectorStore(
    embedding=embeddings,
    collection_name="materials",
    client=qdrant_client
)

retriever = vectorstore.as_retriever()
message_history = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=message_history, return_messages=True)


def log(text):
    print(text)
    return text


history_aware_retriever = create_history_aware_retriever(
    get_llma_model_json(
        temperature=.1), retriever, prompt_summarize_conversation
)

if len(message_history.messages) == 0:
    message_history.clear()
    message_history.add_message(AIMessage(content="How can I help you?"))

avatars = {
    "human": "➖", "ai": "🧬"}

for msg in message_history.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me questions related to material science"):
    st.chat_message(
        "user", avatar="➖").write(user_query)

    with st.chat_message("assistant", avatar="🧬"):
        retrieval_handler = RagRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = history_aware_retriever.invoke({"query": user_query, "chat_history": message_history.messages}, callbacks=[
            retrieval_handler])
