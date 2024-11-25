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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
import streamlit as st
from langchain_qdrant import QdrantVectorStore
from langchain import hub
from utils.custom_embeddings import MatSciEmbeddings
import debugpy
from langchain_core.globals import set_verbose
set_verbose(True)
load_dotenv()

# Sidebar for user inputs and links
with st.sidebar:
    st.markdown("### LLM Settings")
    remote_ollama_url = st.text_input(
        "Remote Ollama URL", value="https://llm.bicbioeng.org")
    st.markdown(
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)")
    st.markdown(
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)]"
        "(https://codespaces.new/streamlit/llm-examples?quickstart=1)"
    )

# App title and initial setup
st.title("💬 MatSciBot")
st.caption("🚀 A Streamlit chatbot powered by a self-hosted LLM")

qdrant_client = QdrantClient(host="localhost", port=6333)
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


# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [("assistant", "How can I help you?")]

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

# Input prompt handling
if prompt := st.chat_input():
    if not remote_ollama_url:
        st.error("Please provide a valid remote Ollama URL.")
        st.stop()

    # Append user message
    st.session_state["messages"].append(("user", prompt))
    st.chat_message("user").write(prompt)
    template = ChatPromptTemplate.from_messages(st.session_state["messages"])

    # Initialize LLM and get response
    try:
        output_parser = StrOutputParser()
        llm = ChatOllama(
            model="llama3.2:1b",
            base_url=remote_ollama_url,
            temperature=0
        )
        # docs = retriever.get_relevant_documents(prompt)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke({"input": prompt})
        # print(response)
        # Extract the assistant's reply
        st.session_state["messages"].append(("assistant", response))
        st.chat_message("assistant").write(response)
    except Exception as e:
        stack_trace = traceback.format_exc()
        st.error(f"Error communicating with the LLM: {e}")
        st.markdown(f"Stack Trace:```{stack_trace}```")
