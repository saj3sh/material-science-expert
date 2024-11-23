from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
import streamlit as st
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain import hub
from custom_embeddings import MatSciEmbeddings

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

vectorstore = Qdrant.from_existing_collection(
    embedding=embeddings,
    collection_name="materials",
    client=qdrant_client
)

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [("assistant", "How can I help you?")]

# Display existing messages
for (role, content) in st.session_state["messages"]:
    st.chat_message(role).write(content)

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
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke({"input": prompt})
        # print(response)
        # Extract the assistant's reply
        st.session_state["messages"].append(("assistant", response))
        st.chat_message("assistant").write(response)
    except Exception as e:
        st.error(f"Error communicating with the LLM: {e}")
