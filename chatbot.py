from langchain_ollama import ChatOllama
import streamlit as st

# Sidebar for user inputs and links
with st.sidebar:
    st.markdown("### LLM Settings")
    remote_ollama_url = st.text_input("Remote Ollama URL", value="https://llm.bicbioeng.org")
    st.markdown("[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)")
    st.markdown(
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)]"
        "(https://codespaces.new/streamlit/llm-examples?quickstart=1)"
    )

# App title and initial setup
st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by a self-hosted LLM")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display existing messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Input prompt handling
if prompt := st.chat_input():
    if not remote_ollama_url:
        st.error("Please provide a valid remote Ollama URL.")
        st.stop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM and get response
    try:
        llm = ChatOllama(
            model="llama3.2:1b",
            base_url=remote_ollama_url
        )
        response = llm.invoke(st.session_state["messages"])
        # print(response)
        # Extract the assistant's reply
        msg = response.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
    except Exception as e:
        st.error(f"Error communicating with the LLM: {e}")