from typing import Optional

from langchain_ollama import ChatOllama

OLLAMA_MODEL_NAME = "llama3.1:8b"


def configure_llm(st):
    with st.sidebar:
        st.markdown("### Configure LLM")
        remote_ollama_url = st.text_input(
            "Remote Ollama URL", value="127.0.0.1:11434",
            disabled=st.session_state.remote_ollama_url_enabled
        )
        import streamlit as st

    def get_llma_model(temperature: Optional[float]):
        try:
            st.session_state.remote_ollama_url_enabled = False
            return ChatOllama(
                model=OLLAMA_MODEL_NAME, base_url=remote_ollama_url, temperature=temperature)
        except:
            st.session_state.remote_ollama_url_enabled = True
            st.error("Please provide a valid remote Ollama URL.")
            st.stop()

    def get_llma_model_json(temperature: Optional[float]):
        try:
            st.session_state.remote_ollama_url_enabled = False
            return ChatOllama(
                model=OLLAMA_MODEL_NAME, format="json", base_url=remote_ollama_url, temperature=temperature)
        except:
            st.session_state.remote_ollama_url_enabled = True
            st.error("Please provide a valid remote Ollama URL.")
            st.stop()

    return get_llma_model, get_llma_model_json
