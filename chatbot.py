from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from streamlit_components.page_styles import format_page_styles
from streamlit_components.sidebar import configure_llm
from langchain_core.globals import set_verbose
from utils.prompts import *
from langgraph.graph import END, StateGraph

from utils.state_graph import GraphState, MatSciStateGraph
set_verbose(True)
format_page_styles(st)
if 'remote_ollama_url_enabled' not in st.session_state:
    st.session_state.remote_ollama_url_enabled = True

get_ollma_model, get_ollma_model_json = configure_llm(st)

st.title("MatSci Expert")
st.caption(
    "A knowledge-based system built using data from [Material Project](https://next-gen.materialsproject.org/)")

# region defining system workflow
sysGraph = MatSciStateGraph(get_ollma_model, get_ollma_model_json)
workflow = StateGraph(GraphState)
workflow.add_node("summarize", sysGraph.summarize)
workflow.add_node(
    "generate_related_attributes",
    sysGraph.generate_related_attributes
)
workflow.add_node("generate_search_query", sysGraph.generate_search_query)
workflow.add_node("generate_results_limit", sysGraph.generate_results_limit)
workflow.add_node("retrieve_context", sysGraph.retrieve_context)
workflow.add_node(
    "generate_final_response",
    sysGraph.generate_final_response
)


workflow.set_conditional_entry_point(lambda x: "summarize")
workflow.add_edge("summarize", "generate_related_attributes")
# shortcircuit to generate_final_response if RAG lacks sufficient content - reduce false positives
workflow.add_conditional_edges(
    "generate_related_attributes",
    sysGraph.has_sufficient_context, {
        True: "generate_search_query",
        False: "generate_final_response"
    })
workflow.add_edge("generate_search_query", "generate_results_limit")
workflow.add_edge("generate_results_limit", "retrieve_context")
workflow.add_edge("retrieve_context", "generate_final_response")
workflow.add_edge("generate_final_response", END)

compiled_workflow = workflow.compile()
# endregion

message_history = StreamlitChatMessageHistory()

if len(message_history.messages) == 0:
    message_history.clear()
    message_history.add_message(AIMessage(content="How can I help you?"))

avatars = {
    "human": "➖", "ai": "🧬"}

for msg in message_history.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me questions related to material science"):
    message_history.add_message(HumanMessage(content=user_query))
    st.chat_message("human", avatar="➖").write(user_query)

    with st.chat_message("ai", avatar="🧬"):
        sysGraph.add_streamlit_containers(
            status_container=st.container(),
            output_container=st.empty()
        )
        response = compiled_workflow.invoke(
            {
                "query": user_query,
                "chat_history": message_history.messages
            }
        )
