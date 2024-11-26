def format_page_styles(st):
    page_style = """
    <style>
    #MainMenu {
        display: none !important;
    }
    header {
        display:none !important;
    }
    # .stChatMessage:nth-child(odd) {
    #     flex-direction:row-reverse;
    #     min-width:50%;
    #     align-self:flex-end;
    # }
    # .stChatMessage:nth-child(odd) > div:first-child {
    # display:none !important;
    # }
    </style>
    """
    st.markdown(page_style, unsafe_allow_html=True)
