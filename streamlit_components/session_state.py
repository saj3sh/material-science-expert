import re
from typing import NamedTuple, Optional
import pandas as pd
from langchain_core.messages import BaseMessage
from utils.data_formatting import extract_markdown_tables, REGEX_ADVANCED_TABLE_PLACEHOLDER_DIV
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import streamlit as st
import uuid
import random


class AiThoughtProcess(NamedTuple):
    label: str
    markdowns: list[str]
    state: str


class MessageWithAdvancedTables(BaseMessage):
    avatar: str
    has_advanced_tables: bool
    html_content: Optional[str] = None
    dataframes: Optional[list[pd.DataFrame]] = None
    # None for human messages
    ai_final_thought: Optional[AiThoughtProcess] = None

    class Config:
        # prevent `PydanticSchemaGenerationError` caused by self.dataframes
        arbitrary_types_allowed = True

    def __write_markdown_with_advanced_tables(self):
        """
        Renders HTML content and inserts advanced tables at placeholders.

        This method looks for placeholders in the HTML content and replaces them with 
        interactive tables from the `dataframes` list. It uses `st.markdown` for regular 
        HTML rendering and `AgGrid` for table rendering. Tables are only shown if the 
        placeholder index is valid.

        """
        last_pos = 0
        for match in re.finditer(REGEX_ADVANCED_TABLE_PLACEHOLDER_DIV, self.html_content):
            # write contents before the current placeholder div
            st.markdown(
                self.html_content[last_pos:match.start()],
                unsafe_allow_html=True
            )
            # id of placeholder div contains table idx
            table_index = int(match.group(2))

            # region configure and load AgGrid
            grid_builder = GridOptionsBuilder.from_dataframe(
                dataframe=self.dataframes[table_index]
            )
            # config sidebar - hide pro features - pivot mode, row groups, and values section
            sidebar_options = {
                "sideBar": {
                    "toolPanels": [
                        "filters",
                        {
                            "id": 'columns',
                            "labelDefault": 'Columns',
                            "labelKey": 'columns',
                            "iconKey": 'columns',
                            "toolPanel": 'agColumnsToolPanel',
                            "toolPanelParams": {
                                "suppressPivotMode": True,
                                "suppressRowGroups": True,
                                "suppressValues": True

                            }
                        }
                    ]
                },
            }
            grid_builder.configure_grid_options(**sidebar_options)
            grid_options = grid_builder.build()
            AgGrid(
                data=self.dataframes[table_index],
                gridOptions=grid_options,
                key=f'advanced-table-{uuid.uuid4()}',
                theme="balham",
                update_mode=GridUpdateMode.NO_UPDATE
            )
            # endregion

            last_pos = match.end()
        # write contents after the last placeholder div
        st.markdown(self.html_content[last_pos:], unsafe_allow_html=True)

    def __display_ai_final_thought(self):
        if not self.ai_final_thought:
            return
        status = st.container().status(
            label=self.ai_final_thought.label,
            expanded=False,
            state=self.ai_final_thought.state
        )
        for markdown in self.ai_final_thought.markdowns:
            status.write(markdown)

    def display_chat_message(self):
        with st.chat_message(self.avatar):
            self.__display_ai_final_thought()
            if self.has_advanced_tables:
                self.__write_markdown_with_advanced_tables()
            else:
                st.write(self.content)


class ChatMessageHistoryWithAdvancedTables:
    SESSION_KEY = 'chat_message_history_with_advance_tables'

    def __init__(self, avatars):
        if self.SESSION_KEY not in st.session_state:
            st.session_state[self.SESSION_KEY] = []
        self._messages = st.session_state[self.SESSION_KEY]
        self._avatars = avatars

    @property
    def messages(self) -> list[MessageWithAdvancedTables]:
        return self._messages

    def add_message(
        self,
        message: BaseMessage,
        skip_advanced_tables: bool = True,
        ai_final_thought: Optional[AiThoughtProcess] = None
    ) -> bool:
        """Returns `True` if any tables are detected, else `False`"""
        message = MessageWithAdvancedTables(
            avatar=self._avatars[message.type],
            ai_final_thought=ai_final_thought,
            has_advanced_tables=False,
            **message.__dict__
        )
        if not skip_advanced_tables:
            message.html_content, message.dataframes = extract_markdown_tables(
                message.content)
            message.has_advanced_tables = len(message.dataframes) > 0
        self._messages.append(message)
        return message.has_advanced_tables

    def clear(self) -> None:
        self._messages.clear()
