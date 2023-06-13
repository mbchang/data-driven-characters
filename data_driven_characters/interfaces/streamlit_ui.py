import streamlit as st
from streamlit_chat import message


def reset_chat():
    st.cache_resource.clear()
    if "messages" in st.session_state:
        del st.session_state["messages"]


def clear_user_input():
    if "user_input" in st.session_state:
        st.session_state["user_input"] = ""


def converse(chatbot):
    left, right = st.columns([4, 1])
    user_input = left.text_input(
        label=f"Chat with {chatbot.character_definition.name}",
        placeholder=f"Chat with {chatbot.character_definition.name}",
        label_visibility="collapsed",
        key="user_input",
    )
    reset_chatbot = right.button("Reset", on_click=clear_user_input)
    if reset_chatbot:
        reset_chat()

    if "messages" not in st.session_state:
        greeting = chatbot.greet()
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": greeting,
                "key": 0,
            }
        ]
    # the old messages
    for msg in st.session_state.messages:
        message(msg["content"], is_user=msg["role"] == "user", key=msg["key"])

    # the new message
    if user_input:
        key = len(st.session_state.messages)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
                "key": key,
            }
        )
        message(user_input, is_user=True, key=key)
        with st.spinner(f"{chatbot.character_definition.name} is thinking..."):
            response = chatbot.step(user_input)
        key = len(st.session_state.messages)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "key": key,
            }
        )
        message(response, key=key)


class Streamlit:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def run(self):
        converse(self.chatbot)
