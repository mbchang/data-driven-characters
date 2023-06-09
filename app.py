from dataclasses import asdict
from io import StringIO
import json
import os
import streamlit as st
from streamlit_chat import message

from data_driven_characters.character import generate_character_definition, Character
from data_driven_characters.corpus import (
    generate_rolling_summaries,
    generate_docs,
)
from data_driven_characters.chatbots import (
    SummaryChatBot,
    RetrievalChatBot,
    GenerativeChatBot,
)


def reset_chat():
    st.cache_resource.clear()
    if "messages" in st.session_state:
        del st.session_state["messages"]


def clear_user_input():
    if "user_input" in st.session_state:
        st.session_state["user_input"] = ""


@st.cache_resource()
def create_chatbot(character_definition, rolling_summaries, chatbot_type):
    if chatbot_type == "summary":
        chatbot = SummaryChatBot(character_definition=character_definition)
    elif chatbot_type == "retrieval":
        chatbot = RetrievalChatBot(
            character_definition=character_definition,
            rolling_summaries=rolling_summaries,
        )
    elif chatbot_type == "generative":
        chatbot = GenerativeChatBot(
            character_definition=character_definition,
            rolling_summaries=rolling_summaries,
        )
    else:
        raise ValueError(f"Unknown chatbot type: {chatbot_type}")
    return chatbot


@st.cache_data(persist="disk")
def process_corpus(corpus):
    # load docs
    docs = generate_docs(
        corpus=corpus,
        chunk_size=2048,
        chunk_overlap=64,
    )

    # generate rolling summaries
    rolling_summaries = generate_rolling_summaries(docs=docs)
    return rolling_summaries


# TODO: this should be a json
@st.cache_data(persist="disk")
def get_character_definition(name, rolling_summaries):
    character_definition = generate_character_definition(
        name=name,
        rolling_summaries=rolling_summaries,
    )
    return asdict(character_definition)


def main():
    st.title("Data-Driven Characters")
    st.write("Create your own character chatbots, grounded in existing corpora.")
    openai_api_key = st.text_input(
        label="Your OpenAI API KEY",
        placeholder="Your OpenAI API KEY",
        type="password",
    )
    os.environ["OPENAI_API_KEY"] = openai_api_key

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload corpus")
        if uploaded_file is not None:
            corpus_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]

            # read file
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            corpus = stringio.read()

            # scrollable text
            st.markdown(
                f"""
                <div style='overflow: auto; height: 300px; border: 1px solid gray; border-radius: 5px; padding: 10px'>
                    {corpus}</div>
                """,
                unsafe_allow_html=True,
            )

            st.divider()

            # get character name
            character_name = st.text_input(f"Enter a character name from {corpus_name}")

            if character_name:
                if not openai_api_key:
                    st.error(
                        "You must enter an API key to use the OpenAI API. Please enter an API key in the sidebar."
                    )
                    return

                if (
                    "character_name" in st.session_state
                    and st.session_state["character_name"] != character_name
                ):
                    clear_user_input()
                    reset_chat()

                st.session_state["character_name"] = character_name

                with st.spinner("Processing corpus (this will take a while)..."):
                    rolling_summaries = process_corpus(corpus)

                with st.spinner("Generating character definition..."):
                    # get character definition
                    character_definition = get_character_definition(
                        name=character_name,
                        rolling_summaries=rolling_summaries,
                    )

                    print(json.dumps(character_definition, indent=4))
                    chatbot_type = st.selectbox(
                        "Select a memory type",
                        options=["summary", "retrieval", "generative"],
                        index=0,
                    )
                    if (
                        "chatbot_type" in st.session_state
                        and st.session_state["chatbot_type"] != chatbot_type
                    ):
                        clear_user_input()
                        reset_chat()

                    st.session_state["chatbot_type"] = chatbot_type

                    st.markdown(
                        f"[Export to character.ai](https://beta.character.ai/editing):"
                    )
                    st.write(character_definition)

    if uploaded_file is not None and character_name:
        st.divider()
        chatbot = create_chatbot(
            character_definition=Character(**character_definition),
            rolling_summaries=rolling_summaries,
            chatbot_type=chatbot_type,
        )

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
            st.session_state["messages"] = [{"role": "assistant", "content": greeting}]

        # the old messages
        for msg in st.session_state.messages:
            message(msg["content"], is_user=msg["role"] == "user")

        # the new message
        if user_input:
            # st.session_state["user_input"] = user_input
            st.session_state.messages.append({"role": "user", "content": user_input})
            message(user_input, is_user=True)
            with st.spinner(f"{chatbot.character_definition.name} is thinking..."):
                response = chatbot.step(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            message(response)


if __name__ == "__main__":
    main()
