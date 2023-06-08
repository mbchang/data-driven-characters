from dataclasses import asdict
from io import StringIO
import json
import os
import streamlit as st
from streamlit_chat import message

from data_driven_characters.character import get_character_definition
from data_driven_characters.corpus import (
    get_rolling_summaries,
    load_docs,
)
from data_driven_characters.chatbots import (
    SummaryChatBot,
    RetrievalChatBot,
    GenerativeChatBot,
)

OUTPUT_ROOT = "app/output"
DATA_ROOT = "app/data"


def reset_chat():
    clear_user_input()
    st.cache_resource.clear()  # but this should be in app.py
    if "messages" in st.session_state:
        del st.session_state["messages"]


def clear_user_input():
    if "user_input" in st.session_state:
        st.session_state.user_input = ""


# Bug: when I change the character name or change the memory type (after a previous session), then I get

# DuplicateWidgetID: There are multiple identical st.streamlit_chat.streamlit_chat widgets with the same generated key.

# When a widget is created, it's assigned an internal key based on its structure. Multiple widgets with an identical structure will result in the same internal key, which causes this error.

# To fix this error, please pass a unique key argument to st.streamlit_chat.streamlit_chat.

# do I get the error if I enter in the same message twice?


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


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(DATA_ROOT, exist_ok=True)

    st.title("Data Driven Characters")
    st.write("Create your own character chatbots, grounded in existing corpora.")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload corpus")
        if uploaded_file is not None:
            # logging
            corpus_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
            corpus_path = f"{DATA_ROOT}/{corpus_name}.txt"
            output_dir = f"{OUTPUT_ROOT}/{corpus_name}"
            os.makedirs(output_dir, exist_ok=True)
            summaries_dir = f"{output_dir}/summaries"
            character_definitions_dir = f"{output_dir}/character_definitions"
            os.makedirs(character_definitions_dir, exist_ok=True)

            # read file
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()

            # scrollable text
            st.markdown(
                f"""
                <div style='overflow: auto; height: 300px; border: 1px solid gray; border-radius: 5px; padding: 10px'>
                    {string_data}</div>
                """,
                unsafe_allow_html=True,
            )
            # save string_data to a file
            with open(corpus_path, "w") as f:
                f.write(string_data)

            st.divider()

            # get character name
            character_name = st.text_input(f"Enter a character name from {corpus_name}")

            if character_name:
                if (
                    "character_name" in st.session_state
                    and st.session_state["character_name"] != character_name
                ):
                    reset_chat()

                st.session_state["character_name"] = character_name

                with st.spinner("Processing corpus (this will take a while)..."):
                    # load docs
                    docs = load_docs(
                        corpus_path=corpus_path,
                        chunk_size=2048,
                        chunk_overlap=64,
                    )

                    # generate rolling summaries
                    rolling_summaries = get_rolling_summaries(
                        docs=docs, cache_dir=summaries_dir
                    )

                with st.spinner("Generating character definition..."):
                    # get character definition
                    character_definition = get_character_definition(
                        name=character_name,
                        rolling_summaries=rolling_summaries,
                        cache_dir=character_definitions_dir,
                    )
                    print(json.dumps(asdict(character_definition), indent=4))
                    chatbot_type = st.selectbox(
                        "Select a memory type",
                        options=["summary", "retrieval", "generative"],
                        index=0,
                    )
                    if (
                        "chatbot_type" in st.session_state
                        and st.session_state["chatbot_type"] != chatbot_type
                    ):
                        reset_chat()

                    st.session_state["chatbot_type"] = chatbot_type

                    st.markdown(
                        f"[Export to character.ai](https://beta.character.ai/editing):"
                    )
                    st.write(asdict(character_definition))

    if uploaded_file is not None and character_name:
        st.divider()
        chatbot = create_chatbot(
            character_definition=character_definition,
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

        # if user_input and not openai_api_key:
        #     st.info("Please add your OpenAI API key to continue.")

        # the new message
        if user_input:  # and openai_api_key:
            # openai.api_key = openai_api_key
            st.session_state.messages.append({"role": "user", "content": user_input})
            message(user_input, is_user=True)
            with st.spinner(f"{chatbot.character_definition.name} is thinking..."):
                response = chatbot.step(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            message(response)


if __name__ == "__main__":
    main()
