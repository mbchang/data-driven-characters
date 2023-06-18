from dataclasses import asdict
from io import StringIO
import json
import os
import streamlit as st

from data_driven_characters.character import generate_character_definition, Character
from data_driven_characters.corpus import (
    generate_corpus_summaries,
    generate_docs,
)
from data_driven_characters.chatbots import (
    SummaryChatBot,
    RetrievalChatBot,
    SummaryRetrievalChatBot,
)
from data_driven_characters.interfaces import reset_chat, clear_user_input, converse


@st.cache_resource()
def create_chatbot(character_definition, corpus_summaries, chatbot_type):
    if chatbot_type == "summary":
        chatbot = SummaryChatBot(character_definition=character_definition)
    elif chatbot_type == "retrieval":
        chatbot = RetrievalChatBot(
            character_definition=character_definition,
            documents=corpus_summaries,
        )
    elif chatbot_type == "summary with retrieval":
        chatbot = SummaryRetrievalChatBot(
            character_definition=character_definition,
            documents=corpus_summaries,
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

    # generate summaries
    corpus_summaries = generate_corpus_summaries(docs=docs, summary_type="map_reduce")
    return corpus_summaries


@st.cache_data(persist="disk")
def get_character_definition(name, corpus_summaries):
    character_definition = generate_character_definition(
        name=name,
        corpus_summaries=corpus_summaries,
    )
    return asdict(character_definition)


def main():
    st.title("Data-Driven Characters")
    st.write(
        "Upload a corpus in the sidebar to generate a character chatbot that is grounded in the corpus content."
    )
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
                <div style='overflow: auto; height: 200px; border: 1px solid gray; border-radius: 5px; padding: 10px'>
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
                    corpus_summaries = process_corpus(corpus)

                with st.spinner("Generating character definition..."):
                    # get character definition
                    character_definition = get_character_definition(
                        name=character_name,
                        corpus_summaries=corpus_summaries,
                    )

                    print(json.dumps(character_definition, indent=4))
                    chatbot_type = st.selectbox(
                        "Select a memory type",
                        options=["summary", "retrieval", "summary with retrieval"],
                        index=2,
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
            corpus_summaries=corpus_summaries,
            chatbot_type=chatbot_type,
        )
        converse(chatbot)


if __name__ == "__main__":
    main()
