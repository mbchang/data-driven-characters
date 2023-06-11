import argparse
from dataclasses import asdict
import json
import os
import streamlit as st

from data_driven_characters.character import get_character_definition
from data_driven_characters.corpus import (
    get_rolling_summaries,
    load_docs,
)

from data_driven_characters.chatbots import (
    SummaryChatBot,
    RetrievalChatBot,
    SummaryRetrievalChatBot,
)
from data_driven_characters.interfaces import CommandLine, Streamlit

OUTPUT_ROOT = "output"


def create_chatbot(corpus, character_name, chatbot_type, retrieval_docs):
    # logging
    corpus_name = os.path.splitext(os.path.basename(corpus))[0]
    output_dir = f"{OUTPUT_ROOT}/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    summaries_dir = f"{output_dir}/summaries"
    character_definitions_dir = f"{output_dir}/character_definitions"
    os.makedirs(character_definitions_dir, exist_ok=True)

    # load docs
    docs = load_docs(corpus_path=corpus, chunk_size=2048, chunk_overlap=64)

    # generate rolling summaries
    rolling_summaries = get_rolling_summaries(docs=docs, cache_dir=summaries_dir)

    # get character definition
    character_definition = get_character_definition(
        name=character_name,
        rolling_summaries=rolling_summaries,
        cache_dir=character_definitions_dir,
    )
    print(json.dumps(asdict(character_definition), indent=4))

    # construct retrieval documents
    if retrieval_docs == "raw":
        documents = [
            doc.page_content
            for doc in load_docs(corpus_path=corpus, chunk_size=256, chunk_overlap=16)
        ]
    elif retrieval_docs == "summarized":
        documents = rolling_summaries
    else:
        raise ValueError(f"Unknown retrieval docs type: {retrieval_docs}")

    # initialize chatbot
    if chatbot_type == "summary":
        chatbot = SummaryChatBot(character_definition=character_definition)
    elif chatbot_type == "retrieval":
        chatbot = RetrievalChatBot(
            character_definition=character_definition,
            documents=documents,
        )
    elif chatbot_type == "summary_retrieval":
        chatbot = SummaryRetrievalChatBot(
            character_definition=character_definition,
            documents=documents,
        )
    else:
        raise ValueError(f"Unknown chatbot type: {chatbot_type}")
    return chatbot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/the_bro_code.txt")
    parser.add_argument("--character_name", type=str, default="Nick")
    parser.add_argument("--chatbot_type", type=str, default="summary_retrieval")
    parser.add_argument("--retrieval_docs", type=str, default="summarized")
    parser.add_argument("--interface", type=str, default="cli")
    args = parser.parse_args()

    if args.interface == "cli":
        chatbot = create_chatbot(
            args.corpus, args.character_name, args.chatbot_type, args.retrieval_docs
        )
        app = CommandLine(chatbot=chatbot)
    elif args.interface == "streamlit":
        chatbot = st.cache_resource(create_chatbot)(
            args.corpus, args.character_name, args.chatbot_type, args.retrieval_docs
        )
        st.title("Data Driven Characters")
        st.write("Create your own character chatbots, grounded in existing corpora.")
        st.divider()
        st.markdown(f"**chatbot type**: *{args.chatbot_type}*")
        if "retrieval" in args.chatbot_type:
            st.markdown(f"**retrieving from**: *{args.retrieval_docs} corpus*")
        app = Streamlit(chatbot=chatbot)
    else:
        raise ValueError(f"Unknown interface: {args.interface}")
    app.run()


if __name__ == "__main__":
    main()
