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
    GenerativeChatBot,
)
from data_driven_characters.interfaces import CommandLine, Streamlit

OUTPUT_ROOT = "chat"


def create_chatbot(character_definition, rolling_summaries, chatbot_type, corpus_path):
    if chatbot_type == "summary":
        chatbot = SummaryChatBot(character_definition=character_definition)
    elif chatbot_type == "retrieval":
        chatbot = RetrievalChatBot(
            character_definition=character_definition,
            rolling_summaries=rolling_summaries,
        )
    elif chatbot_type == "retrieval_raw":
        docs = load_docs(
            corpus_path=corpus_path,
            chunk_size=256,
            chunk_overlap=16,
        )
        chatbot = RetrievalChatBot(
            character_definition=character_definition,
            rolling_summaries=[doc.page_content for doc in docs],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/the_bro_code.txt")
    parser.add_argument("--character_name", type=str, default="Nick")
    parser.add_argument("--refresh_decriptions", action="store_true")
    parser.add_argument("--chatbot_type", type=str, default="generative")
    parser.add_argument("--interface", type=str, default="commandline")
    args = parser.parse_args()

    # logging
    corpus_name = os.path.splitext(os.path.basename(args.corpus))[0]
    output_dir = f"{OUTPUT_ROOT}/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    summaries_dir = f"{output_dir}/summaries"
    character_definitions_dir = f"{output_dir}/character_definitions"
    os.makedirs(character_definitions_dir, exist_ok=True)

    # load docs
    docs = load_docs(
        corpus_path=args.corpus,
        chunk_size=2048,
        chunk_overlap=64,
    )

    # generate rolling summaries
    rolling_summaries = get_rolling_summaries(docs=docs, cache_dir=summaries_dir)

    # get character definition
    character_definition = get_character_definition(
        name=args.character_name,
        rolling_summaries=rolling_summaries,
        cache_dir=character_definitions_dir,
        force_refresh=args.refresh_decriptions,
    )
    print(json.dumps(asdict(character_definition), indent=4))

    if args.interface == "commandline":
        chatbot = create_chatbot(
            character_definition=character_definition,
            rolling_summaries=rolling_summaries,
            chatbot_type=args.chatbot_type,
            corpus_path=args.corpus,
        )
        app = CommandLine(chatbot=chatbot)
    elif args.interface == "streamlit":
        chatbot = st.cache_resource(create_chatbot)(
            character_definition=character_definition,
            rolling_summaries=rolling_summaries,
            chatbot_type=args.chatbot_type,
            corpus_path=args.corpus,
        )
        app = Streamlit(chatbot=chatbot)
    else:
        raise ValueError(f"Unknown interface: {args.interface}")
    app.run()


if __name__ == "__main__":
    main()
