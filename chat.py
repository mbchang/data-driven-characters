import argparse
from dataclasses import asdict
import json
import os

from data_driven_characters.character import get_character_definition
from data_driven_characters.corpus import (
    get_rolling_summaries,
    load_docs,
)
from data_driven_characters.chatbots import SummaryChatBot
from data_driven_characters.chatbots import RetrievalChatBot
from data_driven_characters.chatbots import GenerativeChatBot

OUTPUT_ROOT = "chat"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/the_bro_code.txt")
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--character_name", type=str, default="Nick")
    parser.add_argument("--refresh_decriptions", action="store_true")
    parser.add_argument("--chatbot_type", type=str, default="generative")
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
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
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

    if args.chatbot_type == "summary":
        chatbot = SummaryChatBot(character_definition=character_definition)
    elif args.chatbot_type == "retrieval":
        chatbot = RetrievalChatBot(
            character_definition=character_definition,
            rolling_summaries=rolling_summaries,
        )
    elif args.chatbot_type == "generative":
        chatbot = GenerativeChatBot(
            character_definition=character_definition,
            rolling_summaries=rolling_summaries,
        )
    else:
        raise ValueError(f"Unknown chatbot type: {args.chatbot_type}")

    print(f"{args.character_name}: {chatbot.greet()}")
    while True:
        text = input("You: ")
        if text:
            print(f"{args.character_name}: {chatbot.step(text)}")


if __name__ == "__main__":
    main()
