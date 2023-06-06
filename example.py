import argparse
from dataclasses import asdict
import json
import os

from src.constants import OUTPUT_ROOT
from src.character import get_character_definition
from src.corpus import get_characters, get_rolling_summaries, load_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="thor_love_and_thunder.txt")
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--num_characters", type=int, default=3)
    args = parser.parse_args()

    # logging
    corpus_name = os.path.splitext(args.corpus)[0]
    output_dir = f"{OUTPUT_ROOT}/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    summaries_dir = f"{output_dir}/summaries"
    character_definitions_dir = f"{output_dir}/character_definitions"
    os.makedirs(character_definitions_dir, exist_ok=True)

    # load docs
    docs = load_docs(
        corpus_name=args.corpus,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # generate rolling summaries
    intermediate_summaries = get_rolling_summaries(docs=docs, cache_dir=summaries_dir)
    rolling_summaries = "\n\n".join(intermediate_summaries)

    # generate list of character
    characters = get_characters(
        rolling_summaries=rolling_summaries,
        num_characters=args.num_characters,
        cache_dir=output_dir,
    )
    print(characters)

    # generate character definitions
    for character in characters:
        character_definition = get_character_definition(
            name=character,
            rolling_summaries=rolling_summaries,
            cache_dir=character_definitions_dir,
        )
        print(json.dumps(asdict(character_definition), indent=4))


if __name__ == "__main__":
    main()
