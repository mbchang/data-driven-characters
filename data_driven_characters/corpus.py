import json
import os

from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from data_driven_characters.constants import VERBOSE


def generate_docs(corpus, chunk_size, chunk_overlap):
    """Generate docs from a corpus."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.create_documents([corpus])
    return docs


def load_docs(corpus_path, chunk_size, chunk_overlap):
    """Load the corpus and split it into chunks."""

    with open(corpus_path) as f:
        corpus = f.read()
    docs = generate_docs(corpus, chunk_size, chunk_overlap)
    return docs


def generate_rolling_summaries(docs):
    """Generate rolling summaries of the story."""
    GPT3 = ChatOpenAI(model_name="gpt-3.5-turbo")
    # chain = load_summarize_chain(
    #     GPT3, chain_type="refine", return_intermediate_steps=True, verbose=True
    # )
    chain = load_summarize_chain(
        GPT3, chain_type="map_reduce", return_intermediate_steps=True, verbose=True
    )
    summary = chain({"input_documents": docs}, return_only_outputs=True)
    intermediate_summaries = summary["intermediate_steps"]
    return intermediate_summaries


def get_rolling_summaries(docs, cache_dir, force_refresh=False):
    """Load the rolling summaries from cache or generate them."""
    if not os.path.exists(cache_dir) or force_refresh:
        os.makedirs(cache_dir, exist_ok=True)
        if VERBOSE:
            print("Summaries do not exist. Generating summaries.")
        intermediate_summaries = generate_rolling_summaries(docs)
        for i, intermediate_summary in enumerate(intermediate_summaries):
            with open(os.path.join(cache_dir, f"summary_{i}.txt"), "w") as f:
                f.write(intermediate_summary)
    else:
        if VERBOSE:
            print("Summaries already exist. Loading summaries.")
        intermediate_summaries = []
        for i in range(len(os.listdir(cache_dir))):
            with open(os.path.join(cache_dir, f"summary_{i}.txt")) as f:
                intermediate_summaries.append(f.read())
    return intermediate_summaries


def generate_characters(rolling_summaries, num_characters):
    """Get a list of characters from a list of rolling summaries."""
    GPT4 = ChatOpenAI(model_name="gpt-4")
    characters_prompt_template = """Consider the following corpus.
    ---
    {rolling_summaries}
    ---
    Give a line-separated list of all the characters, ordered by importance, without punctuation.
    """
    characters = LLMChain(
        llm=GPT4, prompt=PromptTemplate.from_template(characters_prompt_template)
    ).run(rolling_summaries="\n\n".join(rolling_summaries))
    # remove (, ), and " for each element of list
    return characters.split("\n")[:num_characters]


def get_characters(rolling_summaries, num_characters, cache_dir, force_refresh=False):
    cache_file = os.path.join(cache_dir, "characters.json")
    if not os.path.exists(cache_file) or force_refresh:
        characters = generate_characters(rolling_summaries, num_characters)
        with open(cache_file, "w") as f:
            json.dump(characters, f)
    else:
        with open(cache_file, "r") as f:
            characters = json.load(f)
    return characters
