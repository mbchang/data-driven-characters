from dataclasses import dataclass, asdict
import json
import os

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

from data_driven_characters.chains import FitCharLimit, define_description_chain

from data_driven_characters.constants import VERBOSE
from data_driven_characters.utils import (
    order_of_magnitude,
    apply_file_naming_convention,
)


@dataclass
class Character:
    name: str
    short_description: str
    long_description: str
    greeting: str


def generate_character_ai_description(name, rolling_summaries, char_limit):
    """Generate a character description with a certain number of characters."""
    lower_limit = char_limit - 10 ** (order_of_magnitude(char_limit))

    description_chain = define_description_chain()
    GPT4 = ChatOpenAI(model_name="gpt-4")
    char_limit_chain = FitCharLimit(
        chain=description_chain,
        character_range=(lower_limit, char_limit),
        llm=GPT4,
        verbose=VERBOSE,
    )
    description = char_limit_chain.run(
        rolling_summaries="\n\n".join(rolling_summaries),
        description=f"{lower_limit}-character description",  # specify a fewer characters than the limit
        name=name,
    )
    return description


def generate_greeting(name, short_description, long_description):
    """Generate a greeting for a character."""
    greeting_template = """Here are a short and long description for a character named {name}:

Short description:
---
{short_description}
---

Long description:
---
{long_description}
---

Generate a greeting that {name} would say to someone they just met, without quotations.
This greeting should reflect their personality.
"""
    GPT3 = ChatOpenAI(model_name="gpt-3.5-turbo")
    greeting = LLMChain(
        llm=GPT3, prompt=PromptTemplate.from_template(greeting_template)
    ).run(
        name=name,
        short_description=short_description,
        long_description=long_description,
    )
    # strip quotations
    greeting = greeting.replace('"', "")
    return greeting


def generate_character_definition(name, rolling_summaries):
    """Generate a Character.ai definition."""
    short_description = generate_character_ai_description(
        name=name, rolling_summaries=rolling_summaries, char_limit=50
    )
    long_description = generate_character_ai_description(
        name=name, rolling_summaries=rolling_summaries, char_limit=500
    )
    greeting = generate_greeting(name, short_description, long_description)

    # populate the dataclass
    character_definition = Character(
        name=name,
        short_description=short_description,
        long_description=long_description,
        greeting=greeting,
    )
    return character_definition


def get_character_definition(name, rolling_summaries, cache_dir, force_refresh=False):
    """Get a Character.ai definition from a cache or generate it."""
    cache_path = f"{cache_dir}/{apply_file_naming_convention(name)}.json"

    if not os.path.exists(cache_path) or force_refresh:
        character_definition = generate_character_definition(name, rolling_summaries)
        with open(cache_path, "w") as f:
            json.dump(asdict(character_definition), f)
    else:
        with open(cache_path, "r") as f:
            character_definition = Character(**json.load(f))
    return character_definition
