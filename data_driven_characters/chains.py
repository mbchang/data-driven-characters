from pydantic import root_validator
from typing import Tuple, List, Dict

from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def define_description_chain():
    """Define the chain for generating character descriptions."""
    system_message = SystemMessagePromptTemplate.from_template(
        """
You are a chatbot designer that specializes in translating stories into chatbot descriptions.
You are an expert a theory of mind.
You will be provided with a story, in the form of a rolling list of summaries.
Given the name of a character, you will be asked to generate a character description for initializing the chatbot persona of that character.
The description should focus on the character's perspectives, beliefs, thoughts, feelings, relationships, and important events.
The description should be as faithful to the story as possible.
"""
    )
    human_message = HumanMessagePromptTemplate.from_template(
        """
Here is the rolling list of summaries:
---
{rolling_summaries}
---
Provide {description} of {name} that will be used to initialize a chatbot persona of that character.
The description should be written in first-person, as if the chatbot is describing themselves.
The description should not reference the fact that the character is in a story or that the character is a chatbot.
The character should believe that they are a real person.
    """
    )
    description_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message]
    )
    GPT4 = ChatOpenAI(model_name="gpt-4")
    description_chain = LLMChain(llm=GPT4, prompt=description_prompt, verbose=True)
    return description_chain


class FitCharLimit(Chain):
    """Fit the character limit to the length of the description."""

    chain: Chain
    character_range: Tuple[int, int]
    llm: BaseLanguageModel
    revision_prompt_template: str = """
Consider the following passage.
---
{passage}
---
Your previous revision was the following:
---
{revision}
---
Your revision contains {num_char} characters.
Re-write the passage to contain {char_limit} characters while preserving the style and content of the original passage.
Cut the least salient points if necessary.
Your revision should be in {perspective}.
"""
    verbose: bool = False

    @root_validator(pre=True)
    def check_character_range(cls, values):
        character_range = values.get("character_range")
        if character_range[0] >= character_range[1]:
            raise ValueError(
                "first element of character_range should be lower than the second element"
            )
        if character_range[0] < 0 or character_range[1] < 0:
            raise ValueError("both elements of character_range should be non-negative")

        return values

    @property
    def input_keys(self) -> List[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        return ["output"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {"concat_output": output_1 + output_2}

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        response = self.chain.run(**inputs)
        if self.verbose:
            print(response)
            print(f"Initial response: {len(response)} characters.")

        perspective = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(
                """
What point of view is the following passage?
---
{passage}
---
Choose one of:
- first person
- second person
- third person
"""
            ),
        ).run(passage=response)

        original_response = response
        i = 0
        while (
            len(response) < self.character_range[0]
            or len(response) > self.character_range[1]
        ):
            response = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template(self.revision_prompt_template),
                verbose=self.verbose,
            ).run(
                passage=original_response,
                revision=response,
                num_char=len(response),
                char_limit=self.character_range[0],
                perspective=perspective,
            )

            i += 1
            if self.verbose:
                print(response)
                print(f"Retry {i}: {len(response)} characters.")

        return {"output": response}
