from typing import Tuple

from langchain import PromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from src.constants import GPT4


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
The description should be written in second-person, as if you were talking to the character.
There is no need to mention the character's name because the character already knows their name.
Do not reference the fact that the character is in a story; talk to the character as if they were a real person.
    """
    )
    description_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message]
    )
    description_chain = LLMChain(llm=GPT4, prompt=description_prompt, verbose=True)
    return description_chain


class FitCharLimit:
    """Fit the character limit to the length of the description."""

    def __init__(
        self,
        chain: Chain,
        character_range: Tuple[int],
        llm: BaseLanguageModel,
        return_intermediate_steps: bool = False,
        verbose: bool = False,
    ):
        self.chain = chain

        assert len(character_range) == 2, "character_range should be two integers"
        assert (
            character_range[0] < character_range[1]
        ), "first element of character_range should be lower than the second element"
        assert (
            character_range[0] >= 0 and character_range[1] >= 0
        ), "both elements of character_range should be non-negative"
        self.character_range = character_range

        self.llm = llm
        self.revision_prompt_template = """
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
        # Re-write the passage to contain between {lo} and {hi} characters while preserving the style and content of the original passage.
        # we know that it is biased to be verbose so we just ask for the lower limit

        self.return_intermediate_steps = return_intermediate_steps
        self.verbose = verbose

    # I suppose if we were to make this modular we would generate some constitutional principles on the fly?
    # no, that won't work, because we would want to give immediate feedback.
    # but in any case, we'd need to have some constraints, make it modular
    # right now we should hack it to make it take both constraints into account, but we should eventually make it modular.
    def run(self, **prompt_kwargs):
        intermediate_steps = []
        response = self.chain.run(**prompt_kwargs)
        if self.return_intermediate_steps:
            intermediate_steps.append(response)
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
            # or you can make this a conversational chain?
            # or you can add memory to this?
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

            if self.return_intermediate_steps:
                intermediate_steps.append(response)
            i += 1
            if self.verbose:
                print(response)
                print(f"Retry {i}: {len(response)} characters.")

        output = {"output": response}
        if self.return_intermediate_steps:
            output.update({"intermediate_steps": intermediate_steps})
        return output
