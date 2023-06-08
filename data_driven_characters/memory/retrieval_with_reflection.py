from loguru import logger
from datetime import datetime
from pydantic import Field
from typing import Any, Dict, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import Callbacks
from langchain.experimental.generative_agents import (
    GenerativeAgentMemory,
)
from langchain.prompts import PromptTemplate
from langchain.schema import Document


class GenerativeMemory(GenerativeAgentMemory):
    num_topics_of_reflection: int
    num_insights_per_topic: int
    callbacks: Callbacks = Field(default=None, exclude=True)

    # putting this here to be consistent with other memories, even though it is probably not the best way to do it
    input_prefix = "Human"
    output_prefix = "AI"

    insight_num = 1  # : :meta private:
    reflections = []

    input_key = "input"
    response_key = "response"

    @property
    def memory_variables(self) -> List[str]:
        """The list of keys emitted from the load_memory_variables method."""
        return [self.relevant_memories_key]

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, callbacks=self.callbacks
        )

    def pause_to_reflect(self, now: Optional[datetime] = None) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        if self.verbose:
            logger.info("Character is reflecting")
        new_insights = []
        topics = self._get_topics_of_reflection(
            num_topics=self.num_topics_of_reflection,
            last_k=self.memory_retriever.k,
        )
        self.reflections.append({topic: [] for topic in topics})
        logger.info("Topics of reflection:")
        for topic in topics:
            logger.info(f"- {topic}")
        for topic in topics:
            insights = self._get_insights_on_topic(
                topic, num_insights=self.num_insights_per_topic, now=now
            )
            logger.info(f"Insights on {topic}:")
            for insight in insights:
                logger.info(f"- {insight}")
            for insight in insights:
                logger.info("\tAdding insight to memory...")
                self.add_memory(insight, now=now)
            new_insights.extend(insights)
            self.reflections[-1][topic].extend(insights)
        logger.info("Done reflecting")
        logger.info("New insights:")
        for insight in new_insights:
            logger.info(f"- {insight}")
        return new_insights

    def _get_topics_of_reflection(self, num_topics, last_k: int = 50) -> List[str]:
        """Return the num_topics most salient high-level questions about recent observations."""
        prompt = PromptTemplate.from_template(
            f"""{{observations}}

Given only the information above, what are the {num_topics} most salient high-level questions we can answer about the subjects in the statements?
Provide each question on a new line.
        """
        )
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join(
            [self._format_memory_detail(o) for o in observations]
        )
        result = self.chain(prompt).run(observations=observation_str)
        return self._parse_list(result)

    def _get_insights_on_topic(
        self, topic: str, num_insights, now: Optional[datetime] = None
    ) -> List[str]:
        """Generate 'insights' on a topic of reflection, based on pertinent memories."""
        prompt = PromptTemplate.from_template(
            f"""Statements relevant to: '{{topic}}'
---
{{related_statements}}
---
What {num_insights} high-level novel insights can you infer from the above statements that are relevant for answering the following question?
Do not include any insights that are not relevant to the question.
Do not repeat any insights that have already been made.
Format your insights as follows: `Insight <insight_num>: <insight> (because of statements [<statement_num>], [<statement_num>], ...)`
Start with insight_num = {self.insight_num}.

Question: {{topic}}
            """
        )

        related_memories = self.fetch_memories(topic, now=now)
        related_statements = "\n".join(
            [
                self._format_memory_detail(memory)
                for i, memory in enumerate(related_memories)
            ]
        )
        result = self.chain(prompt).run(
            topic=topic, related_statements=related_statements
        )
        # TODO: Parse the connections between memories and insights
        self.insight_num += num_insights
        return self._parse_list(result)

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content = []
        for mem in relevant_memories:
            content.append(self._format_memory_detail(mem))
        return "\n".join([f"{mem}" for mem in content])

    def _format_memory_detail(self, memory: Document) -> str:
        buffer_idx = memory.metadata["buffer_idx"]
        created_time = memory.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
        return f"[{buffer_idx}] [{created_time}] {memory.page_content.strip()}"

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context of this model run to memory."""
        # TODO: fix the save memory key
        mem = outputs.get(self.add_memory_key)
        now = outputs.get(self.now_key)
        if mem:
            self.add_memory(mem, now=now)
        else:
            assert self.response_key in outputs
            assert self.input_key in inputs
            # should I concatenate the inputs here?
            mem = "\n".join(
                [
                    f"{self.input_prefix}: {inputs[self.input_key]}",
                    f"\t{self.output_prefix}: {outputs[self.response_key]}",
                ]
            )
            self.add_memory(mem, now=now)
