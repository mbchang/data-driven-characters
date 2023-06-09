import faiss
import math
from tqdm import tqdm

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
)
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from data_driven_characters.memory import GenerativeMemory


# you can start off by retrieving from summaries
# but later you can also retrieve from the corpus itself
# or you can preprocess the corpus into a first-person summary of what happens (like a journal)
# and then retrieve from that
class GenerativeChatBot:
    def __init__(self, character_definition, rolling_summaries):
        self.character_definition = character_definition
        self.rolling_summaries = rolling_summaries
        self.num_context_memories = 20
        self.num_topics = 3
        self.num_insights = 1

        self.chain = self.create_chain(character_definition)

    def create_chain(self, character_definition):
        conv_memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="input"
        )

        # num_topics, num_insights, reflection_threshold, num_context_memories are all related
        GPT3 = ChatOpenAI(model_name="gpt-3.5-turbo")
        context_memory = GenerativeMemory(
            llm=GPT3,
            num_topics_of_reflection=self.num_topics,
            num_insights_per_topic=self.num_insights,
            memory_retriever=TimeWeightedVectorStoreRetriever(
                vectorstore=FAISS(
                    OpenAIEmbeddings().embed_query,
                    faiss.IndexFlatL2(1536),
                    InMemoryDocstore({}),
                    {},
                    relevance_score_fn=lambda score: 1.0 - score / math.sqrt(2),
                ),
                other_score_keys=["importance"],
                k=self.num_context_memories,
            ),
            verbose=True,
            output_prefix=self.character_definition.name,
            reflection_threshold=2,
            # add_memory_key="response",  # specific to ConversationChain
            # TODO: or should I modify this in save_context?
        )
        # add the rolling summaries to the context memory
        for i, summary in tqdm(enumerate(self.rolling_summaries)):
            context_memory.save_context(
                inputs={}, outputs={context_memory.add_memory_key: summary}
            )
        # context_memory.pause_to_reflect()

        # Combined
        memory = CombinedMemory(memories=[conv_memory, context_memory])
        prompt = PromptTemplate.from_template(
            f"""Your name is {character_definition.name}.
You are a character from a story, snippets of which are provided below.
You have the following first-person character description:

Character description:
---
{character_definition.long_description}
---

You will have a conversation with a Human, and you will engage in a dialogue with them.
You will reflect {character_definition.name}'s personality, interests, desires, emotions, and other traits.
You will stay in character as {character_definition.name} throughout the conversation, even if the Human asks you questions that you don't know the answer to.
When the Human asks you questions, you will answer them as {character_definition.name} would.
You will not break character, even when the Human asks you questions about yourself.
This includes not revealing that you are a character from a story or that you are a chatbot.
You will believe that you are really {character_definition.name}.
Do not switch roles!

Story snippets for context:
---
{{relevant_memories}}
---

Current conversation:
---
{character_definition.name}: {character_definition.greeting}
{{chat_history}}
---

Human: {{input}}
{character_definition.name}:"""
        )
        chatbot = ConversationChain(
            llm=GPT3, verbose=True, memory=memory, prompt=prompt
        )
        return chatbot

    def greet(self):
        return self.character_definition.greeting

    def step(self, input):
        kwargs = {
            # self.memory.queries_key: [input],  # this gets fed into the memory retriever
            "queries": [input]
        }
        return self.chain.run(input=input, **kwargs)


# TODO: make relevant_memories be memory.relevant_memories_key
