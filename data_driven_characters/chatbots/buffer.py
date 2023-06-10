from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory


class BufferChatBot:
    """A chatbot that summarizes a conversation.

    This is based on https://python.langchain.com/en/latest/modules/memory/examples/multiple_memory.html
    """

    def __init__(self, character_definition):
        self.character_definition = character_definition
        self.chain = self.create_chain(character_definition)

    def create_chain(self, character_definition):
        GPT3 = ChatOpenAI(model_name="gpt-3.5-turbo")

        memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
        # summary_memory = ConversationSummaryMemory(
        #     llm=GPT3, memory_key="summary", input_key="input"
        # )
        # Combined
        # memory = CombinedMemory(memories=[conv_memory, summary_memory])
        prompt = PromptTemplate.from_template(
            f"""The following is a friendly conversation between a human and an AI.
The AI is a chatbot that has been initialized with the following first-person character description:

Name: {character_definition.name}

Character description:
---
{character_definition.long_description}
---

The AI impersonates the character, {character_definition.name}.
As {character_definition.name}, the AI is talkative and provides lots of specific details from its context.
As {character_definition.name}, the AI engages the human and asks questions to keep the conversation going.
Do not switch roles!

Current conversation:
---
AI: {character_definition.greeting}
{{chat_history}}
---
Human: {{input}}
AI:"""
        )
        chatbot = ConversationChain(
            llm=GPT3, verbose=True, memory=memory, prompt=prompt
        )
        return chatbot

    def greet(self):
        return self.character_definition.greeting

    def step(self, input):
        return self.chain.run(input=input)
