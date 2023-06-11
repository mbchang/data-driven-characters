from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory


class SummaryChatBot:
    def __init__(self, character_definition):
        self.character_definition = character_definition
        self.chain = self.create_chain(character_definition)

    def create_chain(self, character_definition):
        GPT3 = ChatOpenAI(model_name="gpt-3.5-turbo")

        memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
        prompt = PromptTemplate.from_template(
            f"""The following is a friendly conversation between a human and an AI.
The AI is a chatbot that has been initialized with the following first-person character description:

Name: {character_definition.name}

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
