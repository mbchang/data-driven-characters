class CommandLine:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def run(self):
        print(f"{self.chatbot.character_definition.name}: {self.chatbot.greet()}")
        while True:
            text = input("You: ")
            if text:
                print(
                    f"{self.chatbot.character_definition.name}: {self.chatbot.step(text)}"
                )
