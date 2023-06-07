from langchain.chat_models import ChatOpenAI

DATA_ROOT = "data"
OUTPUT_ROOT = "output"
GPT3 = ChatOpenAI(model_name="gpt-3.5-turbo")
GPT4 = ChatOpenAI(model_name="gpt-4")
VERBOSE = True
