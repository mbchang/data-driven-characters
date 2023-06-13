from setuptools import setup, find_packages

setup(
    name="data_driven_characters",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'faiss-cpu',
        'langchain',
        'loguru',
        'openai',
        'streamlit_chat',
        'tiktoken',
        'tqdm',
    ],
)
