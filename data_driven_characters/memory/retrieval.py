from typing import Any, List, Dict
from langchain.memory import VectorStoreRetrieverMemory

from langchain.schema import Document

"""
Human: I like soccer too! But I like Eliza more.
Nick: Oh, really? Eliza's a great girl. She's actually a good friend of mine. What do you like about her?
[24]: Nick seeks advice from Madison about his feelings for Eliza and whether she is jealous of his interest in Hannah. Madison suggests making Eliza jealous, but the plan backfires and Nick realizes his true attraction to Eliza. However, Eliza ignores him for a week, and things are not going well for Carter and Madison either. During a discussion at their regular hangout spot, Nick brings up the idea of feeling stifled by the Bro Code and whether it applies to their situation with Eliza and Hannah. Carter shares his experiences tutoring Madison and how she talks about Nick constantly.

Need to make it more like

[60]\tHuman: I like soccer too! But I like Eliza more.
\tNick: Oh, really? Eliza's a great girl. She's actually a good friend of mine. What do you like about her?
[24]: Nick seeks advice from Madison about his feelings for Eliza and whether she is jealous of his interest in Hannah. Madison suggests making Eliza jealous, but the plan backfires and Nick realizes his true attraction to Eliza. However, Eliza ignores him for a week, and things are not going well for Carter and Madison either. During a discussion at their regular hangout spot, Nick brings up the idea of feeling stifled by the Bro Code and whether it applies to their situation with Eliza and Hannah. Carter shares his experiences tutoring Madison and how she talks about Nick constantly.

So this means we'd have to tag with its buffer idx somehow.
We should probably do this for all retrievers?
"""


class ConversationVectorStoreRetrieverMemory(VectorStoreRetrieverMemory):
    """NOTE: this is tailored specifically for ConversationalChain and ConversationalRetrievalChain."""

    input_prefix = "Human"
    output_prefix = "AI"
    blacklist = []  # gets rid of duplicate key from the other memory

    def _form_documents(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        """Format context from this conversation to buffer."""
        # Each document should only include the current turn, not the chat history
        filtered_inputs = {
            k: v
            for k, v in inputs.items()
            if k != self.memory_key and k not in self.blacklist
        }
        texts = []
        for k, v in list(filtered_inputs.items()) + list(outputs.items()):
            if k == "input":
                k = self.input_prefix
            elif k == "response":
                k = self.output_prefix
            texts.append(f"{k}: {v}")
        page_content = "\n".join(texts)
        return [Document(page_content=page_content)]
