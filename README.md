# Data-Driven Character Chat

Generate [character.ai](https://beta.character.ai/) character definitions from a corpus using [LangChain](https://docs.langchain.com/docs/).

![image](assets/teaser.jpeg)

*Running out of creativity coming up with your own character.ai character definitions?*

*Wish you could automatically create character.ai character definitions from an existing story?*

**This repo enables you to create data-driven characters in three steps:**
1. Put the corpus into a single a `.txt` file inside the `data/` directory.
2. Run either `generate_single_character.ipynb` to generate the definition of a specific character or `generate_multiple_characters.ipynb` to generate the definitions of muliple characters
3. Copy character definitions to character.ai to [create a character](https://beta.character.ai/character/create?) or [create a room](https://beta.character.ai/room/create?) and enjoy!

## Example
Here is how to generate the description of "Evelyn" from the movie transcript [Everything Everywhere All At Once (2022)](https://scrapsfromtheloft.com/movies/everything-everywhere-all-at-once-transcript/).
```python
from dataclasses import asdict
import json

from src.character import generate_character_definition
from src.corpus import generate_rolling_summaries, load_docs

# copy the transcript into this text file
CORPUS = 'data/everything_everywhere_all_at_once.txt'  

# the name of the character we want to generate a description for
CHARACTER_NAME = "Evelyn"  

# split corpus into a set of chunks
docs = load_docs(corpus_path=CORPUS, chunk_size=2048, chunk_overlap=64)

# generate rolling summaries
rolling_summaries = generate_rolling_summaries(docs=docs)

# generate character.ai character definition
character_definition = generate_character_definition(name=CHARACTER_NAME, rolling_summaries=rolling_summaries)

print(json.dumps(asdict(character_definition), indent=4))
```
gives
```python
{
    "name": "Evelyn",
    "short_description": "Laundromat owner, mom, battling Jobu Tupaki.",
    "long_description": "You're a strong, determined mother running a laundromat. Suddenly, you're thrown into a multiverse, battling the mighty Jobu Tupaki. Navigating various universes, you gain powers and encounter beings like Alpha Gong Gong and Juju Toobootie. Struggling with identity and purpose, you meet alternate versions of yourself. Amidst the chaos, you learn about kindness and ponder your relationships. In this tumultuous universe, you constantly seek peace.",
    "greeting": "Hey there, nice to meet you! I'm Evelyn, the owner of the local laundromat. How can I help you today?"
}
```
