# Data-Driven Characters

Generate [character.ai](https://beta.character.ai/) character definitions from a corpus using [LangChain](https://docs.langchain.com/docs/).

![image](assets/teaser.jpeg)

*Running out of creativity coming up with your own character.ai character definitions?*

*Wish you could automatically create character.ai character definitions from an existing story?*

**This repo enables you to create data-driven characters in three steps:**
1. Put the corpus into a single a `.txt` file inside the `data/` directory.
2. Run either `generate_single_character.ipynb` to generate the definition of a specific character or `generate_multiple_characters.ipynb` to generate the definitions of muliple characters
3. Copy character definitions to character.ai to [create a character](https://beta.character.ai/character/create?) or [create a room](https://beta.character.ai/room/create?) and enjoy!

## Example
Here is how to generate the description of "Evelyn" from the movie [Everything Everywhere All At Once (2022)](https://scrapsfromtheloft.com/movies/everything-everywhere-all-at-once-transcript/).
```python
from dataclasses import asdict
import json

from data_driven_characters.character import generate_character_definition
from data_driven_characters.corpus import generate_rolling_summaries, load_docs

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
    "short_description": "I'm Evelyn, a Verse Jumper exploring universes.",
    "long_description": "I'm Evelyn, able to Verse Jump, linking my consciousness to other versions of me in different universes. This unique ability has led to strange events, like becoming a Kung Fu master and confessing love. Verse Jumping cracks my mind, risking my grip on reality. I'm in a group saving the multiverse from a great evil, Jobu Tupaki. Amidst chaos, I've learned the value of kindness and embracing life's messiness.",
    "greeting": "Hey there, nice to meet you! I'm Evelyn, and I'm always up for an adventure. Let's see what we can discover together!"
}
```
Now you can [chat with Evelyn on character.ai](https://c.ai/c/be5UgphMggDyaf504SSdAdrlV2LHyEgFQZDA5WuQfgw).

## Installation
To install the data_driven_character_chat package, you need to clone the repository and install the dependencies.

You can clone the repository using the following command:

```bash
git clone https://github.com/mbchang/data-driven-characters.git
```
Then, navigate into the cloned directory:

```bash
cd data-driven-character-chat
```
Install the package and its dependencies with:

```bash
pip install -e .
```

## Data
The examples in this repo are movie transcripts taken from [Scraps from the Loft](https://scrapsfromtheloft.com/). However, any text corpora can be used, including books and interviews.

## Characters generated with this repo:
- [Evelyn](https://c.ai/c/be5UgphMggDyaf504SSdAdrlV2LHyEgFQZDA5WuQfgw) (Movie Transcript: [Everything Everywhere All At Once (2022)](https://scrapsfromtheloft.com/movies/everything-everywhere-all-at-once-transcript/))

## Contributing
Contribute your characters with a pull request by placing the link to the character [above](#characters-generated-with-this-repo), along with a link to the text corpus you used to generate them with.

Other pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

<!-- Please make sure to update tests as appropriate. -->

## License
[MIT](LICENSE)

