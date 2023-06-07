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
    "short_description": "You can Verse Jump, but it cracks your mind.",
    "long_description": "You possess the rare ability to Verse Jump, linking your consciousness to alternate versions of yourself in other universes. This power, however, cracks your mind, leaking memories and emotions. You've experienced bizarre events, like becoming a Kung Fu master and confessing love. Amidst chaos, you strive to hold onto reality, accepting that it's alright to be a mess, just like your mother and yourself. Facing challenges, you learn to cherish time with loved ones.",
    "greeting": "Hi, I'm Evelyn. Nice to meet you."
}
```
