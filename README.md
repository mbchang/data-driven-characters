# Data-Driven Characters

<!-- Generate [character.ai](https://beta.character.ai/) character definitions from a corpus using [LangChain](https://docs.langchain.com/docs/). -->
Generate character chatbots from existing corpora.

<!-- probably say something like, the problem is that it is hard to have creavitiy, here we can do it automatically, jut input the text -->

<!-- "My hope is someday, when the next Aristotle is alive, we can capture the underlying worldview of that Aristotle - in a computer. And someday, some student will be able not only to read the words Aristotle wrote, but ask Aristotle a question - and get an answer!"- Steve Jobs, 1985 -->

<!-- https://twitter.com/zekedup/status/1661513104356671492 -->

![image](assets/teaser_chatbot.jpg)

*Running out of creativity coming up with your own chatbot personalities?*

*Wish you could automatically create character chatbots that are directly grounded in an existing story?*

## This repo enables you to create data-driven characters in three steps

<!-- **This repo enables you to create data-driven characters in three steps:** -->
1. Upload a text corpus
2. Specify a character in the corpus
3. Enjoy

This repo provides **three ways** to interact with your data-driven characters:
1. Export to character.ai
2. Run in the command line
3. Run in a streamlit app

This repo supports constructing chatbots with **various kinds of memory management**, including:
- summarization
- retrieval
- retrieval + reflection

## Why data-driven characters?
> I could go read what Aristotle wrote, what Plato wrote, and I didn't have to have an intermediary in the way. And a book was a phenomenal thing. It got right from the source to the destination without anything in the middle! **The problem was you can't ask Aristotle a question.** And I think as we look towards the next fifty to a hundred years, if we really can come up with these machines, that can capture an underlying spirit or an underlying set of principles or an underlying way of looking at the world, then when the next Aristotle comes around, maybe if he carries around one of these machines with him his whole life, his or her whole life, and types in all this stuff. Then maybe someday after the person's dead and gone we can ask this machine, **hey what would Aristotle have said? What about this? And maybe we won't get the right answer, but maybe we will.** And that's really exciting to me.
>
> <p align="right">- Steve Jobs, <a href="https://www.youtube.com/watch?v=vbdy5x5MNS8">International Design Conference, APSEN, 1983</a></p>

<!-- LLMs might capture stuff in pre-training but now recent stuff, can impersonate chatbots
problem: hard to be creative to create new chatbots
solution: ground the chatbots in text corpora -->


1. Put the corpus into a single a `.txt` file inside the `data/` directory.
2. Run either `generate_single_character.ipynb` to generate the definition of a specific character or `generate_multiple_characters.ipynb` to generate the definitions of muliple characters
3. Export character definitions to character.ai to [create a character](https://beta.character.ai/character/create?) or [create a room](https://beta.character.ai/room/create?) and enjoy!

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

# generate character.ai character definition
character_definition = generate_character_definition(
    name=CHARACTER_NAME,
    rolling_summaries=generate_rolling_summaries(docs=docs))

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

## About
This repo generates [character.ai](https://beta.character.ai/) character descriptions from an existing corpus of text. It uses `gpt_3.5_turbo` to summarize the corpus and `gpt4` to generate character descriptions from the corpus. The character descriptions can be direclty copied to create a character on character.ai.

## Installation
To install the data_driven_character_chat package, you need to clone the repository and install the dependencies.

You can clone the repository using the following command:

```bash
git clone https://github.com/mbchang/data-driven-characters.git
```
Then, navigate into the cloned directory:

```bash
cd data-driven-characters
```
Install the package and its dependencies with:

```bash
pip install -e .
```

## Data
The examples in this repo are movie transcripts taken from [Scraps from the Loft](https://scrapsfromtheloft.com/). However, any text corpora can be used, including books and interviews.

## Characters that have been generated with this repo:
- Movie Transcript: [Everything Everywhere All At Once (2022)](https://scrapsfromtheloft.com/movies/everything-everywhere-all-at-once-transcript/)
    - [Evelyn](https://c.ai/c/be5UgphMggDyaf504SSdAdrlV2LHyEgFQZDA5WuQfgw)
    - [Alpha Waymond](https://c.ai/c/5-9rmqhdVPz_MkFxh5Z-zhb8FpBi0WuzDNXF45T6UoI)
    - [Jobu Tupaki](https://c.ai/c/PmQe9esp_TeuLM2BaIsBZWgdcKkQPbQRe891XkLu_NM)

- Movie Transcript: [Thor: Love and Thunder (2022)](https://scrapsfromtheloft.com/movies/thor-love-and-thunder-transcript/)
    - [Thor](https://c.ai/c/1Z-uA7GCTQAFOwGdjD8ZFmdNiGZ4i2XbUV4Xq60UMoU)
    - [Jane Foster](https://c.ai/c/ZTiyQY3D5BzpLfliyhqg1HJzM7V3Fl_UGb-ltv4yUDk)
    - [Gorr the God Butcher](https://c.ai/c/PM9YD-mMxGMd8aE6FyCELjvYas6GLIS833bjJbEhE28)
    - [Korg](https://c.ai/c/xaUrztPYZ32IQFO6wBjn2mk2a4IkfM1_0DH5NAmFGkA)

- Movie Transcript: [Top Gun: Maverick (2022)](https://scrapsfromtheloft.com/movies/top-gun-maverick-transcript/)
    - [Peter "Maverick" Mitchell](https://c.ai/c/sWIpYun3StvmhHshlBx4q2l3pMuhceQFPTOvBwRpl9o)
    - [Bradley "Rooster" Bradshaw](https://c.ai/c/Cw7Nn7ufOGUwRKsQ2AGqMclIPwtSbvX6knyePMETev4)
    - [Admiral Cain](https://c.ai/c/5X8w0ZoFUGTOOghki2QtQx4QSfak2CEJC86Zn-jJCss)

## Contributing
Contribute your characters with a pull request by placing the link to the character [above](#characters-generated-with-this-repo), along with a link to the text corpus you used to generate them with.

Other pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

<!-- Please make sure to update tests as appropriate. -->

## License
[MIT](LICENSE)

