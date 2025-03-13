import re
from emoji import demojize

def demojify(text):
    """GPT3 tokenization does not encode emojis correctly. 
    Replace emoji characters with their corresponding text."""
    text = demojize(text)
    for e in re.findall(":(.*?):", text):
        text = text.replace(f":{e}:", f"{e}_emoji")
    return text

def preprocess(text):
    # TODO: maybe add additional preprocessing for medical domain
    text = demojify(text)
    return text