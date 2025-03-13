import langchain
import logging
import openai
import os

openai_logger = logging.getLogger('openai')
OpenAI_GPT3_5 = langchain.OpenAI(model_name='text-davinci-003', temperature=0, batch_size=1, max_tokens=2048)

def get_llm():
    openai_logger.propagate = False
    openai.organization = os.getenv("OPENAI_ORGANIZATION")
    return OpenAI_GPT3_5
