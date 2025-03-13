import logging
import re

import openai
from langchain import LLMChain, PromptTemplate
from tenacity import (after_log, before_sleep_log, retry,
                      retry_if_exception_type, wait_exponential)
from cipd.utils import CacheStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CIPDRunner:

    """
    A class to extract patient dimensions using a large language model with prompt-based prediction.

    Attributes:
        cache_store (CacheStore): A caching mechanism to store responses for multiple trials.
        llm_chain (LLMChain): A chain consisting of a prompt template and an LLM for text generation.

    Methods:
        from_template(template_text_path, llm):
            Creates a CIPDRunner instance from a given template file.

        predict(**kwargs):
            Uses the LLM to generate predictions based on the provided input variables.
    """

    def __init__(self, prompt: PromptTemplate, llm, num_trials=1):
        self.cache_store = CacheStore(num_trials=num_trials)
        self.llm_chain = LLMChain(prompt=prompt, llm=llm)

    @staticmethod
    def from_template(template_text_path, llm):
        """
        Creates a CIPDRunner instance from a given prompt template file.

        Args:
            template_text_path (str): Path to the text file containing the prompt template.
            llm: The language model used for text generation.

        Returns:
            CIPDRunner: An instance of CIPDRunner initialized with the parsed template and LLM.
        """

        with open(template_text_path, 'r', encoding='utf-8') as file:
            template = file.read().replace('\n', '').strip()

        # Extract input variable names enclosed in curly braces
        input_variables = re.findall(r"\{(.*?)\}", template)

        # Create a PromptTemplate instance with extracted variables
        prompt = PromptTemplate(
            input_variables=input_variables, template=template
        )
        return CIPDRunner(prompt, llm=llm)

    @retry(
        retry=retry_if_exception_type(openai.error.RateLimitError),
        wait=wait_exponential(multiplier=1, min=10, max=60),
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
    )
    @retry(
        retry=retry_if_exception_type(openai.error.ServiceUnavailableError),
        wait=wait_exponential(multiplier=1, min=3, max=10),
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
    )
    
    def predict(self, **kwargs):
        return self.llm_chain(kwargs)['text'].strip()
