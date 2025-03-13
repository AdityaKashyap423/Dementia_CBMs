import glob
import json
import re
from collections import defaultdict
from pprint import pprint
from typing import Dict

import langchain
import spacy

from cipd.cipd_runner import CIPDRunner
from cipd.llms import OpenAI_GPT3_5
from cipd.prompts import CIPD_PROMPT_TEMPLATES

VERBOSE = True
NUM_TRIALS = 1

def normalize(x: str):
    """
    Normalizes the input string `x` to follow a specific format.
    
    Args:
    x (str): A string representing a clinical feature description that starts with "The patient exhibits ".
    
    Returns:
    str: The normalized string with the third character in lowercase and the string properly formatted.
    """
    return "The patient exhibits " + x[2].lower() + x[3:] + "."

def standardize_dimensions(all_dimensions):

    """
    Standardizes the clinical feature dimensions by simplifying them and ensuring consistent formatting.

    The function processes the dimensions associated with each condition in `all_dimensions`, simplifies them using a 
    Language Model (GPT-3.5 via the CIPDRunner), and formats them consistently, ensuring each dimension follows a
    specific structure starting with "The patient" and ending with a period. 

    Args:
    all_dimensions (dict): A dictionary where keys are condition names (e.g., clinical conditions) and values are lists 
                            of clinical dimensions (strings) related to that condition.

    Returns:
    dict: A dictionary where the keys are condition names and the values are sets containing the standardized, 
          simplified, and formatted dimensions for each condition.
    """

    # Load the spaCy model for natural language processing
    nlp = spacy.load('en_core_web_sm')

    # Initialize the CIPDRunner, which uses GPT-3.5 to simplify dimensions
    runner = CIPDRunner.from_template(CIPD_PROMPT_TEMPLATES["simplification_prompt"], llm=OpenAI_GPT3_5)

    # Initialize a dictionary to store the simplified dimensions
    new_dimensions = defaultdict(set)

    # Iterate through each condition and its associated dimensions
    for condition_name, dimensions in all_dimensions.items():
        for dimension in dimensions:

            # Simplify each dimension using the CIPDRunner (GPT-3.5)
            simplified_dimensions = runner.predict(sentence=dimension)

            # Process the simplified dimensions by stripping whitespace and converting them into spaCy Doc objects
            simplified_dimensions = [nlp(x.strip()) for x in simplified_dimensions.split("\n") if x.strip() != ""]

            # Split each simplified dimension into sentences (using spaCy's sentence segmentation)
            simplified_dimensions = [list(y.sents) for y in simplified_dimensions]

            # Flatten the list of sentences into a single list of strings
            simplified_dimensions = [str(item) for sublist in simplified_dimensions for item in sublist]

            # Further split the dimensions using the phrase ".The patient" (for multi-sentence dimensions)
            simplified_dimensions = [re.split(".The patient",a) for a in simplified_dimensions]

            # Flatten the list of split dimensions
            simplified_dimensions = [item for sublist in simplified_dimensions for item in sublist]

            # Ensure each dimension ends with a period if it does not already
            simplified_dimensions = [a + "." if a[-1] != "." else a for a in simplified_dimensions]

            # Ensure that each dimension starts with the phrase "The patient"
            simplified_dimensions = [a if "The patient" in a else "The patient" + a for a in simplified_dimensions]

            # Add the processed dimensions to the new_dimensions dictionary for the current condition
            new_dimensions[condition_name].update(simplified_dimensions)

    return new_dimensions

def dedup_cipd_dims(standard_cipd_dims):
    deduped_cipd_dims = set()
    for _, dimensions in standard_cipd_dims.items():
        for dimension in dimensions:
            deduped_cipd_dims.add(dimension)
    # TODO: dedup using sbert embeddings
    with open("data/deduped_cipd_dimensions.json", 'w', encoding='utf-8') as file:
        json.dump(list(deduped_cipd_dims), file)


def build_cipd_from_data_path(data_path="data/textbook-conditions", output_json="data/raw-cipd-dimensions.json") -> Dict:

    """
    Builds the Clinical Item and Patient Dimensions (CIPD) from a collection of condition descriptions stored as text files.
    This function processes the text files, extracts dimensions related to each condition using GPT-3.5, 
    and saves the extracted dimensions along with the condition descriptions to a JSON file.

    Args:
    - data_path (str): The directory path where the text files for each condition are stored. Each file contains a condition name 
      on the first line and the condition's detailed description on subsequent lines.
    - output_json (str): The path to the output JSON file where the extracted dimensions and condition descriptions will be saved.

    Returns:
    - Dict: A dictionary containing the standardized CIPD dimensions, where each key is a condition name and each value is a list of 
      simplified dimensions for that condition.
    """

    # Initialize an NLP runner with a predefined template and model
    runner = CIPDRunner.from_template(CIPD_PROMPT_TEMPLATES["textbook_prompt"], llm=OpenAI_GPT3_5)

    # Dictionary to store extracted dimensions for each condition
    all_dimensions = defaultdict(set)
    output_dict = {}

    # Iterate over all text files in the given data path
    for txt_file in glob.glob(data_path + "/*.txt"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            condition_text = f.readlines()

        # Extract condition name from the first line and description from the rest
        condition_name = condition_text[0].strip()
        condition_description = "\n".join([x.strip() for x in condition_text[1:]])

        candidate_dimensions = set()

        # Run multiple trials to generate candidate dimensions
        for i in range(NUM_TRIALS):
            runner.cache_store.switch_cache(i)
            bullets = runner.predict(condition=condition_name, description=condition_description)

            # Normalize and filter valid candidate dimensions
            candidate_dimensions.update([normalize(x) for x in bullets.split("\n") if x.strip() != ""])

            # Store extracted dimensions for the condition
            all_dimensions[condition_name].update(candidate_dimensions)

            if VERBOSE:
                pprint(f"Condition: {condition_name}")
                pprint(f"Description: {condition_description}")
                pprint(f"Dimensions: {candidate_dimensions}")

        # Merge new candidate dimensions with existing ones for the condition        
        if condition_name in output_dict:
            output_dict[condition_name]["candidate_dimensions"].extend(candidate_dimensions)
            prev_description = output_dict[condition_name]["condition_description"]
            output_dict[condition_name]["condition_description"] = f"{prev_description}\n{condition_description}"
        else:
            output_dict[condition_name] = {"condition_name": condition_name, "condition_description": condition_description, "candidate_dimensions": list(candidate_dimensions)}

    with open(output_json, 'w', encoding='utf-8') as raw_json_file:
        json.dump([v for k, v in output_dict.items()], raw_json_file)

    return standardize_dimensions(all_dimensions)
        
if __name__ == '__main__':
    standard_cipd_dims = build_cipd_from_data_path()
    with open("data/simple-cipd-dimensions.json", 'w', encoding='utf-8') as file:
        json.dump({k: list(v) for k, v in standard_cipd_dims.items()}, file)
    dedup_cipd_dims(standard_cipd_dims)
