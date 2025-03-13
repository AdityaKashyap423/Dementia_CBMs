from datetime import datetime
import pandas as pd
from collections import Counter
from tqdm import tqdm
import os
import pickle
import time
import numpy as np
import re
from collections import defaultdict
import logging
import json
from sklearn.metrics import accuracy_score 
import argparse
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import confusion_matrix
import random
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


def count_num_tokens():

    """
    This function counts the number of tokens in each sentence from a dataset containing tokenized sentences.
    The function uses the `tiktoken` library to tokenize each sentence and compute the number of tokens.

    Parameters:
    None

    Returns:
    None
    """

	import tiktoken 

	# Specify the encoding name to use for tokenization
	encoding_name = "cl100k_base"
	encoding = tiktoken.get_encoding(encoding_name)

	# Load the dataset containing tokenized sentences and remove any rows with missing data
	data = pd.read_csv("/nlp/data/kashyap/Dissertation/Data/All_Dementia_Data_Sentence_Tokenized.csv").dropna()
	sentences = list(data["SENTENCE"])

	# Initialize lists to store the number of tokens for each sentence
	temp = []
	for sent in tqdm(sentences):

		# Encode the sentence and calculate the number of tokens
		num_tokens = len(encoding.encode(sent))
		temp.append(num_tokens)
 



def get_sentence_embeddings_for_clinical_notes(model_name,batch_size):

    """
    This function generates sentence embeddings for clinical notes using a pre-trained model from the INSTRUCTOR library.
    The embeddings are generated for each sentence in the dataset and saved for further analysis or use in machine learning tasks.

    Parameters:
    - model_name (str): The name of the pre-trained model to be used for generating sentence embeddings.
    - batch_size (int): The batch size to be used when generating embeddings to optimize performance.

    Returns:
    None
    """

	from InstructorEmbedding import INSTRUCTOR

	# Directory where the generated embeddings will be saved
	save_directory = "/nlp/data/kashyap/Dissertation/Data/Dementia_Sentence_Embeddings/"

	# Load the dataset containing the tokenized clinical note sentences
	data = pd.read_csv("/nlp/data/kashyap/Dissertation/Data/All_Dementia_Data_Sentence_Tokenized.csv")

	# Load the pre-trained model using the specified model name
	model = INSTRUCTOR("hkunlp/" + model_name)

	# Define the instruction template to guide the model in generating relevant embeddings
	instruction = "Represent the clinical note sentence for patient feature identification:"

	# Lists to store sentences, HADM_IDs, and the embeddings
	texts_with_instructions = []
	hadmids = []
	sentences = []

	# Iterate over each row in the dataset to prepare the input data for the model
	for index, row in data.iterrows():
		# Add the instruction and the sentence to the texts list
		texts_with_instructions.append([instruction,row["SENTENCE"]])
		hadmids.append(row["HADM_ID"])
		sentences.append(row["SENTENCE"])

	# Generate sentence embeddings using the model, with progress bar and batch processing
	customized_embeddings = model.encode(texts_with_instructions, show_progress_bar=True, batch_size=batch_size)

	# Save the embeddings, HADM_IDs, and sentences to a file for future use
	with open(save_directory + model_name,"wb") as f:
		pickle.dump((sentences,hadmids,customized_embeddings),f)


def get_openai_sentence_embeddings(model_name):

    """
    This function generates sentence embeddings for clinical notes using the OpenAI API. 
    The embeddings are generated for each sentence and saved for future analysis.

    Parameters:
    - model_name (str): The name of the OpenAI model to use for generating sentence embeddings (e.g., "text-embedding-ada-002").

    Returns:
    None
    """


	use_batch = True # Whether to process sentences in batches or one by one
	batch_size = 512 # Batch size used when processing in batches

	num_sent = 3 # Number of sentences to consider for each clinical note chunk

	# Parse the API key for authentication with OpenAI
	parser = argparse.ArgumentParser() 
	parser.add_argument('-k','--key', required=True)
	args = vars(parser.parse_args())
	api_key = str(args["key"])

	# Directory to save the generated embeddings
	save_directory = "/nlp/data/kashyap/Dissertation/Data/Dementia_Sentence_Embeddings/"

	# Helper function to get embeddings for a single or a batch of sentences
	def get_embedding(text, model, use_batch):

		# If not using batch processing, process each sentence individually
		if not use_batch:
			text = text.replace("\n", " ") # Clean up newlines
			return client.embeddings.create(input = [text], model=model).data[0].embedding

		# If using batch processing, process sentences in a list
		else:
			text = [a.replace("\n", " ") for a in text] # Clean up newlines
			return client.embeddings.create(input = text, model=model).data

	# Import necessary OpenAI libraries
	import openai
	from openai import OpenAI

	# Authenticate with the OpenAI API using the provided API key
	openai.api_key = api_key
	client = OpenAI()

	# Load the dataset containing the tokenized clinical notes (sentences)
	data = pd.read_csv("/nlp/data/kashyap/Dissertation/Data/All_Dementia_Data_Sentence_Tokenized_" + str(num_sent) + ".csv") 

	print("Loaded Data!")

	# If not using batch processing, process sentences one by one
	if not use_batch:

		hadmids = list(data["HADM_ID"])
		sentences = list(data["SENTENCE"])
		embeddings = []

		# Iterate over each sentence to generate and collect its embedding
		for sent in tqdm(sentences):
			temp_embedding = get_embedding(sent,model_name, use_batch)
			embeddings.append(temp_embedding)

		# Save the generated embeddings, sentences, and HADM_IDs to a file
		with open(save_directory + model_name,"wb") as f:
			pickle.dump((sentences,hadmids,embeddings),f) 

	# If using batch processing, process sentences in batches to improve performance
	else:
		sentences = list(data["SENTENCE"])
		hadmids = list(data["HADM_ID"])

		# Filter out invalid or overly long sentences
		good_inds = [i for i,a in enumerate(sentences) if type(a) != float and len(a) <8510]
		sentences = [sentences[a] for a in good_inds]
		hadmids = [hadmids[a] for a in good_inds]

		# Initialize lists to store the final sentences, HADM_IDs, and embeddings
		final_sentences = []
		final_hadmids = []
		final_embeddings = []

		# Process sentences in batches of the specified batch size
		for i in tqdm(range(0,len(sentences),batch_size)):

			temp_batch = sentences[i:i+batch_size] # Extract a batch of sentences
			temp_hadmids = hadmids[i:i+batch_size] # Extract corresponding HADM_IDs

			# Get embeddings for the current batch
			temp_embedding = get_embedding(temp_batch,model_name, use_batch)
			temp_embedding = [a.embedding for a in temp_embedding]

			for sent, hadmid, emb in zip(temp_batch, temp_hadmids, temp_embedding):
				final_sentences.append(sent)
				final_hadmids.append(hadmid)
				final_embeddings.append(emb) 

		# Save the final sentences, HADM_IDs, and embeddings after processing all batches
		with open(save_directory + str(num_sent) + "_" + model_name,"wb") as f:
			pickle.dump((final_sentences,final_hadmids,final_embeddings),f)     



def get_feature_embeddings(approach, model_name, batch_size=64):

	"""
    This function generates embeddings for dementia-related features using either OpenAI or Instructor model.
    It retrieves feature descriptions from files, processes them, and saves the generated embeddings.

    Parameters:
    - approach (str): Specifies the method for generating embeddings. It can be either "openai" or "instructor".
    - model_name (str): The name of the model to use for generating embeddings (e.g., "text-embedding-ada-002" for OpenAI, or a model from 'hkunlp' for Instructor).
    - batch_size (int, optional): The batch size used when processing with the Instructor model. Default is 64.

    Returns:
    None
    """

    # Directory to save the generated feature embeddings
	save_directory = "/nlp/data/kashyap/Dissertation/Data/Dementia_Sentence_Embeddings/"

	# Helper function to retrieve dementia features from a text file
	def get_dementia_features():
		filepath = "/nlp/data/kashyap/Dissertation/Data/Dementia_Features.txt"
		with open(filepath,"r") as f:
			data = f.read().split("\n")
		data = [a.split(":")[1].strip() for a in data] # Extract the feature description after ":"
		return data

	# Helper function to get embeddings using OpenAI's API
	def get_embedding(text, model): 
		text = text.replace("\n", " ") # Remove newlines to prevent errors
		return client.embeddings.create(input = [text], model=model).data[0].embedding

	# Helper function to retrieve dementia feature definitions from a JSON file
	def get_dementia_features_definition():
		filepath = "/nlp/data/kashyap/Dissertation/Data/GPT_4_JSON_Input_refined.json"

		with open(filepath) as f:
			data = json.load(f)

		features = [a["Concept Question"] for a in data] # Extract concept questions
		features = [a[65:-1] for a in features] # Clean up the feature text by removing unwanted parts

		definitions = [a["Definition"] for a in data]  # Extract the definitions of the features

		return features, definitions

	# Flag to choose whether to use definitions or dementia feature names
	use_definitions = True

	# If using definitions, combine feature questions with definitions
	if use_definitions:
		features, definitions = get_dementia_features_definition() 
		feature_index = {a:i for i,a in enumerate(features)} 

		features = [a + "\n" + b for a,b in zip(features,definitions)]

	else:
		features = get_dementia_features()
		feature_index = {a:i for i,a in enumerate(features)}


	# Handle embedding generation based on the chosen approach: "openai" or "instructor"
	if approach == "openai":

		# Parse the API key for OpenAI
		parser = argparse.ArgumentParser() 
		parser.add_argument('-k','--key', required=True)
		args = vars(parser.parse_args())
		api_key = str(args["key"])

		# Import OpenAI libraries and initialize the client
		import openai
		from openai import OpenAI

		openai.api_key = api_key
		client = OpenAI()

		# Initialize a dictionary to store the embeddings for each feature
		feature_embeddings = {}

		# Generate embeddings for each feature using OpenAI API
		for feature in tqdm(features):
			feature_embeddings[feature] = get_embedding(feature,model_name)

		# Save the generated embeddings and their indices to a pickle file
		with open(save_directory + "Features_Definitions_" + model_name,"wb") as f:
			pickle.dump((feature_index,feature_embeddings),f)  

	elif approach == "instructor":

		from InstructorEmbedding import INSTRUCTOR

		 # Initialize the model from the specified name
		model = INSTRUCTOR("hkunlp/" + model_name)

		# Define an instruction for embedding the features (used by Instructor model)
		instruction = "Represent the patient feature for clinical note feature identification:"

		# Prepare the list of features with instructions for embedding
		texts_with_instructions = []
		sentences = []
		for feature in features:
			texts_with_instructions.append([instruction,feature])
			sentences.append(feature)

		# Generate embeddings for the features using the Instructor model
		customized_embeddings = model.encode(texts_with_instructions, show_progress_bar=True, batch_size=batch_size)

		# Save the generated embeddings, feature indices, and sentences to a pickle file
		with open(save_directory + "Features_" + model_name,"wb") as f:
			pickle.dump((feature_index, customized_embeddings, sentences),f)  



def get_gpt4_pred_for_testset():

    """
    This function is responsible for obtaining predictions from GPT-4 for clinical feature annotations based on a given test set. 
    It can either process new test data (using GPT-4 to predict clinical features) or retrieve previously saved predictions.
    
    The function has two modes of operation:
    - 'process': This mode retrieves new test data and uses GPT-4 (via Kani) to make predictions.
    - 'retrieve': This mode loads previously generated predictions from a file and processes them for further use.
    
    Parameters:
    None 

    Returns:
    - In 'retrieve' mode, returns a dictionary `final_data` mapping `hadmid` to the predicted feature responses.
    """	

    # Set the mode of operation (either "process" or "retrieve")
	mode = "retrieve"


	if mode == "process":

		from kani.engines.openai import OpenAIEngine
		import asyncio

		# Command-line argument parsing for API key
		parser = argparse.ArgumentParser() 
		parser.add_argument('-k','--key', required=True)
		args = vars(parser.parse_args())
		api_key = str(args["key"])

		# Initialize the OpenAI engine with the provided API key and model as "gpt-4"
		engine = OpenAIEngine(api_key, model="gpt-4")

		# Load test data from a CSV file (clinical notes and features)

		# filepath = "/nlp/data/kashyap/Dissertation/Data/Gold_Standard_Labels.csv" 
		filepath = "/nlp/data/kashyap/Dissertation/Data/Additional_annotation/Combined_Gold_Labels_2.csv" 
		test_data = pd.read_csv(filepath,low_memory=False).dropna() 

		# Extract `hadmid`, `note`, and `clinical_feature` columns for further processing
		hadmids = list(test_data["hadmid"])
		notes = list(test_data["note"])
		features = list(test_data["clinical_feature"])

		# Run the GPT-4 model through the `run_kani` function asynchronously, which processes the notes, hadmids, and features.
		asyncio.run(run_kani(notes,hadmids,features,engine))

	elif mode == "retrieve":

		# Specify the file path for retrieving previously saved GPT-4 predictions

		# filepath = "/nlp/data/kashyap/Dissertation/Data/GPT_4_TestData_Preds"
		filepath = "/nlp/data/kashyap/Dissertation/Data/Additional_annotation/GPT_4_TestData_Preds_2"
		# filepath = "/nlp/data/kashyap/Dissertation/Data/Additional_annotation/GPT_4_TestData_Preds_No_Definition"

		# Load the saved prediction data from the specified file
		with open(filepath,"rb") as f:
			data = pickle.load(f)

		# Initialize an empty dictionary to store the final formatted prediction results
		final_data = {}

		# Process the loaded data and convert the raw predictions into a structured format
		for hadmid in data.keys():

			# Initialize the `hadmid` entry in the final data dictionary if it doesn't already exist
			if hadmid not in final_data.keys():
				final_data[hadmid] = {}

			# Iterate over the features and predicted responses for the current `hadmid`
			for item in data[hadmid]:
				feature = item[0]

				# Extract and clean the response for the current feature (Yes/No/Uncertain)
				if "\"Yes\"" in item[1]:
					response = "Yes"
				elif "\"No\"" in item[1]:
					response = "No"
				elif "\"Uncertain\"" in item[1]:
					response = "Uncertain"
				else:
					print("ERROR")
					exit() # If an invalid response format is found, terminate the program

				# Store the response for the feature in the final data dictionary under the corresponding `hadmid`	
				final_data[hadmid][feature] = response

		# Return the final formatted predictions for all test samples
		return final_data


async def run_kani(notes,hadmids, features, engine):
    """
    This asynchronous function processes clinical notes, generates GPT-4 responses for clinical feature predictions,
    and saves the results in a specified location. It breaks the clinical notes into smaller chunks if they exceed the 
    desired length, sends them along with associated features to GPT-4, and stores the responses.

    Parameters:
    - notes: A list of clinical notes.
    - hadmids: A list of patient admission IDs associated with the notes.
    - features: A list of clinical features to predict.
    - engine: An instance of the OpenAI GPT-4 engine to generate predictions.

    Returns:
    - None. The predictions are saved to a file.
    """

	from kani import Kani, chat_in_terminal
	from kani import ChatMessage

	# Set the desired length of clinical notes to be sent to GPT-4
	desired_part_length = 10000

	# Path to save the final predictions

	# savepath = "/nlp/data/kashyap/Dissertation/Data/GPT_4_TestData_Preds"
	savepath = "/nlp/data/kashyap/Dissertation/Data/Additional_annotation/GPT_4_TestData_Preds_2"

	# Load system prompt for GPT-4, which guides how GPT-4 should interpret the clinical notes
	with open("Data/System_Prompt_2.txt",'r') as f:
		system_prompt = f.read()

	# Load the user message list, which contains feature-question pairs
	with open("Data/GPT_4_JSON_Input_refined.json","r") as f:
		user_message_list = json.load(f)


	# Map each feature to its corresponding GPT-4 input
	feature_to_gpt4_input = {}

	for feature in features:

		# Find the corresponding question in the JSON input list for each feature
		for item in user_message_list:
			if feature in item["Concept Question"]:
				feature_to_gpt4_input[feature] = item  # Map the feature to its corresponding input
				break

	# Initialize the dictionary to store the final results
	final_data = {}

	# Process each clinical note and generate GPT-4 predictions for each feature
	for note, hadmid, feature in tqdm(zip(notes, hadmids, features),total=len(notes)):

		# Initialize a new entry in final_data for the current `hadmid` if not already present
		if hadmid not in final_data.keys():
			final_data[hadmid] = []

		# Split the clinical note into smaller parts if it exceeds the desired length
		note_parts = []
		if len(note) > desired_part_length:
			num_parts = int(len(note)/desired_part_length) + 1
		else:
			num_parts = 1

		for j in range(num_parts):
			# Append each part of the note (or the entire note if not split) to note_parts
			note_parts.append(note[j*desired_part_length:(j+1)*desired_part_length])

		# Create a list of messages for the chat history (for GPT-4 processing)
		chat_history = [ChatMessage.user("Clinical Note: \n" + note_parts[0])]
		if num_parts > 1:
			# If the note was split, include the continuation parts in the chat history
			for item in note_parts[1:]:
				chat_history.append(ChatMessage.user("Continuation of the Clinical Note: \n" + item))

		# Extract the corresponding user message for the current feature
		user_message = str(feature_to_gpt4_input[feature])

		# Create an instance of the Kani class to interact with GPT-4
		ai = Kani(engine, system_prompt=system_prompt,chat_history=chat_history.copy()) 

		# Send the user message to GPT-4 and get the response
		message = await ai.chat_round_str(user_message)  

		final_data[hadmid].append((feature,message))   

		# Save the predictions to the specified file path
		with open(savepath,"wb") as g:  
			pickle.dump(final_data,g)      



def change_format_of_pickle(model_name): 

    """
    This function changes the format of the pickle file containing sentence embeddings into two separate files for faster loading:
    1. A NumPy file for embeddings.
    2. A pickle file for associated patient IDs (hadmids).

    The function operates in two modes:
    - "save" mode: Converts pickle files into separate files (embedding arrays and patient IDs).
    - "load" mode: Loads the previously saved files (embedding arrays and patient IDs).

    Parameters:
    - model_name (str): The name of the model (used as part of the file name) to load/save the embeddings and hadmids.
    
    Returns:
    - None. The function either saves or loads the data depending on the mode.
    """

    # Set mode to 'save' to convert and save files, or 'load' to load the files
	mode = "save"

	# Directory path where the data is saved or loaded from
	dir_path = "/nlp/data/kashyap/Dissertation/Data/Dementia_Sentence_Embeddings/"

	print("LOADING: ",dir_path + model_name, "......")


	if mode == "save":

		# Load the pickle file containing sentences, hadmids, and embeddings
		with open(dir_path + model_name,"rb") as f:
			(sentences,hadmids,embeddings) = pickle.load(f)

		print("LOADED ",dir_path + model_name)

		embeddings = np.array(embeddings)

		print("CONVERTED TO NP ARRAY")

		# Save the embeddings as a .npy file (NumPy format)
		np.save(dir_path + model_name + "_1",embeddings)

		print("SAVED ", dir_path + model_name + "_1")

		# Save hadmids as a separate pickle file
		with open(dir_path + model_name + "_2","wb") as f:
			pickle.dump(hadmids,f)  

		print("SAVED ", dir_path + model_name + "_2")


	elif mode == "load":

		# Load the embeddings from the .npy file
		embeddings = np.load(dir_path + model_name + "_1.npy")

		# Load the hadmids from the corresponding pickle file
		with open(dir_path + model_name + "_2", "rb") as f:
			hadmids = pickle.load(f)




def obtain_embeddings(data_type, model_name):

    """
    This function retrieves embeddings for either clinical notes or features, based on the specified data type and model name.

    The function handles two types of data:
    1. Clinical Notes (`data_type="Notes"`)
    2. Features (`data_type="Features"`)

    The function supports different models for retrieving the embeddings in different formats.

    Parameters:
    - data_type (str): The type of data for which embeddings are being obtained. It can either be "Notes" or "Features".
    - model_name (str): The name of the model whose embeddings are being used.

    Returns:
    - A tuple containing:
      - hadmids: Patient IDs (if `data_type="Notes"`)
      - embeddings: The model-generated embeddings for either the clinical notes or features.
      - feature_index: The feature index for features data (if `data_type="Features"`).
    """

    # Directory path where the data is stored
	dir_path = "/nlp/data/kashyap/Dissertation/Data/Dementia_Sentence_Embeddings/"

	# Case when the embeddings are for clinical notes
	if data_type == "Notes":

		if model_name in ["instructor-base", "instructor-large", "instructor-xl"]:
			with open(dir_path + model_name,"rb") as f:
				data = pickle.load(f)
			(sentences,hadmids,embeddings) = data
			return hadmids, embeddings
			
		elif model_name in ["text-embedding-3-small","text-embedding-3-large" ,"text-embedding-ada-002","2_text-embedding-3-large","3_text-embedding-3-large", "4_text-embedding-3-large", "5_text-embedding-3-large"]:
			embeddings = np.load(dir_path + model_name + "_1.npy")

			with open(dir_path + model_name + "_2", "rb") as f:
				hadmids = pickle.load(f)

			return hadmids, embeddings

	# Case when the embeddings are for dementia features 
	elif data_type == "Features":

		with open(dir_path + "Features_" + model_name,"rb") as f: 
			data = pickle.load(f)		

		if model_name in ["instructor-base", "instructor-large", "instructor-xl"]:
			(feature_index, embeddings, sentences) = data
			return feature_index, embeddings


		elif model_name in ["text-embedding-3-small","text-embedding-3-large" ,"text-embedding-ada-002", "2_text-embedding-3-large","3_text-embedding-3-large", "4_text-embedding-3-large", "5_text-embedding-3-large"]:

			(feature_index,embeddings) = data

			# Create an index mapping from feature index to feature name
			index_feature = {feature_index[feature]:feature for feature in feature_index.keys()}

			# Create a list to store final embeddings in the correct order
			final_embeddings = []

			# Rearrange the embeddings based on the feature index
			for i in range(len(index_feature)):
				final_embeddings.append(embeddings[index_feature[i]])

			final_embeddings = np.array(final_embeddings)

			return feature_index, final_embeddings



def get_gold_labels_test():

    """
    This function reads a CSV file containing gold standard labels for clinical features and 
    structures the data into a dictionary format, where the key is the patient ID (hadmid) 
    and the value is another dictionary with clinical features and their corresponding gold labels.


    Returns:
    - final_data (dict): A nested dictionary where the outer key is the hadmid (patient ID), 
                         and the inner dictionary contains clinical features as keys and their 
                         corresponding gold labels as values.
    """

    # File path to the gold standard labels CSV file

	# filepath = "/nlp/data/kashyap/Dissertation/Data/Gold_Standard_Labels.csv"
	filepath = "/nlp/data/kashyap/Dissertation/Data/Additional_annotation/Combined_Gold_Labels_2.csv"

	data = pd.read_csv(filepath).dropna()

	# Initialize an empty dictionary to store the final structured data
	final_data = {}
	for index,row in data.iterrows():
		hadmid = row["hadmid"]
		feature = row["clinical_feature"]
		label = row["gold_label"]

		# If the hadmid is not already in the final_data dictionary, add it
		if hadmid not in final_data.keys():
			final_data[hadmid] = {}

		# Add the clinical feature and its corresponding label to the dictionary for this patient
		final_data[hadmid][feature] = label

	return final_data
		



def perform_RAG_testdata(model_name,temp=None,prob_threshold=0):

	"""
    This function uses Text Embeddings to reduce the number of calls to GPT-4 for clinical feature prediction 
    based on cosine similarity between note embeddings and feature embeddings.

    The function evaluates the model by comparing its predictions with the gold standard labels and calculates
    various performance metrics (accuracy, precision, recall, F1-score, confusion matrix). It also tracks the
    number of GPT-4 annotations required based on the cosine distance threshold.

    Parameters:
    ----------
    model_name : str
        The name of the model used to obtain note and feature embeddings.
    temp : str, optional
        A temporary prefix for the model name (default is None).
    prob_threshold : float, optional
        The cosine distance threshold used to consider relevant notes for a feature (default is 0).

    Returns:
    -------
    tuple
        A tuple containing:
        - accuracy score (float): The accuracy of the model's predictions.
        - num_annotations_req (int): The number of annotations required (i.e., those with cosine distances below the threshold).
        - tot_num (int): The total number of predictions made.

	"""

	# Obtain note embeddings based on the model name (and optionally the prefex `temp`)
	if temp:
		hadmids, note_embeddings = obtain_embeddings("Notes",str(temp) + "_" + model_name)
	else:
		hadmids, note_embeddings = obtain_embeddings("Notes",model_name)

	# Load gold labels for the test data
	hadmid_feature_gold_labels = get_gold_labels_test()

	# Filter test hadmids that are present in the gold labels
	test_hadmids = [a for a in hadmids if a in hadmid_feature_gold_labels.keys()]
	test_hadmid_inds = [i for i,a in enumerate(hadmids) if a in hadmid_feature_gold_labels.keys()]
	test_hadmid_inds_map = {a:i for i,a in enumerate(test_hadmids)} 

	# Filter note embeddings for the relevant hadmids
	note_embeddings = note_embeddings[test_hadmid_inds,:]

	# Obtain feature embeddings
	feature_index, feature_embeddings = obtain_embeddings("Features",model_name)
	index_feature = {feature_index[feature]:feature for feature in feature_index.keys()}
	
	# Calculate cosine distance between feature embeddings and note embeddings
	dist_matrix = cosine_distances(feature_embeddings,note_embeddings)

	# Sort the cosine distance matrix to retrieve the most relevant notes for each feature
	dist_matrix_argsort = np.argsort(dist_matrix,axis=1)

	# Get GPT-4 model predictions
	gpt4_response = get_gpt4_pred_for_testset()

	# Flatten the distance matrix and count the number of annotations required based on threshold
	temp_list = dist_matrix.tolist()
	temp_list = [x for xs in temp_list for x in xs]
	tot_num = len(temp_list)
	num_annotations_req = sum([1 for a in temp_list if a < prob_threshold])

	# Prepare to generate predictions and calculate evaluation metrics
	y_test = []
	y_pred = []

	for hadmid in tqdm(list(hadmid_feature_gold_labels.keys())):
		for feature in hadmid_feature_gold_labels[hadmid].keys():
			gold_label = hadmid_feature_gold_labels[hadmid][feature]
			y_test.append(gold_label)

			feature_index_value = feature_index[feature]
			hadmid_inds = dist_matrix_argsort[feature_index_value]

			considered_hadmids = set()
			for ind in hadmid_inds:
				if dist_matrix[feature_index_value][ind] < prob_threshold:
					considered_hadmids = considered_hadmids | {test_hadmids[ind]}   
				else:
					break

			if hadmid not in considered_hadmids:
				y_pred.append("No")
			else: 
				y_pred.append(gpt4_response[hadmid][feature])   

	# Map the labels ("No", "Uncertain", "Yes") to numerical values
	label_map = {"No":0, "Uncertain": 1, "Yes":2}
	y_test = [label_map[a] for a in y_test]
	y_pred = [label_map[a] for a in y_pred]


	print(model_name, ": THRESHOLD: ", prob_threshold)
	print("Accuracy Score: ", accuracy_score(y_test,y_pred))
	print("Precision, Recall and F1: ", precision_recall_fscore_support(y_test,y_pred,average="weighted"))
	print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred)) 
	print("Number of Annotations Required: ", num_annotations_req," out of ",tot_num)
	print("\n\n\n\n")   

	# Return the evaluation results
	return accuracy_score(y_test,y_pred), num_annotations_req, tot_num 


def obtain_num_notes_per_feature(approach=None):

    """
    This function calculates the number of notes per feature based on GPT-4 responses in clinical annotations.
    It reads the annotation files from a specified directory, counts the occurrences of each clinical feature,
    and adjusts the frequency based on the responses: "Yes", "No", or "Uncertain". It provides a feature-frequency
    mapping, where the key is a clinical feature, and the value is the number of notes associated with that feature.

    Parameters:
    ----------
    approach : str, optional
        If specified, alters the behavior of frequency calculation. If "constant" is provided, 
        each feature will have a fixed frequency of 745 notes, overriding the calculated frequency based on responses.

    Returns:
    -------
    feature_freq : dict
        A dictionary where the keys are clinical features (as strings) and the values are the number of notes 
        associated with each feature. The frequency is calculated based on GPT-4 annotations and adjusted by the 
        `approach` parameter if provided.
    """

    # Defining the path and directory to read annotation files.
	savefile = "Part_"
	savedir = "/nlp/data/kashyap/Dissertation/Data/GPT4_Entire_Dementia/"

	# Total number of notes (used in frequency calculation).
	total_number_of_notes = 7450

	# Getting all files in the directory that match the savefile pattern.
	directory_files = [savedir + a for a in os.listdir(savedir) if savefile in a]

	# Dictionary to store all data loaded from files.
	all_data = {}

	# Loop through all files and load the data into all_data dictionary.
	for file in tqdm(directory_files):
		with open(file,"rb") as f:
			dir_data = pickle.load(f)
		
			for hadmid in dir_data.keys():
				all_data[hadmid] = dir_data[hadmid]

	num_files = len(all_data)

	# Counters for frequency calculation of features and responses.
	feature_freq = Counter()
	temp = Counter()

	# Loop through all the data and count occurrences of features and their responses.
	for hadmid in all_data.keys():
		for item in all_data[hadmid]:

			# Extract the clinical feature from the response.
			feature = item[0]["Concept Question"][65:-1]
			gpt4_response = item[1]

			# Count occurrences of different responses.
			if "\"Yes\"" in gpt4_response:
				feature_freq[feature] += 1
				temp["Yes"] += 1

			elif "\"Uncertain\"" in gpt4_response:
				feature_freq[feature] += 1
				temp["Uncertain"] += 1

			elif "\"No\"" in gpt4_response:
				feature_freq[feature] += 0
				temp["No"] += 1

	# Normalize the feature frequency by dividing by the number of files and scaling to the total number of notes.
	for feature in feature_freq.keys():
		feature_freq[feature] = max([feature_freq[feature],1])/num_files
		feature_freq[feature] = max([int(feature_freq[feature] * total_number_of_notes),1])

		# If "constant" approach is selected, set all feature frequencies to 745.
		if approach == "constant":
			feature_freq[feature] = 745 

	return feature_freq 



def obtain_additional_test_data_for_annotation():

    """
    This function generates additional test data for manual annotation. It samples clinical notes for specific features
    related to dementia and creates a dataset that can be used for annotation (via MTurk). The features are 
    predefined, and the function ensures that a specific number of notes per feature are selected. The resulting
    data includes the HADM_ID, clinical note, and associated clinical feature for annotation.

	Parameters:
	----------
	None

    Returns:
    -------
    None
    """

    # Set random seed for reproducibility.
	random.seed(42)

	# Helper function to load dementia features and their definitions from a JSON file.
	def get_dementia_features_definition():
		filepath = "/nlp/data/kashyap/Dissertation/Data/GPT_4_JSON_Input_refined.json"

		with open(filepath) as f:
			data = json.load(f)

		# Extract the clinical features and their corresponding definitions from the data.
		features = [a["Concept Question"] for a in data]
		features = [a[65:-1] for a in features]
		definitions = [a["Definition"] for a in data]

		return features, definitions


	filepath = "/nlp/data/kashyap/Dissertation/NOTES_Control_Dementia_Alzheimer_Vascular.csv"
	
	# Load the clinical notes dataset from a CSV file.
	data = pd.read_csv(filepath)

	# Map each HADM_ID to the corresponding clinical note.
	all_hadmids = list(data["HADM_ID"])
	hadmid_to_note = {}
	for index,row in data.iterrows():
		hadmid_to_note[row["HADM_ID"]] = row["TEXT"]

	# Retrieve the dementia features and definitions.
	features, definitions = get_dementia_features_definition()
	
	# Predefined list of considered features that we want to manually annotate.
	considered_features = ["The patient exhibits fluctuating cognitive performance.",
	"The patient exhibits difficulty answering questions.",
	"The patient exhibits reduced speech output.",
	"The patient exhibits forgetfulness.",
	"The patient exhibits bladder dysfunction.",
	"The patient exhibits white-matter changes.",
	"The patient exhibits delirium.",
	"The patient exhibits a loss of libido.",
	"The patient exhibits substance dependence.",
	"The patient exhibits low energy."]

	# Predefined number of annotations already available for each feature.
	number_already_annotated = [9,6,5,5,5,4,4,4,1,2]

	# Calculate the number of additional notes required for each feature.
	num_notes_required = [20-a for a in number_already_annotated]

	# Print the total number of notes needed for all features.
	print(np.sum(num_notes_required))

	# Final data list to store the selected clinical notes for each feature.
	final_data = []

	# Loop through each considered feature and select the required number of notes.
	for i in range(len(considered_features)):
		current_feature = considered_features[i]

		# Randomly sample the required number of HADM_IDs for the current feature.
		considered_hadmids = np.random.choice(all_hadmids,num_notes_required[i],replace=False)

		# Get the corresponding clinical notes for the selected HADM_IDs.
		considered_notes = [hadmid_to_note[a] for a in considered_hadmids]

		# Loop through the selected HADM_IDs and notes to structure the data for each entry.
		for j in range(len(considered_hadmids)):
			temp = {}
			temp["hadmid"] = considered_hadmids[j]
			temp["clinical_note"] = re.sub("\n|\r","<br>",considered_notes[j]) 
			temp["clinical_feature"] = current_feature
			final_data.append(temp) 

	final_data = pd.DataFrame(final_data)
	final_data.to_csv("/nlp/data/kashyap/Dissertation/Data/Additional_annotation/Additional_Annotation_MTurk_input.csv",index=False)


def combine_final_testset():

    """
    This function combines multiple sources of manual annotation data into a final dataset. It reads annotation 
    results from two CSV files containing MTurk output and merges them with additional gold standard labels 
    provided in another CSV file. The final dataset includes clinical notes, associated features, and gold 
    standard labels for further analysis or evaluation.

	Parameters:
	----------
	None

    Returns:
    -------
    None
    - The function saves the combined dataset as a CSV file (`Combined_Gold_Labels_2.csv`).
    """

    # Predefined list of clinical features to consider for gold standard labels.
	considered_features = ["The patient exhibits fluctuating cognitive performance.",
	"The patient exhibits difficulty answering questions.",
	"The patient exhibits reduced speech output.",
	"The patient exhibits forgetfulness.",
	"The patient exhibits bladder dysfunction.",
	"The patient exhibits white-matter changes.",
	"The patient exhibits delirium.",
	"The patient exhibits a loss of libido.",
	"The patient exhibits substance dependence.",
	"The patient exhibits low energy."]

	# Read the two MTurk output files into dataframes.
	data_1 = pd.read_csv("/nlp/data/kashyap/Dissertation/Data/Additional_annotation/Additional_Annotation_MTurk_output.csv")
	data_2 = pd.read_csv("/nlp/data/kashyap/Dissertation/Data/Additional_annotation/2_concept_MTurk_output.csv")

	final_data = []

	# Process each of the two dataframes (MTurk outputs) to extract relevant information.
	for data in [data_1,data_2]:
		for index,row in data.iterrows():
			temp = {}
			temp["note"] = row["Input.clinical_note"]
			temp["hadmid"] = row["Input.hadmid"]
			temp["clinical_feature"] = row["Input.clinical_feature"]

			# Extract the gold label based on the MTurk answer options.
			if row["Answer.option1.No"]:
				temp["gold_label"] = "No"
			elif row["Answer.option1.Not Sure (Possible Correlation)"]:
				temp["gold_label"] = "Uncertain"
			elif row["Answer.option1.Yes"]:
				temp["gold_label"] = "Yes"
			elif row["Answer.option1.Not Sure (Medical Jargon)"]:
				temp["gold_label"] = "Uncertain"

			final_data.append(temp)


	final_data = pd.DataFrame(final_data)

	# Filter the gold standard labels to only include the considered features.
	data_2 = pd.read_csv("/nlp/data/kashyap/Dissertation/Data/Gold_Standard_Labels.csv")
	data_2 = data_2[data_2["clinical_feature"].isin(considered_features)]

	final_data = pd.concat([final_data,data_2])

	final_data.to_csv("/nlp/data/kashyap/Dissertation/Data/Additional_annotation/Combined_Gold_Labels_2.csv",index=False)



def evaluate():

    """
    This function evaluates the performance of a model by comparing its predictions against gold standard labels.
    It computes several evaluation metrics (accuracy, precision, recall, F1 score, confusion matrix) for each clinical 
    feature in the dataset. The function processes the gold labels and model predictions, and then outputs the results 
    for each feature.

	Parameters:
	----------
	None

    Returns:
    -------
    None
    - The function prints the evaluation metrics for each feature but does not return any values.
    """

    # Retrieve gold labels and GPT-4 predictions for the test set.
	gold_labels = get_gold_labels_test()
	predictions = get_gpt4_pred_for_testset()

	# Initialize lists to store predictions and true labels, as well as feature-specific data.
	pred_list = []
	test_list = []

	# Dictionaries to store predictions and gold labels for each feature separately.
	pred_feature_list = {}
	test_feature_list = {}

	# Mapping gold label values to numerical values (for evaluation).
	label_map = {"No":0,"Uncertain":1,"Yes":2}

	# Iterate over the gold labels and predictions for each patient (HADM_ID) and feature.
	for hadmid in gold_labels.keys():
		for feature in gold_labels[hadmid].keys():

			# Initialize lists for the features if not already done.
			if feature not in pred_feature_list.keys():
				pred_feature_list[feature] = []
			if feature not in test_feature_list.keys():
				test_feature_list[feature] = []

			# Convert gold labels and predictions to numerical values using label_map.
			pred_list.append(label_map[predictions[hadmid][feature]]) 
			test_list.append(label_map[gold_labels[hadmid][feature]])

			# Store the numerical values for each feature separately.
			pred_feature_list[feature].append(label_map[predictions[hadmid][feature]])
			test_feature_list[feature].append(label_map[gold_labels[hadmid][feature]])


	for feature in pred_feature_list.keys():
		preds = pred_feature_list[feature]
		tests = test_feature_list[feature]

		print(feature)
		print("Accuracy: ", accuracy_score(preds,tests))
		print("Precision, Recall and F1: ", precision_recall_fscore_support(tests,preds,average='weighted'))
		print("Confusion Matrix:\n", confusion_matrix(tests,preds))
		print("\n\n\n")       


def additional_concept_annotation_2():

    """
    This function prepares a dataset for additional concept manual annotation. It samples a set of clinical 
    notes corresponding to the features and prepares them in a format that can be used for further annotation 
    using Mechanical Turk. The final dataset is saved as a CSV file for annotation.

	Parameters:
	----------
	None

    Returns:
    -------
    None
    - The function saves the resulting dataset to a CSV file and prints the shape of the dataset.
    """

    # Set the random seed for reproducibility
	random.seed(42)

	# Helper function to load dementia feature definitions from a JSON file
	def get_dementia_features_definition():
		filepath = "/nlp/data/kashyap/Dissertation/Data/GPT_4_JSON_Input_refined.json"

		with open(filepath) as f:
			data = json.load(f)

		# Extract features (concept questions) and their definitions
		features = [a["Concept Question"] for a in data]
		features = [a[65:-1] for a in features]
		definitions = [a["Definition"] for a in data]

		return features, definitions


	filepath = "/nlp/data/kashyap/Dissertation/NOTES_Control_Dementia_Alzheimer_Vascular.csv"

	# Load the clinical notes data
	data = pd.read_csv(filepath) 
	all_hadmids = list(data["HADM_ID"])

	# Create a mapping from HADM_ID to the corresponding clinical note
	hadmid_to_note = {}
	for index,row in data.iterrows():
		hadmid_to_note[row["HADM_ID"]] = row["TEXT"]

	features, definitions = get_dementia_features_definition()
	
	# Define the list of considered features and the number of notes required for each feature
	considered_features = ["The patient exhibits forgetfulness.","The patient exhibits substance dependence."]

	num_notes_required = [30,30]


	final_data = []
	for i in range(len(considered_features)):
		current_feature = considered_features[i]

		# Randomly select HADM_IDs that will be used for annotation
		considered_hadmids = np.random.choice(all_hadmids,num_notes_required[i],replace=False)
		considered_notes = [hadmid_to_note[a] for a in considered_hadmids]

		# Loop over the selected HADM_IDs and prepare the data for annotation
		for j in range(len(considered_hadmids)):
			temp = {}
			temp["hadmid"] = considered_hadmids[j]
			temp["clinical_note"] = re.sub("\n|\r","<br>",considered_notes[j]) 
			temp["clinical_feature"] = current_feature
			final_data.append(temp) 

	final_data = pd.DataFrame(final_data)
	final_data.to_csv("/nlp/data/kashyap/Dissertation/Data/Additional_annotation/2_concept_MTurk_input.csv",index=False)


def plot_RAG_results():

    """
    This function loads metadata related to our Text Embedding approach's results at various cosine distance thresholds.
    It then generates two plots: 
    - Plot of "Cosine Distance Threshold vs Test Accuracy"
    - Plot of "Cosine Distance Threshold vs Fraction of Dataset Annotated by GPT-4"

    The function saves these plots as PNG images in the specified directory.
    """

	with open("/nlp/data/kashyap/Dissertation/Data/plots/MetaData","rb") as f:
		(x_plot,x1,y_plot,tot_num) = pickle.load(f)

	plt.plot(x_plot,y_plot)
	plt.xlabel("Cosine Distance Threshold")
	plt.ylabel("Test Accuracy")
	plt.savefig("/nlp/data/kashyap/Dissertation/Data/plots/RAG_Threshold_vs_Acc.png")

	x1 = [a/tot_num for a in x1]

	plt.plot(x_plot,x1)
	plt.xlabel("Cosine Distance Threshold")
	plt.ylabel("Fraction of the Dataset Annotated by GPT-4")
	plt.savefig("/nlp/data/kashyap/Dissertation/Data/plots/Threshold_vs_Annotation_Fraction.png")




if __name__ == "__main__":

	start = datetime.now()

	# plot_RAG_results() 

	# evaluate()

	# obtain_additional_test_data_for_annotation()
	# additional_concept_annotation_2() 
	# combine_final_testset() 

	# ------------------------------GET INSTRUCTOR NOTE EMBEDDINGS------------------------------
	# model_name = "instructor-base"
	# model_name = "instructor-large"
	# model_name = "instructor-xl"

	# batch_size = 128

	# get_sentence_embeddings_for_clinical_notes(model_name,batch_size=batch_size)  

	# ------------------------------GET OPENAI NOTE EMBEDDINGS------------------------------

	# model_name = "text-embedding-3-small"  
	# model_name = "text-embedding-3-large"    
	# model_name = "text-embedding-ada-002"    

	# get_openai_sentence_embeddings(model_name) 


	# ------------------------------GET FEATURE EMBEDDINGS------------------------------

	# approach = "instructor" 
	# approach = "openai"

	# if approach == "instructor":
		# model_name = "instructor-base"
		# model_name = "instructor-large"
		# model_name = "instructor-xl"
		# pass

	# elif approach == "openai":
		# model_name = "text-embedding-3-small"
		# model_name = "text-embedding-3-large" 
		# model_name = "text-embedding-ada-002"  

	# get_feature_embeddings(approach, model_name)  

	# --------------------------------------------------------------------------------------

	# get_gpt4_pred_for_testset() 

	# ------------------------------RUN ENTIRE RAG PIPELINE------------------------------


	# model_name = "instructor-base" 
	# model_name = "instructor-large"
	# model_name = "instructor-xl"
	# model_name = "text-embedding-3-small" 
	model_name = "text-embedding-3-large"          
	# model_name = "text-embedding-ada-002" 

	# obtain_embeddings(data_type, model_name)
	# change_format_of_pickle(model_name)     
	# perform_RAG_testdata(model_name)       
	# obtain_num_notes_per_feature() 

	x_plot = []
	x1 = []
	y_plot = []
	for prob_threshold in range(1,100):
		acc,num_annotations,tot_num = perform_RAG_testdata(model_name,prob_threshold=prob_threshold/100)
		x1.append(num_annotations)
		x_plot.append(prob_threshold/100)
		y_plot.append(acc)

	with open("/nlp/data/kashyap/Dissertation/Data/plots/MetaData","wb") as f:
		pickle.dump((x_plot,x1,y_plot,tot_num),f)

	plt.plot(x_plot,y_plot)
	plt.xlabel("Probability Threshold")
	plt.ylabel("Test Accuracy")
	plt.savefig("/nlp/data/kashyap/Dissertation/Data/RAG_Threshold_vs_Acc.png")  




	end = datetime.now()
	print("Total Run Time: ", end-start)




