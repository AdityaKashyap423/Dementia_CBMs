from datetime import datetime
import pandas as pd
from collections import Counter
from kani import Kani, chat_in_terminal
from kani.engines.openai import OpenAIEngine
from kani import ChatMessage
import asyncio
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
from sklearn.model_selection import train_test_split
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt 


def run_gpt4_on_dementia_dataset():

    """
    This function processes a dementia dataset created from MIMIC-III by using GPT-4 to analyze clinical notes.

    Parameters:
    None
    
    Returns:
    None
    """

	# Create an argument parser for the OPENAI API key passed via command-line
	parser = argparse.ArgumentParser() 
	parser.add_argument('-k','--key', required=True)
	args = vars(parser.parse_args())
	api_key = str(args["key"])

	# Initialize the OpenAI GPT-4 engine with the provided API key
	engine = OpenAIEngine(api_key, model="gpt-4")

	# Read in a csv file containing "TEXT" (discharge clinical notes) and HADM_ID (patient admission identifiers)
	data_basepath = "NOTES_Control_Dementia_Alzheimer_Vascular.csv"
	data = pd.read_csv(data_basepath)

	notes = list(data["TEXT"])
	hadmids = list(data["HADM_ID"])

	# Asynchronously run the "run_kani" function to process the notes and patient IDs using GPT-4
	asyncio.run(run_kani(notes,hadmids,engine))



async def run_kani(notes,hadmids,engine):

    """
    This asynchronous function processes clinical notes using GPT-4, splits them into parts if necessary,
    and generates responses for each note in the dataset. It also ensures that previously processed HADM_IDs
    are not reprocessed by checking for existing saved files and skips them.

    Parameters:
    - notes (list): A list of clinical notes to be processed.
    - hadmids (list): A list of patient identifiers (HADM_IDs) corresponding to each note.
    - engine (OpenAIEngine): The GPT-4 engine used for generating responses.

    Returns:
    None: The function saves the processed data into a file and does not return anything.
    """


	# Desired maximum length for each note part (in characters)
	desired_part_length = 10000

	# Define the base file name and directory path for saving the results
	savefile = "Part_"
	savedir = "/nlp/data/kashyap/Dissertation/Data/GPT4_Entire_Dementia/"

	# GENERATING LIST OF HADMIDS ALREADY DONE 
	#------------------------------------------------------------------------------------

	# List all files in the save directory that match the savefile prefix
	directory_files = [savedir + a for a in os.listdir(savedir) if savefile in a]

	# Define the path to save the new results, based on how many files already exist
	savepath = savedir + savefile + str(len(directory_files))

	# Load all previously covered HADM_IDs from the existing saved files
	already_covered_hadmids = []
	for file in directory_files:
		with open(file,"rb") as f:
			dir_data = pickle.load(f)
			already_covered_hadmids += list(dir_data.keys())

	# Convert the list of already covered HADM_IDs into a set for faster lookup
	already_covered_hadmids = set(already_covered_hadmids)

	#------------------------------------------------------------------------------------

	# Read the system prompt (used to initiate conversations with GPT-4)
	with open("Data/System_Prompt_2.txt",'r') as f:
		system_prompt = f.read()

	# Load user input data (messages for GPT-4)
	with open("Data/GPT_4_JSON_Input_refined.json","r") as f:
		user_message_list = json.load(f)

	# Initialize a dictionary to store the final results for each HADM_ID
	final_data = {}

	for i in tqdm(range(len(notes))):  

		# Skip the current note if its HADM_ID has already been processed
		if hadmids[i] in already_covered_hadmids:
			print("SKIPPING ",hadmids[i])
			continue 

		# Initialize an empty list for storing responses for each clinical dementia feature for the current HADM_ID
		final_data[hadmids[i]] = []

		# Split the note into smaller parts if it exceeds the desired length
		note_parts = []
		if len(notes[i]) > desired_part_length:
			num_parts = int(len(notes[i])/desired_part_length) + 1
		else:
			num_parts = 1

		# Break the note into parts based on the desired length
		for j in range(num_parts):
			note_parts.append(notes[i][j*desired_part_length:(j+1)*desired_part_length])

		# Initialize chat history with the first part of the clinical note
		chat_history = [ChatMessage.user("Clinical Note: \n" + note_parts[0])]

		# If the note has more than one part, add subsequent parts to the chat history
		if num_parts > 1:
			for item in note_parts[1:]:
				chat_history.append(ChatMessage.user("Continuation of the Clinical Note: \n" + item))

		# Process each user message containing clinical dementia features
		for j in tqdm(range(len(user_message_list))):

			# Initialize a new GPT-4 chat instance
			ai = Kani(engine, system_prompt=system_prompt,chat_history=chat_history.copy()) 

			# Get the user message containing a dementia clinical feature and send it to the GPT-4 engine
			user_message = str(user_message_list[j])
			message = await ai.chat_round_str(user_message)  

			final_data[hadmids[i]].append((user_message_list[j],message))   

		# Save the processed data for the current batch of HADM_IDs into a pickle file
		with open(savepath,"wb") as f:  
			pickle.dump(final_data,f)    


def create_vector_repr(final_data):

    """
    This function converts the GPT-4 clinical dementia feature responses from the provided final_data 
    into a binary vector representation for each patient (identified by HADM_ID). Each 
    concept is represented by a combination of "Yes", "No", and "Uncertain" responses. 
    The function creates a mapping of these combinations to a unique index and produces a 
    binary vector for each patient admission, where each index corresponds to a specific response 
    for a particular clinical feature.

    Parameters:
    - final_data (dict): A dictionary where each key is a HADM_ID (patient identifier) and the value is a list of tuples.
                          Each tuple contains a dementia feature concept and the corresponding response.

    Returns:
    - final_data (dict): A dictionary where each key is a HADM_ID and the value is a binary vector representing the dementia features from GPT-4's response.
    - all_concepts (dict): A dictionary mapping each unique concept-response combination (e.g., "delusions_Yes") to an index.
    """


    # Initialize lists and dictionaries to store all dementia feature concepts and HADM_ID feature data
	all_concepts = []
	hadmid_features = {}

	# Iterate over each patient admission (HADM_ID) in the dementia dataset(final_data)
	for hadmid in final_data.keys():
		concepts_list = []

		# Iterate over each concept-response pair for the current HADM_ID
		for item in final_data[hadmid]:

			# Extract the concept (removing extraneous characters from the string)
			concept = item[0]["Concept Question"][65:-1]

			# Add the possible responses for this concept (Yes, No, Uncertain) to the list of all concepts
			all_concepts += [concept + "_" + a for a in ["Yes","No","Uncertain"]]

			# Determine the response for this concept (Yes, No, or Uncertain)
			if "Uncertain" in item[1]:
				response = "Uncertain"
			elif "Yes" in item[1]:
				response = "Yes"
			else:
				response = "No"

			# Append the (concept, response) tuple to the list of features for the current HADM_ID
			concepts_list.append((concept,response)) 

		# Store the list of features for the current HADM_ID	
		hadmid_features[hadmid] = concepts_list

	# Remove duplicates from the all_concepts list and sort it
	all_concepts = sorted(list(set(all_concepts)))

	# Create a dictionary mapping each unique concept-response combination to a unique index
	all_concepts = {a:i for i,a in enumerate(all_concepts)}

	# Initialize an empty dictionary to store the binary vectors for each HADM_ID
	final_data = {}

	for hadmid in hadmid_features.keys():

		# Initialize a vector of zeros, with length equal to the number of unique concept-response combinations
		temp_vec = [0 for a in range(len(all_concepts))]

		# For each (concept, response) pair for the current HADM_ID, update the binary vector
		for item in hadmid_features[hadmid]:

			# Set the corresponding index in the vector to 1 (marking the presence of this concept-response combination)
			temp_vec[all_concepts[item[0] + "_" + item[1]]] = 1

		# Store the generated binary vector for the current HADM_ID
		final_data[hadmid] = temp_vec

	# Return the binary vector representations for all patients and the dictionary of concept-response mappings
	return final_data, all_concepts





def evaluation():

    """
    This function evaluates the performance of different models for classifying dementia types based on clinical notes. 
    It uses two approaches:
    1. **Concept Bottleneck Approach**: A binary vector representation of clinical dementia features from patient notes is created, and a Logistic Regression model 
       is trained to classify different dementia types.
    2. **Logistic Regression with N-Grams**: A model using n-grams (1 to 3 words) from the raw clinical notes is trained to classify 
       dementia types.

    Parameters:
    None
    
    Returns:
    None
    """

    # File path configurations for storing and loading processed data
	savefile = "Part_"
	savedir = "/nlp/data/kashyap/Dissertation/Data/GPT4_Entire_Dementia/"

	# List all files in the save directory that match the savefile prefix
	directory_files = [savedir + a for a in os.listdir(savedir) if savefile in a]  

	# Define the path for saving the results of the current run
	savepath = savedir + savefile + str(len(directory_files))

	# Load the preprocessed data from saved files and store it in final_data
	final_data = {}	
	for file in tqdm(directory_files):
		with open(file,"rb") as f:
			dir_data = pickle.load(f)
			for hadmid in dir_data.keys():
				final_data[hadmid] = dir_data[hadmid]

	# Generate a vector representation for the concepts from the patient notes
	vect_repr, concept_to_index = create_vector_repr(final_data)
	index_to_concept = {concept_to_index[a]:a for a in concept_to_index.keys()}
	
	# Load the original dataset with clinical notes and corresponding labels
	data_basepath = "NOTES_Control_Dementia_Alzheimer_Vascular.csv"
	all_data = pd.read_csv(data_basepath)

 	# Create dictionaries to store clinical notes and labels for each HADM_ID
	hadmid_notes = {}
	hadmid_labels = {}

	for index,row in all_data.iterrows():
		hadmid = row["HADM_ID"]
		note = row["TEXT"]

		# Clean the clinical note by removing certain terms and digits to prevent information leakage
		note = [re.sub('dementia|alzheimer|vascular|[0-9]', '', a, flags=re.IGNORECASE) for a in note] 
		label = row["Label"]
		hadmid_notes[hadmid] = note
		hadmid_labels[hadmid] = label

	#--------------- CONCEPT BOTTLENECK APPROACH ----------------#

	# Shuffle the HADM_IDs and split them into training and testing sets
	random.seed(42)
	all_hadmids = list(vect_repr.keys())
	random.shuffle(all_hadmids)

	# Split into training (all except last 350) and test (last 350)
	train_hadmids = all_hadmids[:-350]
	test_hadmids = all_hadmids[-350:]

	# Prepare training data (features and labels)
	X_train = []
	y_train = []
	for hadmid in train_hadmids:
		X_train.append(vect_repr[hadmid])
		if "no dementia" in hadmid_labels[hadmid]:
			y_train.append(0)
		elif "alzheimer" in hadmid_labels[hadmid]:
			y_train.append(2)
		elif "vascular" in hadmid_labels[hadmid]:
			y_train.append(3)
		else:
			y_train.append(1)

	# Prepare testing data (features and labels)
	X_test = []
	y_test = []
	for hadmid in test_hadmids:
		X_test.append(vect_repr[hadmid])
		if "no dementia" in hadmid_labels[hadmid]:
			y_test.append(0)
		elif "alzheimer" in hadmid_labels[hadmid]:
			y_test.append(2)
		elif "vascular" in hadmid_labels[hadmid]:
			y_test.append(3)
		else:
			y_test.append(1)

	# Initialize and train a Logistic Regression classifier
	clf = LogisticRegression()
	clf.fit(X_train, y_train)	
	y_pred = clf.predict(X_test)


	print("Accuracy: ",accuracy_score(y_pred,y_test))
	print("Precision, Recall & F1: ", precision_recall_fscore_support(y_test,y_pred,average="weighted"))    
	print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))  


	# Display the most important features for each class (dementia type)
	coefs = clf.coef_
	labels = ["No Dementia","Dementia","Alzheimers Dementia","Vascular Dementia"]
	for i in range(coefs.shape[0]):
		temp_coefs = coefs[i]
		most_imp_inds = np.argsort(-temp_coefs)
		print(labels[i])
		for j in range(10):
			print(index_to_concept[most_imp_inds[j]])
		print("\n\n\n") 



	#------------ SAVE TEST SET FOR GPT-4 BASELINE --------------#

	# Save the test set text and corresponding labels for future GPT-4 baseline evaluation
	save_text = [hadmid_notes[a] for a in test_hadmids]
	temp = {0:"no dementia",1:"dementia",2:"alzheimer's dementia",3:"vascular dementia"}
	save_labels = [temp[a] for a in y_test]

	# Save the test set in a pickle file
	with open("/nlp/data/kashyap/Dissertation/Data/Dementia_GPT4_Baseline_input","wb") as f:
		pickle.dump((save_text,save_labels),f)

	print("Test Set Saved for GPT-4 Baseline!") 


	#--------------- LOGREG NGRAM APPROACH ----------------#

	# Prepare training and testing data using the original clinical notes (without vector representation)
	X_train = [hadmid_notes[a] for a in train_hadmids]
	X_test = [hadmid_notes[a] for a in test_hadmids]

	# Apply a CountVectorizer to create n-grams (1 to 3 words) from the clinical notes
	cv = CountVectorizer(ngram_range=(1,3),max_features=20000,stop_words='english') 
	X_train = cv.fit_transform(X_train)
	X_test = cv.transform(X_test)


	# Train a Logistic Regression model using the n-gram features
	clf = LogisticRegression()
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy: ",accuracy_score(y_pred,y_test))
	print("Precision, Recall & F1: ",precision_recall_fscore_support(y_test,y_pred,average="weighted"))
	print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))


	# Display the most important features (n-grams) for each class (dementia type)
	vocab = cv.vocabulary_
	index_to_feature = {vocab[a]:a for a in vocab.keys()}

	coefs = clf.coef_
	labels = ["No Dementia","Dementia","Alzheimers Dementia","Vascular Dementia"]
	for i in range(coefs.shape[0]):
		temp_coefs = coefs[i]
		most_imp_inds = np.argsort(-temp_coefs)
		print(labels[i])
		for j in range(10):
			print(index_to_feature[most_imp_inds[j]])
		print("\n\n\n")    




def dementia_gpt_baseline():

    """
    This function is used to perform baseline evaluation for dementia classification using GPT-4.
    It loads the test set (clinical notes and corresponding labels) from a pre-saved file and 
    sends the data to the GPT-4 model for inference. The results are processed asynchronously 
    using the `run_gpt4_baseline` function.

    Parameters:
    None
    
    Returns:
    None
    """

    # Load the pre-saved test data (clinical notes and labels)
	with open("/nlp/data/kashyap/Dissertation/Data/Dementia_GPT4_Baseline_input","rb") as f:
		(notes,labels) = pickle.load(f)

	# Set up argument parser to accept the OPENAI API key for accessing GPT-4
	parser = argparse.ArgumentParser() 
	parser.add_argument('-k','--key', required=True)
	args = vars(parser.parse_args())
	api_key = str(args["key"])

	# Initialize the OpenAIEngine with the provided API key and GPT-4 model
	engine = OpenAIEngine(api_key, model="gpt-4")

	# Run the baseline inference using GPT-4 asynchronously with the provided notes and labels
	asyncio.run(run_gpt4_baseline(notes,labels,engine))


async def run_gpt4_baseline(notes,labels,engine):

    """
    This function performs baseline inference using GPT-4 for classifying dementia types from clinical notes. 
    It splits clinical notes into smaller parts if necessary, sends each part to the GPT-4 model, and compares 
    the predicted diagnoses to the true labels.

    Parameters:
    - notes (list of str): The list of clinical notes to be classified.
    - labels (list of str): The true labels for each clinical note (e.g., 'dementia', 'alzheimer's dementia', etc.).
    - engine (OpenAIEngine): The GPT-4 model engine used for inference.

    Returns:
    None
    """

    # Set the maximum length for each note part (to ensure that GPT-4 handles it in manageable chunks)
	desired_part_length = 10000

	# Define the system prompt to instruct GPT-4 on the task
	system_prompt = """For the given patient clinical note, chose one of the following diagnosis labels: 

						no dementia
						dementia
						alzheimer's dementia
						vascular dementia

						You should only reply with the diagnosis label and nothing else."""

	# List to store the final results
	final_data = []

	# Create a mapping from label names to integer indices
	label_to_index = Counter()
	label_to_index["dementia"] = 1
	label_to_index["alzheimer's dementia"] = 2
	label_to_index["vascular dementia"] = 3 

	# Iterate over each clinical note for prediction
	for i in tqdm(range(len(notes))):  

		# Split the note into parts if its length exceeds the maximum allowed length
		note_parts = []
		if len(notes[i]) > desired_part_length:
			num_parts = int(len(notes[i])/desired_part_length) + 1
		else:
			num_parts = 1
		for j in range(num_parts):
			note_parts.append(notes[i][j*desired_part_length:(j+1)*desired_part_length])

		# Build the chat history for GPT-4, sending the first part of the note and continuation parts if applicable
		chat_history = [ChatMessage.user("Clinical Note: \n" + note_parts[0])]
		if num_parts > 1:
			for item in note_parts[1:]:
				chat_history.append(ChatMessage.user("Continuation of the Clinical Note: \n" + item))

			# Create an AI instance (Kani) to interact with GPT-4 and get the diagnosis prediction
			ai = Kani(engine, system_prompt=system_prompt,chat_history=chat_history.copy()) 
			user_message = "What is the patient diagnosis?"

			# Get the response from GPT-4 asynchronously
			message = await ai.chat_round_str(user_message)  

			# Append the predicted and true label (converted to integer indices) to final_data
			final_data.append((label_to_index[message],label_to_index[labels[i]])) 


	y_test = [a[1] for a in final_data]
	y_pred = [a[0] for a in final_data] 
	print("Accuracy: ", accuracy_score(y_pred,y_test))
	print("Precision, Recall and F1: ", precision_recall_fscore_support(y_test,y_pred,average="weighted")) 
	print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))     





if __name__ == "__main__":

	start = datetime.now()

	run_gpt4_on_dementia_dataset()
	evaluation()
	dementia_gpt_baseline()  

	end = datetime.now()

	print("Total Run Time: ", end-start)
	