import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

def identity(data):
    return data


def load_prediction_data(filepath):
    with open(filepath,"r") as f:
        data = json.load(f)

    return data


def obtain_llama2_predictions(llama_size):

    filepath = "/nlp/data/kashyap/Dissertation/paralleLM/output_dir/llama2/dementia_" + llama_size + "/predictions.json"

    data = load_prediction_data(filepath)

    hadmid_question_predictions = {}
    hadmid_question_gold_label = {}

    for item in data:
        prediction = item["prediction"]
        model_input = item["inputs"]
        question = item["question"]
        gold_label = item["target"].split(",")[0]
        hadmid = item["target"].split(",")[1]

        if hadmid not in hadmid_question_predictions.keys():
            hadmid_question_predictions[hadmid] = {}

        if question not in hadmid_question_predictions[hadmid].keys():
            hadmid_question_predictions[hadmid][question] = []

        if hadmid not in hadmid_question_gold_label.keys():
            hadmid_question_gold_label[hadmid] = {}

        hadmid_question_predictions[hadmid][question].append(prediction.strip())
        hadmid_question_gold_label[hadmid][question] = gold_label


    temp = 0

    test_labels = []
    pred_labels = []

    for hadmid in hadmid_question_predictions.keys():
        for question in hadmid_question_predictions[hadmid].keys():

            responses = []
            for i in range(len(hadmid_question_predictions[hadmid][question])):

                if hadmid_question_predictions[hadmid][question][i].split(".")[0] in ["Yes","No","Uncertain"]:
                    responses.append(hadmid_question_predictions[hadmid][question][i].split(".")[0])

                elif hadmid_question_predictions[hadmid][question][i].split(",")[0] in ["Yes","No","Uncertain"]:
                    responses.append(hadmid_question_predictions[hadmid][question][i].split(",")[0])

                elif "cannot" in hadmid_question_predictions[hadmid][question][i] or "does not" in hadmid_question_predictions[hadmid][question][i] or "No" in hadmid_question_predictions[hadmid][question][i]:
                    responses.append("No")

                elif "Yes" in hadmid_question_predictions[hadmid][question][i] or "following can be inferred" in hadmid_question_predictions[hadmid][question][i]:
                    responses.append("Yes")

                elif "uncertain" in hadmid_question_predictions[hadmid][question][i].lower():
                    responses.append("Uncertain")

                else:
                    if llama_size == "7b":
                        responses.append("Yes")
                    elif llama_size == "13b":
                        responses.append("No")
                        


            test_labels.append(hadmid_question_gold_label[hadmid][question])

            if "Yes" in responses:
                pred_labels.append("Yes")
            elif "Uncertain" in responses:
                pred_labels.append("Uncertain")
            else:
                pred_labels.append("No")

    label_id = {"No":0,"Uncertain":1,"Yes":2}
    test_labels = [label_id[a] for a in test_labels]
    pred_labels = [label_id[a] for a in pred_labels]


    print("Accuracy: ", accuracy_score(test_labels,pred_labels))
    print("Precision, Recall and F1: ", precision_recall_fscore_support(test_labels,pred_labels,average="weighted"))   


if __name__ == "__main__":
    obtain_llama2_predictions("7b")
    obtain_llama2_predictions("13b")









    