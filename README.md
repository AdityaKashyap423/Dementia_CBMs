# Predicting Explainable Dementia Types with LLM-aided Feature Engineering

## Overview

This repository contains the code for the research paper **Predicting Explainable Dementia Types with LLM-aided Feature Engineering** (yet to be published). The study introduces a novel Large Language Model (LLM)-aided feature engineering pipeline that enhances interpretability in predicting dementia subtypes using clinical notes from Electronic Health Records (EHRs). Our approach achieves superior accuracy compared to baseline models while maintaining interpretability crucial for clinical deployment.

## Key Contributions

* Developed a method to extract dementia-related clinical features from the Oxford Textbook of Medicine using GPT-4.

* Created an LLM-aided feature engineering pipeline that converts clinical notes into concept vector representations.

* Trained a linear, interpretable model to predict dementia subtypes, achieving high accuracy while maintaining transparency.

## Dataset

We use MIMIC-III, a publicly available dataset containing de-identified clinical notes. The dataset includes patient admissions labeled with dementia subtypes using ICD-9 codes:

* Vascular Dementia: 290.40, 290.41, 290.42, 290.43

* Alzheimer’s Dementia: 331.0

* Other Dementias: 290.0, 290.10, 290.11, 290.12, etc.

* No Dementia (Control Group)

## Files

* *CBM.py* contains code to build concept vectors of patient notes using extracted high level clinical concepts from the Oxford Textbook of Medicine.
* *CBM_with_Text_Embeddings.py* contains code to build concept vectors using text embeddings. This reduces the time/cost by 97%.
* The folder *Extracting Dementia Concepts from Textbooks* contains code to extract high level clinical features related to dementia from the Oxford Textbook of Medicine using OPENAI's GPT model
* The folder *Finetuning_Llama2* contains code to finetune 7b and 13b Llama2 models for clinical feature identification in patient notes using knowledge distillation (training data obtained from GPT-4)
  
