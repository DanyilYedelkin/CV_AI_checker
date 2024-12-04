import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt

# Loading the model and the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Functions for loading data
def load_job_descriptions(data_job_folder):
    job_descriptions = []
    for filename in os.listdir(data_job_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(data_job_folder, filename), 'r', encoding='utf-8') as file:
                job_text = file.read().strip()
                job_descriptions.append(job_text)
    return job_descriptions

def load_resumes(data_cv_folder):
    resumes = []
    for filename in os.listdir(data_cv_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(data_cv_folder, filename), 'r', encoding='utf-8') as file:
                resumes.append(file.read().strip())
    return resumes

# Function for calculating cosine similarity using the [CLS] token
def calculate_cosine_similarity(resume, job_description):
    inputs_resume = tokenizer(resume, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_job = tokenizer(job_description, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        resume_embeddings = model(**inputs_resume).last_hidden_state[:, 0, :]
        job_embeddings = model(**inputs_job).last_hidden_state[:, 0, :]

    cosine_sim = torch.nn.functional.cosine_similarity(resume_embeddings, job_embeddings)
    return cosine_sim.item()

# Preparation of data for classification
def prepare_data_for_classification(resumes, job_descriptions):
    X, y = [], []
    similarities = []

    for resume in resumes:
        for job_description in job_descriptions:
            similarity = calculate_cosine_similarity(resume, job_description)
            similarities.append(similarity)
            X.append([similarity])
            y.append(1 if similarity >= 0.8 else 0)  # The threshold can be varied

    # Visualization of the distribution of similarities
    plt.hist(similarities, bins=30, color='skyblue', alpha=0.7)
    plt.title('Distribution of Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.show()

    # Class diagnostics
    print(f"Class distribution in y: {Counter(y)}")
    return X, y

# Data loading
data_job_folder = os.path.join(os.getcwd(), "..", "data_job")
data_cv_folder = os.path.join(os.getcwd(), "..", "data_cv")

resumes = load_resumes(data_cv_folder)
job_descriptions = load_job_descriptions(data_job_folder)

# Data preparation
X, y = prepare_data_for_classification(resumes, job_descriptions)

if len(set(y)) == 1:
    print("Error: all labels belong to the same class.")
else:
    # Separation of data into training and test samples, taking into account class proportions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic regression training
    classifier = LogisticRegression(class_weight='balanced', max_iter=1000)
    classifier.fit(X_train, y_train)

    # Model estimation
    y_pred = classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Prediction function
    def predict_fit(resume, job_description):
        similarity = calculate_cosine_similarity(resume, job_description)
        prediction = classifier.predict([[similarity]])
        return "Fit" if prediction[0] == 1 else "Not Fit"

    # Example of use
    result = predict_fit(resumes[0], job_descriptions[0])
    print(f"Does the candidate fit? {result}")