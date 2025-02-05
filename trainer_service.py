import torch
import pickle
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import re
import matplotlib.pyplot as plt

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to load data from text files
def load_data(folder):
    data = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:
                        data.append(text)
    return data

# Function to extract keywords from text
def extract_keywords(text):
    keywords = re.findall(r'\b[A-Za-z-]+\b', text)
    return set(word.lower() for word in keywords if len(word) > 2)

# Function to calculate cosine similarity
def calculate_cosine_similarity(resume, job_description):
    inputs_resume = tokenizer(resume, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_job = tokenizer(job_description, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        resume_embeddings = model(**inputs_resume).last_hidden_state[:, 0, :]
        job_embeddings = model(**inputs_job).last_hidden_state[:, 0, :]

    cosine_sim = torch.nn.functional.cosine_similarity(resume_embeddings, job_embeddings)
    return cosine_sim.item()

# Function to prepare data for classification
def prepare_data(resumes, job_descriptions):
    X, y = [], []
    similarities = []

    for resume in resumes:
        for job in job_descriptions:
            similarity = calculate_cosine_similarity(resume, job)
            skill_overlap = len(extract_keywords(resume) & extract_keywords(job)) / len(
                extract_keywords(job)) if extract_keywords(job) else 0
            combined_score = (similarity + skill_overlap) / 2

            similarities.append(combined_score)
            X.append([combined_score])
            y.append(1 if combined_score >= 0.6 else 0)

    # Visualization
    plt.hist(similarities, bins=30, color='skyblue', alpha=0.7)
    plt.title('Similarity Distribution')
    plt.xlabel('Combined Similarity Index')
    plt.ylabel('Frequency')
    plt.show()

    return X, y

# Load data
resumes = load_data(".\\Data_CV")
job_descriptions = load_data(".\\Data_Job")

X, y = prepare_data(resumes, job_descriptions)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
classifier = LogisticRegression(class_weight='balanced', max_iter=1000)
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
with open("trained_model.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)

print("Model successfully trained and saved!")
