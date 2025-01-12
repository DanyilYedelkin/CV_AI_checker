import os
import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from collections import Counter
import numpy as np
import re
import matplotlib.pyplot as plt

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to load resumes (good and bad) per job
def load_resumes(data_cv_folder):
    resumes, labels = [], []
    for root, dirs, files in os.walk(data_cv_folder):
        for file in files:
            if file.endswith(".txt"):
                label = 1 if "good" in root.lower() else 0
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read().strip()  # Read content of the file
                        if data:  # Ensure the file is not empty
                            resumes.append(data)
                            labels.append(label)
                        else:
                            print(f"Warning: File {file_path} is empty.")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    return resumes, labels

# Function to load job descriptions
def load_job_descriptions(data_job_folder):
    job_descriptions = {}
    for root, dirs, files in os.walk(data_job_folder):
        for file in files:
            if file.endswith(".txt"):
                job_id = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read().strip()  # Read content of the file
                        if data:  # Ensure the file is not empty
                            job_descriptions[job_id] = data
                        else:
                            print(f"Warning: File {file_path} is empty.")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    return job_descriptions

# Extract keywords from text
def extract_keywords(text):
    keywords = re.findall(r'\b[A-Za-z-]+\b', text)
    return set([word.lower() for word in keywords if len(word) > 2])

# Calculate skill overlap
def calculate_skill_overlap(resume, job_description):
    resume_keywords = extract_keywords(resume)
    job_keywords = extract_keywords(job_description)
    common_keywords = resume_keywords.intersection(job_keywords)
    return len(common_keywords) / len(job_keywords) if len(job_keywords) > 0 else 0

# Calculate cosine similarity using BERT embeddings
def calculate_cosine_similarity(resume, job_description):
    inputs_resume = tokenizer(resume, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_job = tokenizer(job_description, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        resume_embeddings = model(**inputs_resume).last_hidden_state[:, 0, :]
        job_embeddings = model(**inputs_job).last_hidden_state[:, 0, :]

    cosine_sim = torch.nn.functional.cosine_similarity(resume_embeddings, job_embeddings)
    return cosine_sim.item()

# Calculate TF-IDF features
def calculate_tfidf_features(resumes, job_descriptions):
    all_text = resumes + list(job_descriptions.values())
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_text)
    resume_tfidf = tfidf_matrix[:len(resumes)]
    job_tfidf = tfidf_matrix[len(resumes):]
    return resume_tfidf, job_tfidf

# Prepare data for classification
def prepare_data_for_classification(resumes, labels, job_descriptions):
    X, y = [], []
    resume_tfidf, job_tfidf = calculate_tfidf_features(resumes, job_descriptions)

    for i, resume in enumerate(resumes):
        for j, job_desc in enumerate(job_descriptions.values()):
            # TF-IDF similarity
            resume_vec = resume_tfidf[i].toarray().flatten()
            job_vec = job_tfidf[j].toarray().flatten()
            tfidf_similarity = np.dot(resume_vec, job_vec) / (np.linalg.norm(resume_vec) * np.linalg.norm(job_vec) + 1e-6)

            # Combined score
            similarity = calculate_cosine_similarity(resume, job_desc)
            skill_overlap = calculate_skill_overlap(resume, job_desc)
            combined_score = (similarity + skill_overlap + tfidf_similarity) / 3

            X.append([similarity, skill_overlap, tfidf_similarity])
            y.append(labels[i])

    # Visualize similarity distribution
    similarities = [x[0] for x in X]
    plt.hist(similarities, bins=30, color='skyblue', alpha=0.7)
    plt.title('Distribution of Similarities')
    plt.xlabel('Combined Similarity Score')
    plt.ylabel('Frequency')
    plt.show()

    print(f"Class distribution in y: {Counter(y)}")
    return X, y

# Paths to folders
data_job_folder = "../Data_Job"
data_cv_folder = "../Data_CV"

# Debugging folder paths
print("Checking Data_Job folder...")
for root, dirs, files in os.walk(data_job_folder):
    print(f"Root: {root}, Files: {files}")

print("\nChecking Data_CV folder...")
for root, dirs, files in os.walk(data_cv_folder):
    print(f"Root: {root}, Files: {files}")

# Load data
job_descriptions = load_job_descriptions(data_job_folder)
resumes, labels = load_resumes(data_cv_folder)

print(f"Loaded {len(job_descriptions)} job descriptions and {len(resumes)} resumes.")
if len(resumes) == 0 or len(job_descriptions) == 0:
    raise ValueError("No resumes or job descriptions were loaded. Please check data folders.")

# Prepare data
X, y = prepare_data_for_classification(resumes, labels, job_descriptions)

# Handle class imbalance using SMOTE
if len(set(y)) < 2:
    raise ValueError("Insufficient class diversity in target variable 'y'. Adjust threshold or input data.")

print(f"Class distribution before SMOTE: {Counter(y)}")
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train XGBoost model
classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Prediction function
def predict_fit(resume, job_description):
    # Combine resume and job description for TF-IDF
    all_text = [resume] + [job_description]
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_text)

    # Extract TF-IDF vectors
    resume_vec = tfidf_matrix[0].toarray().flatten()
    job_vec = tfidf_matrix[1].toarray().flatten()

    # TF-IDF similarity
    tfidf_similarity = np.dot(resume_vec, job_vec) / (np.linalg.norm(resume_vec) * np.linalg.norm(job_vec) + 1e-6)

    # Cosine similarity and skill overlap
    similarity = calculate_cosine_similarity(resume, job_description)
    skill_overlap = calculate_skill_overlap(resume, job_description)

    # Combine scores
    combined_score = (similarity + skill_overlap + tfidf_similarity) / 3

    # Prediction
    prediction = classifier.predict([[similarity, skill_overlap, tfidf_similarity]])
    return "Fit" if prediction[0] == 1 else "Not Fit"


# Example usage
if resumes and job_descriptions:
    result = predict_fit(resumes[0], list(job_descriptions.values())[0])
    print(resumes[0])
    print(list(job_descriptions.values())[0])
    print(f"Does the candidate fit? {result}")
