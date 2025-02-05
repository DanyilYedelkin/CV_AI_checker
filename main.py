import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import os
import matplotlib.pyplot as plt
import re

# Loading the model and the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Functions for loading data
def load_job_descriptions(data_job_folder):
    job_descriptions = []
    for root, dirs, files in os.walk(data_job_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        job_text = f.read().strip()
                        if job_text:  # Ensure file is not empty
                            job_descriptions.append(job_text)
                        else:
                            print(f"Warning: Job description file {file_path} is empty.")
                except Exception as e:
                    print(f"Error reading job description file {file_path}: {e}")
    return job_descriptions

def load_resumes(data_cv_folder):
    resumes = []
    for root, dirs, files in os.walk(data_cv_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        resume_text = f.read().strip()
                        if resume_text:  # Ensure file is not empty
                            resumes.append(resume_text)
                        else:
                            print(f"Warning: Resume file {file_path} is empty.")
                except Exception as e:
                    print(f"Error reading resume file {file_path}: {e}")
    return resumes

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
            skill_overlap = calculate_skill_overlap(resume, job_description)
            combined_score = (similarity + skill_overlap) / 2

            similarities.append(combined_score)
            X.append([combined_score])
            y.append(1 if combined_score >= 0.7 else 0)  # Stricter threshold

    # Visualization of the distribution of similarities
    plt.hist(similarities, bins=30, color='skyblue', alpha=0.7)
    plt.title('Distribution of Similarities')
    plt.xlabel('Combined Similarity Score')
    plt.ylabel('Frequency')
    plt.show()


    # Class diagnostics
    print(f"Class distribution in y: {Counter(y)}")
    return X, y

# Paths to folders
data_job_folder = "..\\Data_Job"
data_cv_folder = "..\\Data_CV"

print(f"Resumes path: {data_cv_folder}")
print(f"Job descriptions path: {data_job_folder}")

resumes = load_resumes(data_cv_folder)
job_descriptions = load_job_descriptions(data_job_folder)

print(f"Number of resumes loaded: {len(resumes)}")
print(f"Number of job descriptions loaded: {len(job_descriptions)}")

# Print some samples to verify
if resumes:
    print(f"Sample resume: {resumes[10]}\n")
else:
    print("No resumes found.")

if job_descriptions:
    print(f"Sample job description: {job_descriptions[0]}\n")
else:
    print("No job descriptions found.")

if len(resumes) == 0 or len(job_descriptions) == 0:
    raise ValueError("No resumes or job descriptions were loaded. Please check data folders.")

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
        skill_overlap = calculate_skill_overlap(resume, job_description)
        combined_score = (similarity + skill_overlap) / 2
        prediction = classifier.predict([[combined_score]])
        return "Fit" if prediction[0] == 1 else "Not Fit"


    # suitable_resume
    suitable_resume = """
    {
        "Core Responsibilities": "Developed and deployed high-quality full-stack web applications to meet business objectives with a focus on usability. Collaborated with cross-functional teams through code reviews and feature implementations. Actively monitored production environments and implemented improvements based on customer feedback.",
        "Required Skills": "Bachelor's degree in Computer Science. Proficiency in Ruby on Rails, Java, and JavaScript/React. Strong experience with React Native and Elasticsearch.",
        "Educational Requirements": "Bachelor's degree in Computer Science.",
        "Experience Level": "3+ years in full-stack web application development.",
        "Preferred Qualifications": "Experience with iOS, Android, Google Maps APIs, and multichannel application design."
    }
    """

    # unsuitable_resume
    unsuitable_resume = """
    {
        "Core Responsibilities": "Managed retail operations, including staff supervision, inventory control, and financial reporting. Developed customer engagement strategies to improve store performance and ensure compliance with company policies.",
        "Required Skills": "Strong customer service and sales skills. Experience in inventory management, financial planning, and team leadership.",
        "Educational Requirements": "Bachelor's degree in Business Administration.",
        "Experience Level": "5+ years in retail management.",
        "Preferred Qualifications": "Experience with Point of Sale systems and retail marketing campaigns."
    }
    """

    # job_description
    job_description = """
    {
      "Core Responsibilities": "Builds high quality features that meet business objectives with a focus on usability. Collaborates closely with teammates through code reviews. Deploys projects to production frequently and monitors results. Actively solicits feedback from teammates and customers to improve usability.",
      "Required Skills": "Bachelor's degree or equivalent experience. Experience building and supporting web applications in a fullstack capacity, preferably 2+ years. Proficiency with Ruby on Rails, Java, .NET, Python, PHP or Groovy. Javascript/React preferred.", 
      "Educational Requirements": "Bachelor's degree or equivalent experience required",
      "Experience Level": "2+ years building and supporting web applications in a fullstack capacity",
      "Preferred Qualifications": "Experience with React Native, iOS, Android, Elasticsearch, Google Maps APIs"
    }
    """

    # Check suitable or unsuitable
    print("Checking suitable resume:")
    result_suitable = predict_fit(suitable_resume, job_description)
    print(f"Result: {result_suitable}\n")

    print("Checking unsuitable resume:")
    result_unsuitable = predict_fit(unsuitable_resume, job_description)
    print(f"Result: {result_unsuitable}\n")