import pandas as pd
import spacy
from transformers import pipeline

# Load the data
df = pd.read_csv('/mnt/data/training_data.csv')

# Initialize NLP models (e.g., for summarization or entity extraction)
nlp_summarize = pipeline("summarization")
nlp_skills = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Categories for classification
categories = ["Core Responsibilities", "Required Skills", "Educational Requirements", "Experience Level",
              "Preferred Qualifications", "Compensation and Benefits"]


def extract_information(resume_text):
    # Process resume text to extract key points
    summary = nlp_summarize(resume_text, max_length=100, min_length=50, do_sample=False)[0]['summary_text']

    # Initialize an empty dictionary to store extracted attributes
    attributes = {category: "" for category in categories}

    for category in categories:
        # Perform zero-shot classification to determine the best category
        result = nlp_skills(summary, candidate_labels=categories)
        best_label = result['labels'][0]

        if best_label == category:
            attributes[category] = summary

    return attributes


# Apply the function to each resume and add results to the DataFrame
df['model_response'] = df['Resume_str'].apply(extract_information)

# Save the result to a new CSV file
df.to_csv('/mnt/data/resume_with_model_response.csv', index=False)
