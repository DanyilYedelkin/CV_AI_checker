from flask import Flask, request, jsonify
import re
import spacy
from keybert import KeyBERT
import json
from datetime import datetime
import PyPDF2
import os

# Initialize Flask app
app = Flask(__name__)

# Load models
nlp = spacy.load("en_core_web_sm")  # Spacy model for NER
kw_model = KeyBERT()  # Model for extracting key phrases

# Categories for recognition
def get_empty_categories():
    return {
        "Core Responsibilities": [],
        "Required Skills": [],
        "Educational Requirements": [],
        "Experience Level": [],
        "Preferred Qualifications": []
    }

# Function to read text from a PDF file
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() or ""  # Ensure no None values
    return text

# Function to calculate total work experience
def calculate_total_experience(text):
    date_ranges = re.findall(r'(\w+\s\d{4})\s*–\s*(\w+\s\d{4}|Current)', text)
    total_months = 0

    for start, end in date_ranges:
        try:
            start_date = datetime.strptime(start, "%B %Y")
            end_date = datetime.now() if "Current" in end else datetime.strptime(end, "%B %Y")
            months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            total_months += months
        except ValueError:
            pass  # Ignore invalid dates

    total_years = total_months // 12
    return total_years

# Function to extract entities
def extract_entities(text):
    doc = nlp(text)
    skills = set()
    dates = set()
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates.add(ent.text)
        elif ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]:
            skills.add(ent.text)
    return skills, dates

# Function to extract keywords
def extract_keywords(text, num_keywords=10):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
    return [kw[0] for kw in keywords[:num_keywords]]

# Process resume text and classify into categories
def process_resume(text):
    categories = get_empty_categories()
    sentences = text.split("\n")
    skills, dates = extract_entities(text)
    keywords = extract_keywords(text)

    for sentence in sentences:
        if "responsible" in sentence.lower() or "managed" in sentence.lower():
            categories["Core Responsibilities"].append(sentence)
        elif any(skill in sentence for skill in skills):
            categories["Required Skills"].append(sentence)
        elif any(date in sentence for date in dates):
            categories["Educational Requirements"].append(sentence)
        elif "experience" in sentence.lower():
            categories["Preferred Qualifications"].append(sentence)

    # Add extracted keywords to Required Skills
    categories["Required Skills"].extend(keywords)

    # Calculate total experience and add to Experience Level
    total_experience = calculate_total_experience(text)
    categories["Experience Level"] = f"More than: {total_experience} years of experience."

    # Combine results
    for key in categories:
        categories[key] = " ".join(categories[key]) if isinstance(categories[key], list) and categories[key] else categories[key]

    return categories

# Flask route to handle file upload and parsing
@app.route('/parse', methods=['POST'])
def parse_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Check if the file is a PDF
    if file and file.filename.endswith('.pdf'):
        resume_text = extract_text_from_pdf(file)
        result = process_resume(resume_text)
        return jsonify(result), 200
    else:
        return jsonify({"error": "File format not supported. Please upload a PDF file."}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
