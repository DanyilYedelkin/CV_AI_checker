import torch
import pickle
from transformers import BertTokenizer, BertModel
import re
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from ..user.models import Resume
from ..offer.models import Offer
import os
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# Global variables for lazy initialization
_classifier = None
_tokenizer = None
_bert_model = None

# Function for lazy initialization of models and classifier
def get_models():
    global _classifier, _tokenizer, _bert_model

    if _classifier is None:
        model_path = os.path.abspath(os.path.join(settings.BASE_DIR, 'trained_model.pkl'))
        with open(model_path, "rb") as model_file:
            _classifier = pickle.load(model_file)

    if _tokenizer is None or _bert_model is None:
        _tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        _bert_model = BertModel.from_pretrained('bert-base-uncased')

    return _classifier, _tokenizer, _bert_model

# Function to extract keywords from text
def extract_keywords(text):
    keywords = re.findall(r'\b[A-Za-z-]+\b', text)
    return set(word.lower() for word in keywords if len(word) > 2)

# Function to calculate cosine similarity
def calculate_cosine_similarity(resume, job_description):
    classifier, tokenizer, bert_model = get_models()

    # Tokenize the texts
    inputs_resume = tokenizer(resume, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_job = tokenizer(job_description, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get embeddings
    with torch.no_grad():
        resume_embeddings = bert_model(**inputs_resume).last_hidden_state[:, 0, :]
        job_embeddings = bert_model(**inputs_job).last_hidden_state[:, 0, :]

    # Calculate cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(resume_embeddings, job_embeddings)
    return cosine_sim.item()

# Main function to evaluate resumes against job descriptions
def evaluate_candidate_view(resume_id):
    """
    Evaluates a resume against all job descriptions and adds matching offers to matching_offers.
    """
    try:
        # Get models
        classifier, tokenizer, bert_model = get_models()

        # Retrieve resume from the database
        resume = get_object_or_404(Resume, id=resume_id)
        offers = Offer.objects.all()  # Fetch all job descriptions

        # Extract resume text
        resume_text = " ".join([
            resume.core_responsibilities or "",
            resume.required_skills or "",
            resume.educational_requirements or "",
            resume.experience_level or "",
            resume.preferred_qualifications or ""
        ])

        matching_offer_ids = []  # List of matching offers

        for offer in offers:
            job_description = offer.description or ""

            # Calculate similarity
            similarity = calculate_cosine_similarity(resume_text, job_description)
            skill_overlap = len(extract_keywords(resume_text) & extract_keywords(job_description)) / len(extract_keywords(job_description)) if extract_keywords(job_description) else 0
            combined_score = (similarity + skill_overlap) / 2

            # Predict match
            prediction = classifier.predict([[combined_score]])

            # If the offer matches, add its ID
            if prediction[0] == 1:
                matching_offer_ids.append(offer.id)

        # Add matching offers to the resume
        if matching_offer_ids:
            resume.matching_offers.add(*matching_offer_ids)

        return {"matched_offers": matching_offer_ids}

    except Exception as e:
        # Log errors
        logger.error("Error during evaluation: %s", str(e))
        return {"error": str(e)}
