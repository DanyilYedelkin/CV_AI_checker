from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.http import JsonResponse
import torch
from transformers import BertTokenizer, BertModel
import re

# Загрузка модели BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Функция для извлечения ключевых слов
def extract_keywords(text):
    keywords = re.findall(r"\b[A-Za-z-]+\b", text)
    return set([word.lower() for word in keywords if len(word) > 2])

# Расчет совпадения навыков
def calculate_skill_overlap(resume, job_description):
    resume_keywords = extract_keywords(resume)
    job_keywords = extract_keywords(job_description)
    common_keywords = resume_keywords.intersection(job_keywords)
    return len(common_keywords) / len(job_keywords) if len(job_keywords) > 0 else 0

# Расчет косинусного сходства через BERT
def calculate_cosine_similarity(resume, job_description):
    inputs_resume = tokenizer(resume, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_job = tokenizer(job_description, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        resume_embeddings = model(**inputs_resume).last_hidden_state[:, 0, :]
        job_embeddings = model(**inputs_job).last_hidden_state[:, 0, :]

    cosine_sim = torch.nn.functional.cosine_similarity(resume_embeddings, job_embeddings)
    return cosine_sim.item()

class PredictFitAPIView(APIView):
    def post(self, request):
        try:
            data = request.data
            resume = data.get('resume', '')
            job_description = data.get('job_description', '')

            similarity = calculate_cosine_similarity(resume, job_description)
            skill_overlap = calculate_skill_overlap(resume, job_description)
            combined_score = (similarity + skill_overlap) / 2

            prediction = "Fit" if combined_score >= 0.7 else "Not Fit"

            return Response({
                "similarity": similarity,
                "skill_overlap": skill_overlap,
                "combined_score": combined_score,
                "prediction": prediction,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"detail": f"Error processing request: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def root_view(request):
    return JsonResponse({"message": "Welcome to the Job Matching API!"})
