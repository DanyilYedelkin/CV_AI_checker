from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
import re
import os

# Инициализация FastAPI
app = FastAPI()

# Загрузка модели BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Классификатор (заглушка, замените обучением модели, если требуется)
classifier = LogisticRegression(class_weight="balanced", max_iter=1000)

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

# Pydantic модель для входных данных
class JobMatchingRequest(BaseModel):
    resume: str
    job_description: str

# Эндпоинт для предсказания
@app.post("/predict/")
def predict_fit(request: JobMatchingRequest):
    try:
        similarity = calculate_cosine_similarity(request.resume, request.job_description)
        skill_overlap = calculate_skill_overlap(request.resume, request.job_description)
        combined_score = (similarity + skill_overlap) / 2

        # Пример порогового значения
        prediction = "Fit" if combined_score >= 0.7 else "Not Fit"
        return {
            "similarity": similarity,
            "skill_overlap": skill_overlap,
            "combined_score": combined_score,
            "prediction": prediction,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Главный эндпоинт
@app.get("/")
def read_root():
    return {"message": "Welcome to the Job Matching API!"}

import uvicorn

if __name__ == "__main__":
    uvicorn.run("endpointsFastAPI:app", host="127.0.0.1", port=8000, reload=True)