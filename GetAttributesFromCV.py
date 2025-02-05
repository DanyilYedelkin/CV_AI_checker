import pandas as pd
import spacy
from transformers import pipeline

# Загрузка данных
df = pd.read_csv('/mnt/data/training_data.csv')

# Инициализация NLP модели (например, для суммаризации или извлечения сущностей)
nlp_summarize = pipeline("summarization")
nlp_skills = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Категории для классификации
categories = ["Core Responsibilities", "Required Skills", "Educational Requirements", "Experience Level",
              "Preferred Qualifications", "Compensation and Benefits"]


def extract_information(resume_text):
    # Обработка текста резюме, чтобы извлечь ключевые моменты
    summary = nlp_summarize(resume_text, max_length=100, min_length=50, do_sample=False)[0]['summary_text']

    # Инициализация пустого словаря для хранения атрибутов
    attributes = {category: "" for category in categories}

    for category in categories:
        result = nlp_skills(summary, candidate_labels=categories)
        best_label = result['labels'][0]

        if best_label == category:
            attributes[category] = summary

    return attributes


# Применение функции к каждому резюме и добавление результатов в DataFrame
df['model_response'] = df['Resume_str'].apply(extract_information)

# Сохранение результата в новый CSV
df.to_csv('/mnt/data/resume_with_model_response.csv', index=False)
