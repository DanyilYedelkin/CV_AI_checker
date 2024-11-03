from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Пример текстовых данных резюме и описаний вакансий
resumes = [
    "Experienced data scientist skilled in Python, SQL, and machine learning.",
    "Software engineer with expertise in Java and project management.",
    "Financial analyst with experience in investment analysis and data visualization.",
    "Marketing specialist with strong background in social media and digital marketing."
]

job_descriptions = [
    "Looking for a data scientist with experience in Python, SQL, and machine learning.",
    "Seeking software engineer with Java skills and project management experience.",
    "Hiring financial analyst with expertise in investment and data visualization.",
    "Searching for a digital marketer with a focus on social media and digital marketing."
]

# Шаг 1: Преобразование текста с помощью TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
resumes_tfidf = vectorizer.fit_transform(resumes)
job_descriptions_tfidf = vectorizer.transform(job_descriptions)

# Шаг 2: Рассчёт косинусного сходства и создание меток
similarity_matrix = cosine_similarity(resumes_tfidf, job_descriptions_tfidf)

# Установка порога для подходящих и неподходящих пар
threshold = 0.3  # Порог можно настроить экспериментально

# Создаём данные для обучения модели
X = []
y = []

for i, resume_vector in enumerate(similarity_matrix):
    for j, similarity in enumerate(resume_vector):
        # Создаём текстовые пары резюме и описания вакансий
        X.append((resumes[i], job_descriptions[j]))
        # Присваиваем метку: 1, если сходство выше порога, и 0, если ниже
        y.append(1 if similarity >= threshold else 0)

# Преобразуем пары текста для модели
combined_text = [resume + " " + job for resume, job in X]
X_tfidf = vectorizer.fit_transform(combined_text)

# Шаг 3: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Шаг 4: Обучение модели логистической регрессии
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Шаг 5: Функция для предсказания соответствия резюме и описания вакансии
def predict_fit(resume, job_description):
    combined_text = resume + " " + job_description
    X_new = vectorizer.transform([combined_text])
    prediction = model.predict(X_new)
    prediction_prob = model.predict_proba(X_new)[:, 1]  # Вероятность класса "1"
    print("Prediction Probability for fit:", prediction_prob[0])  # Вывод вероятности
    return bool(prediction[0])  # True, если подходит, False — если не подходит

# Пример использования функции предсказания
new_resume = "Experienced data scientist with expertise in Python, SQL, and machine learning."
new_job_description = "Looking for a data scientist with experience in Python, SQL, and machine learning."
result = predict_fit(new_resume, new_job_description)
print("Does the candidate fit?", result)

# Пример неподходящего случая
new_resume_non_fit = "Graphic designer skilled in Adobe Photoshop and Illustrator."
new_job_description_non_fit = "Hiring a software engineer with Java and Python experience."
result_non_fit = predict_fit(new_resume_non_fit, new_job_description_non_fit)
print("Does the candidate fit?", result_non_fit)