from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Пример текстовых данных резюме и описаний вакансий
resumes = [
    "Experienced data scientist skilled in Python, SQL, and machine learning.",
    "Software engineer with expertise in Java, C++, and project management.",
    "Financial analyst with experience in investment analysis, financial modeling, and data visualization.",
    "Marketing specialist with strong background in social media, SEO, and digital marketing.",
    "Machine learning engineer with a focus on neural networks, deep learning, and computer vision.",
    "Business analyst proficient in data analysis, Tableau, and SQL with experience in reporting.",
    "HR manager with expertise in recruitment, employee training, and performance management.",
    "Project manager with 5+ years of experience in Agile, Scrum, and cross-functional team management.",
    "Cybersecurity specialist experienced in network security, intrusion detection, and risk management.",
    "Frontend developer skilled in HTML, CSS, JavaScript, and React with a strong UX/UI focus.",
    "Backend developer with knowledge of Node.js, Python, databases, and microservices architecture.",
    "Customer support representative with experience in CRM, customer satisfaction analysis, and resolution management.",
    "Operations manager with skills in logistics, supply chain management, and process optimization.",
    "Data analyst with strong skills in SQL, R, and Python, specializing in data cleaning and visualization.",
    "Healthcare data specialist with knowledge of medical terminology, EMR systems, and patient data privacy.",
]

job_descriptions = [
    "Looking for a data scientist with experience in Python, SQL, and machine learning.",
    "Seeking software engineer with Java, C++ skills and project management experience.",
    "Hiring financial analyst with expertise in financial modeling and data visualization.",
    "Searching for a digital marketer with experience in social media, SEO, and online advertising.",
    "Machine learning engineer needed with experience in deep learning and computer vision.",
    "Business analyst required with strong SQL and Tableau skills for data-driven insights.",
    "We are hiring an HR manager with a background in recruitment, onboarding, and team engagement.",
    "Project manager with Agile and Scrum experience needed to lead cross-functional teams.",
    "Seeking a cybersecurity expert with experience in network security and risk assessment.",
    "Frontend developer needed with skills in JavaScript, React, and UX/UI best practices.",
    "Backend developer position open for candidates with Node.js, Python, and database management skills.",
    "Customer support representative needed to handle CRM, issue resolution, and customer feedback.",
    "Looking for an operations manager experienced in logistics and supply chain management.",
    "Data analyst required with expertise in SQL, R, and data visualization techniques.",
    "Hiring healthcare data specialist knowledgeable in EMR systems and patient data compliance.",
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