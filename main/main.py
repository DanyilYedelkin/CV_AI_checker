from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Пример данных
resumes = [
    "5 years of experience in data analysis and machine learning",
    "Experienced in software engineering and project management",
    "Worked as a financial analyst for 3 years",
    "Marketing specialist with focus on social media and SEO"
]

job_descriptions = [
    "Looking for a data analyst with experience in machine learning",
    "Software engineer with strong project management skills required",
    "Seeking a financial analyst with over 2 years of experience",
    "Hiring a marketing expert in social media and SEO"
]

# Целевые метки (1 — подходит, 0 — не подходит)
labels = [1, 0, 1, 0]  # Modify this based on actual criteria of fitting

# 1. Преобразование текста в векторы TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform([resume + " " + job for resume, job in zip(resumes, job_descriptions)])

# 2. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 3. Обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Оценка модели
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 5. Функция для предсказания подхода кандидата
def predict_fit(resume, job_description):
    combined_text = resume + " " + job_description
    X_new = vectorizer.transform([combined_text])
    prediction = model.predict(X_new)
    return bool(prediction[0])  # True, если подходит, False — если не подходит

# Пример использования функции
new_resume = "5 years experience in data analysis and strong skills in machine learning"
new_job_description = "Data analyst needed with experience in machine learning"
result = predict_fit(new_resume, new_job_description)
print("Does the candidate fit?", result)