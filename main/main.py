from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Путь к папке с данными (относительный путь)
data_job_folder = os.path.join(os.getcwd(), "..", "data_job")
# Путь к папке с данными для СВ
data_cv_folder = os.path.join(os.getcwd(), "..", "data_cv")

# Функция для загрузки данных о вакансиях из файлов в папке
def load_job_descriptions(data_job_folder):
    job_descriptions = []

    # Перебираем все файлы в папке data
    for filename in os.listdir(data_job_folder):
        # Убедимся, что это .txt файл
        if filename.endswith(".txt"):
            file_path = os.path.join(data_job_folder, filename)

            # Открываем файл и читаем его содержимое
            with open(file_path, 'r', encoding='utf-8') as file:
                job_text = file.read().strip()

                # Разделим текст на различные разделы (например, Core Responsibilities, Required Skills и т.д.)
                job_data = {}
                sections = ["Core Responsibilities", "Required Skills", "Educational Requirements",
                            "Experience Level", "Preferred Qualifications"]

                # Парсим каждый раздел в вакансии
                for section in sections:
                    if section in job_text:
                        start = job_text.find(section)
                        end = job_text.find("\n", start)  # Находим конец раздела
                        job_data[section] = job_text[start:end].strip()

                # Собираем весь текст вакансии в одну строку
                full_job_description = ' '.join(job_data.values())

                # Добавляем в список вакансий
                job_descriptions.append(full_job_description)

    return job_descriptions

# Пример текстовых данных резюме
resumes = load_job_descriptions(data_cv_folder)

# Загружаем вакансии из папки
job_descriptions = load_job_descriptions(data_job_folder)

# Шаг 1: Преобразование текста с помощью TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')

# Обучение TF-IDF на всех текстах резюме и вакансий
vectorizer.fit(resumes + job_descriptions)

# Преобразование резюме и вакансий в TF-IDF
resumes_tfidf = vectorizer.transform(resumes)
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
X_tfidf = vectorizer.fit_transform(combined_text)  # Повторно обучаем на текстовых парах

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
new_resume = resumes[0]
new_job_description = job_descriptions[0]
result = predict_fit(new_resume, new_job_description)
print("Does the candidate fit?", result)

# Пример неподходящего случая
new_resume_non_fit = resumes[0]
new_job_description_non_fit = job_descriptions[1]
result_non_fit = predict_fit(new_resume_non_fit, new_job_description_non_fit)
print("Does the candidate fit?", result_non_fit)