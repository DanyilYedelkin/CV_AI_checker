import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# Загрузка BERT-модели и токенизатора
bert_model = TFBertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Функция для подготовки текста
def preprocess_text(texts, max_len=128):
    """
    Преобразует текст в входные данные для модели BERT.
    :param texts: Список текстов
    :param max_len: Максимальная длина текста
    :return: input_ids, attention_mask
    """
    encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="tf")
    return encodings['input_ids'], encodings['attention_mask']


# Построение нейронной сети
def build_model(bert_model):
    """
    Создаёт модель для оценки соответствия вакансий и резюме.
    :param bert_model: Предварительно обученная BERT-модель
    :return: Компилированная модель
    """
    # Входы для вакансии
    job_input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="job_input_ids")
    job_attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="job_attention_mask")

    # Входы для резюме
    resume_input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="resume_input_ids")
    resume_attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="resume_attention_mask")

    # Эмбеддинги для вакансии
    job_embeddings = bert_model(job_input_ids, attention_mask=job_attention_mask)[1]

    # Эмбеддинги для резюме
    resume_embeddings = bert_model(resume_input_ids, attention_mask=resume_attention_mask)[1]

    # Объединение
    merged = tf.keras.layers.Concatenate()([job_embeddings, resume_embeddings])

    # Полносвязные слои
    dense = tf.keras.layers.Dense(128, activation="relu")(merged)
    dense = tf.keras.layers.Dropout(0.3)(dense)
    dense = tf.keras.layers.Dense(64, activation="relu")(dense)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    # Модель
    model = tf.keras.models.Model(inputs=[job_input_ids, job_attention_mask,
                                          resume_input_ids, resume_attention_mask], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# Построение модели
model = build_model(bert_model)
model.summary()

# Пример данных
job_texts = [
    "Builds high quality features that meet business objectives with a focus on usability. Collaborates closely with teammates through code reviews."
]
resume_texts = [
    "Developed high-quality features for web applications, ensuring usability and alignment with business objectives."
]

# Подготовка данных
job_input_ids, job_attention_mask = preprocess_text(job_texts)
resume_input_ids, resume_attention_mask = preprocess_text(resume_texts)

# Пример меток (0 — не подходит, 1 — подходит)
labels = [1]

# Обучение модели (тестовые данные, используется только для примера)
history = model.fit(
    [job_input_ids, job_attention_mask, resume_input_ids, resume_attention_mask],
    labels,
    batch_size=1,
    epochs=1
)

# Прогноз
prediction = model.predict([job_input_ids, job_attention_mask, resume_input_ids, resume_attention_mask])
print(f"Matching probability: {prediction[0][0]:.2f}")
