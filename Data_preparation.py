import pandas as pd
import os

# Загрузите CSV-файл
file_path = "C:/Users/User/Downloads/training_data (1).csv"
data = pd.read_csv(file_path)

# print(data)
# print(data.columns)

# Укажите папку для сохранения файлов
output_dir = "D:/Job_dataset/Job_dataset"
os.makedirs(output_dir, exist_ok=True)  # создаст папку, если её нет

# Проверьте текущую рабочую директорию
print("Текущая рабочая директория:", os.getcwd())

# Проверка названий столбцов
if 'model_response' in data.columns and 'position_title' in data.columns and 'company_name' in data.columns:
    # Пройдитесь по каждой строке данных
    for index, row in data.iterrows():
        model_response = row['model_response']
        position_title = row['position_title']
        company_name = row['company_name']

        # Очистка имени файла от запрещенных символов
        safe_position_title = "".join(c for c in position_title if c.isalnum() or c in " _-")
        safe_company_name = "".join(c for c in company_name if c.isalnum() or c in " _-")

        # Сформируйте полный путь к файлу с указанием директории
        file_name = os.path.join(output_dir + "/", f"{safe_position_title}_{safe_company_name}.txt")
        print(f"Создаем файл: {file_name}")

        # Сохраните model_response в отдельный текстовый файл
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(model_response)
    print("Файлы успешно созданы в папке:", output_dir)
else:
    print("Убедитесь, что в файле присутствуют столбцы 'model_response', 'position_title' и 'company_name'.")
