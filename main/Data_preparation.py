import pandas as pd
import os

# Load the CSV file
file_path = "C:/Users/User/Downloads/training_data (1).csv"
data = pd.read_csv(file_path)

# print(data)
# print(data.columns)

# Specify the folder to save the files
output_dir = "D:/Job_dataset/Job_dataset"
os.makedirs(output_dir, exist_ok=True)  # Create the folder if it does not exist

# Check the current working directory
print("Current working directory:", os.getcwd())

# Check for required column names
if 'model_response' in data.columns and 'position_title' in data.columns and 'company_name' in data.columns:
    # Iterate through each row in the dataset
    for index, row in data.iterrows():
        model_response = row['model_response']
        position_title = row['position_title']
        company_name = row['company_name']

        # Clean the file name to remove forbidden characters
        safe_position_title = "".join(c for c in position_title if c.isalnum() or c in " _-")
        safe_company_name = "".join(c for c in company_name if c.isalnum() or c in " _-")

        # Build the full file path with the directory
        file_name = os.path.join(output_dir + "/", f"{safe_position_title}_{safe_company_name}.txt")
        print(f"Creating file: {file_name}")

        # Save the model_response to a separate text file
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(model_response)
    print("Files have been successfully created in the folder:", output_dir)
else:
    print("Ensure that the file contains the columns 'model_response', 'position_title', and 'company_name'.")
