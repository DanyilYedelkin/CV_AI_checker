import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Step 1: Load and preprocess data
def load_resumes(data_cv_folder):
    resumes, labels, filenames = [], [], []
    for root, dirs, files in os.walk(data_cv_folder):
        for file in files:
            if file.endswith(".txt"):
                label = 1 if "good" in root.lower() else 0
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read().strip()  # Read content of the file
                        if data:  # Ensure the file is not empty
                            resumes.append(data)
                            labels.append(label)
                            filenames.append(file)
                        else:
                            print(f"Warning: File {file_path} is empty.")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    return resumes, labels, filenames

def load_job_descriptions(data_job_folder):
    job_descriptions = {}
    job_filenames = {}
    for root, dirs, files in os.walk(data_job_folder):
        for file in files:
            if file.endswith(".txt"):
                job_id = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read().strip()  # Read content of the file
                        if data:  # Ensure the file is not empty
                            job_descriptions[job_id] = data
                            job_filenames[job_id] = file
                        else:
                            print(f"Warning: File {file_path} is empty.")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    return job_descriptions, job_filenames

def create_dataset(job_descriptions, job_filenames, resumes, labels, resume_filenames):
    data = []
    for i, resume in enumerate(resumes):
        for job_id, job_desc in job_descriptions.items():
            data.append((job_desc, resume, labels[i], job_filenames[job_id], resume_filenames[i]))
    return pd.DataFrame(data, columns=["job", "cv", "label", "job_file", "cv_file"])

# Define custom collate function to handle variable-length sequences
def collate_fn(batch):
    inputs = {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [item[0]["input_ids"].squeeze(0) for item in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [item[0]["attention_mask"].squeeze(0) for item in batch],
            batch_first=True,
            padding_value=0
        ),
        "token_type_ids": torch.nn.utils.rnn.pad_sequence(
            [item[0]["token_type_ids"].squeeze(0) for item in batch],
            batch_first=True,
            padding_value=0
        ),
    }
    labels = torch.stack([item[1] for item in batch])
    filenames = [(item[2], item[3]) for item in batch]
    return inputs, labels, filenames

# Paths to folders
data_job_folder = "../Data_Job"
data_cv_folder = "../Data_CV"

# Load data
job_descriptions, job_filenames = load_job_descriptions(data_job_folder)
resumes, labels, resume_filenames = load_resumes(data_cv_folder)
data = create_dataset(job_descriptions, job_filenames, resumes, labels, resume_filenames)

# Step 2: Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Step 3: Tokenizer and Model from HuggingFace
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Step 4: Create Dataset Class
class MatchDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        job_text = self.data.iloc[idx]["job"]
        cv_text = self.data.iloc[idx]["cv"]
        label = self.data.iloc[idx]["label"]
        job_file = self.data.iloc[idx]["job_file"]
        cv_file = self.data.iloc[idx]["cv_file"]
        inputs = tokenizer(job_text, cv_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        label = torch.tensor(label, dtype=torch.float)
        return inputs, label, job_file, cv_file

# Step 5: Data Loaders
train_dataset = MatchDataset(train_data)
test_dataset = MatchDataset(test_data)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Step 6: Define Model
class JobCVMatchModel(nn.Module):
    def __init__(self, bert_model):
        super(JobCVMatchModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

model = JobCVMatchModel(bert_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 7: Define Optimizer and Loss Function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.BCEWithLogitsLoss()

# Step 8: Training Loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels, _ = batch
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids).squeeze(-1)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Step 9: Evaluation
model.eval()
total_correct = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, labels, filenames = batch
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask, token_type_ids).squeeze(-1)
        predictions = torch.round(torch.sigmoid(outputs))
        total_correct += (predictions == labels).sum().item()

accuracy = total_correct / len(test_dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 10: Test with examples from the dataset
print("\nTesting model with examples:\n")
model.eval()
with torch.no_grad():
    for i in range(5):  # Test with first 5 samples
        job_text = test_data.iloc[i]["job"]
        cv_text = test_data.iloc[i]["cv"]
        job_file = test_data.iloc[i]["job_file"]
        cv_file = test_data.iloc[i]["cv_file"]
        inputs = tokenizer(job_text, cv_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        outputs = model(
            input_ids=inputs["input_ids"].squeeze(1),
            attention_mask=inputs["attention_mask"].squeeze(1),
            token_type_ids=inputs["token_type_ids"].squeeze(1)
        ).squeeze(-1)
        prediction = torch.sigmoid(outputs).item()

        print(f"Job File: {job_file}")
        print(f"CV File: {cv_file}")
        print(f"Job: {job_text[:100]}...")
        print(f"CV: {cv_text[:100]}...")
        print(f"Predicted Match: {'Good' if prediction > 0.5 else 'Bad'}\n")
