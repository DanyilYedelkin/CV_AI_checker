# Original resume text
resume_text = """
ADMINISTRATIVE ASSISTANT
Summary
Experienced Administrative Assistant with extensive background providing support to college Dean of Student Success, Associate Vice President of Student Affairs, and Vice President of Student Affairs.

Technically proficient, with experience in a wide range of computer software and systems.
Provided support and counsel on projects requiring confidentiality, independent thinking, and competence.
Processed budgetary issues and employee records.
Triaged and managed student needs.
Assisted in college-wide committees and projects.
Core Qualifications
Microsoft Office Suite
Datatel
Power Campus
BlackBoard
Student Affairs Record System (SARS)
ImageNow
Siemens (Call Center Software)
Professional Experience
Administrative Assistant
Company Name – City, State
July 2013 – Current

Responsible for administrative support functions for division Dean.
Maintained schedule and coordinated plans for meetings (logistical, catering, technical).
Provided support services for department faculty.
Supervision/Management: Oversaw federal student worker; processed and approved payroll.
Budgetary Support: Managed budget for School of Education programs; tracked requisitions and purchase orders; maintained accounting records.
Administrative Assistant
Company Name – City, State
January 2005 – January 2011

Provided administrative support to the Dean of Student Success and Vice President of Student Affairs.
Coordinated files for disciplinary actions and judicial hearings.
Monitored student servicing levels and provided metrics.
Supervised front desk staff and student workers.
Maintained Student Success Center Operating Budget and federal grant records.
Technology Support: Acted as SARS Administrator and Super User for scheduling management system.
Administrative Assistant
Company Name – City, State
January 1999 – January 2004

Assisted daily functions of a small family-owned sign shop.
Designed basic signs using CASmate and CASwin software.
Converted paper files to business software.
Managed office supplies and customer service.
Administrative Assistant
Company Name – City, State
January 1998 – January 1999

Performed receptionist duties, including phone answering and scheduling appointments.
Organized office filing systems.
Processed insurance reimbursement forms.
Education and Training
Spring 2013: Bachelor of Science in Business Administration, Albright College.
January 2016: Master of Science.
Fall 2006: Associate of Liberal Studies, Montgomery County Community College.
Skills
Academic, accounting, administrative support, streamlining processes.
Microsoft Office Suite, Access, Microsoft Project, Publisher, Visio.
Scheduling, payroll, budget management, grants management.
Customer service, supervision, call center operations.
References
Rodney Altemose, EdD, Executive Director, Bucks County Community College.
Email: Rodney.Altemose@bucks.edu
Phone: 215-258-7700 Ext. 7750
Andrea M. Porter, M.L.A., Registrar, University of Pennsylvania.
Email: anporter@design.upenn.edu
Phone: 215-898-6210
"""

import re
import spacy
from keybert import KeyBERT
import json
from datetime import datetime

# Загрузка моделей
nlp = spacy.load("en_core_web_sm")  # Модель Spacy для NER
kw_model = KeyBERT()  # Модель для ключевых фраз


# Категории для распознавания
categories = {
    "Core Responsibilities": [],
    "Required Skills": [],
    "Educational Requirements": [],
    "Experience Level": [],
    "Preferred Qualifications": []
}

# Функция для подсчёта общего опыта
def calculate_total_experience(text):
    date_ranges = re.findall(r'(\w+\s\d{4})\s*–\s*(\w+\s\d{4}|Current)', text)
    total_months = 0

    for start, end in date_ranges:
        try:
            start_date = datetime.strptime(start, "%B %Y")
            end_date = datetime.now() if "Current" in end else datetime.strptime(end, "%B %Y")
            months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            total_months += months
        except ValueError:
            pass  # Игнорируем некорректные даты

    total_years = total_months // 12
    return total_years

# Функция для выделения сущностей
def extract_entities(text):
    doc = nlp(text)
    skills = set()
    dates = set()
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates.add(ent.text)
        elif ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]:  # Пример для навыков и инструментов
            skills.add(ent.text)
    return skills, dates

# Функция для ключевых фраз
def extract_keywords(text, num_keywords=10):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
    return [kw[0] for kw in keywords[:num_keywords]]

# Обработка текста и распределение по категориям
def process_resume(text):
    sentences = text.split("\n")
    skills, dates = extract_entities(text)
    keywords = extract_keywords(text)

    for sentence in sentences:
        if "responsible" in sentence.lower() or "managed" in sentence.lower():
            categories["Core Responsibilities"].append(sentence)
        elif any(skill in sentence for skill in skills):
            categories["Required Skills"].append(sentence)
        elif any(date in sentence for date in dates):
            categories["Educational Requirements"].append(sentence)
        elif "experience" in sentence.lower():
            categories["Preferred Qualifications"].append(sentence)

    # Добавление ключевых слов для категорий
    categories["Required Skills"].extend(keywords)

    # Подсчёт общего опыта и добавление в Experience Level
    total_experience = calculate_total_experience(text)
    categories["Experience Level"] = f"More than: {total_experience} years of experience."

    # Объединение результатов
    for key in categories:
        categories[key] = " ".join(categories[key]) if isinstance(categories[key], list) and categories[key] else categories[key]

    return categories

# Обработка резюме
result = process_resume(resume_text)

# Вывод результата
print(json.dumps(result, indent=2, ensure_ascii=False))
