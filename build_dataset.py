import os
import fitz
import pandas as pd

dataset_path = "dataset"

texts = []
labels = []

def extract_text(pdf_path):

    text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            text += page.get_text()

        text = text.replace("\n", " ")

    except:
        print("Error reading:", pdf_path)

    return text


for category in os.listdir(dataset_path):

    category_path = os.path.join(dataset_path, category)

    if os.path.isdir(category_path):

        print("Processing:", category)

        for file in os.listdir(category_path):

            if file.endswith(".pdf"):

                pdf_path = os.path.join(category_path, file)

                text = extract_text(pdf_path)

                texts.append(text)

                labels.append(category)


df = pd.DataFrame({
    "text": texts,
    "label": labels
})

df.to_csv("training_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.head())