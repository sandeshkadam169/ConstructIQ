import pandas as pd
import re

df = pd.read_csv("training_dataset.csv")


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # remove scale info
    text = re.sub(r'scale\s*[:=]?.*', ' ', text)

    # remove dimensions like 12'-6"
    text = re.sub(r'\d+\'\d*\"?', ' ', text)

    # remove decimal numbers
    text = re.sub(r'\b\d+\.\d+\b', ' ', text)

    # remove standalone numbers
    text = re.sub(r'\b\d+\b', ' ', text)

    # remove punctuation
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # remove single letters (grid labels)
    text = re.sub(r'\b[a-z]\b', ' ', text)

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


df["clean_text"] = df["text"].apply(clean_text)

df.to_csv("clean_training_dataset.csv", index=False)

print("Number filtering completed")