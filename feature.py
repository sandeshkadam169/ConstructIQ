import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# -----------------------------
# 1. Load cleaned dataset
# -----------------------------

df = pd.read_csv("clean_training_dataset.csv")

# Fix NaN values in text column
df["clean_text"] = df["clean_text"].fillna("").astype(str)


# -----------------------------
# 2. Keyword Dictionaries
# -----------------------------

keywords = {

    "architectural": [
        "floor", "ceiling", "door", "window",
        "corridor", "room", "wall", "stair"
    ],

    "structural": [
        "beam", "column", "foundation",
        "rebar", "slab", "concrete", "steel"
    ],

    "mechanical": [
        "duct", "hvac", "fan",
        "air", "pump", "mechanical",
        "chiller", "exhaust"
    ],

    "plumbing": [
        "pipe", "water", "drain",
        "toilet", "fixture", "valve",
        "sanitary"
    ],

    "electrical": [
        "electrical", "panel",
        "receptacle", "gfi",
        "conduit", "breaker",
        "circuit", "inverter",
        "photovoltaic"
    ],

    "fire_protection": [
        "sprinkler", "fire",
        "alarm", "detector",
        "tamper", "pump"
    ]
}


# -----------------------------
# 3. Keyword Feature Extraction
# -----------------------------

def extract_keyword_features(text):

    text = str(text)

    features = {}

    for category in keywords:

        count = 0

        for word in keywords[category]:
            count += text.count(word)

        features[category] = count

    return pd.Series(features)


keyword_features = df["clean_text"].apply(extract_keyword_features)


# -----------------------------
# 4. TF-IDF Feature Extraction
# -----------------------------

vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(df["clean_text"])
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

tfidf_features = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)


# -----------------------------
# 5. Combine Features
# -----------------------------

feature_dataset = pd.concat(
    [df["label"], keyword_features, tfidf_features],
    axis=1
)
joblib.dump(feature_dataset.columns.tolist(), "feature_columns.pkl")

# -----------------------------
# 6. Save Dataset
# -----------------------------

feature_dataset.to_csv("feature_dataset.csv", index=False)

print("Feature dataset created successfully")
print(feature_dataset.head())