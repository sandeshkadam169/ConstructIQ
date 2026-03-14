import fitz
import joblib
import pandas as pd
import re
import os

print("Starting PDF classification...")

# 1. Load trained objects
model = joblib.load("document_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# Extract text
# -----------------------------
def extract_text(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------
# Keyword dictionary
# -----------------------------
keywords = {
    "architectural": ["floor","door","window","wall","room","corridor","stair"],
    "structural": ["beam","column","foundation","rebar","slab","concrete"],
    "mechanical": ["duct","hvac","fan","air","pump","chiller"],
    "plumbing": ["pipe","water","drain","toilet","valve","fixture"],
    "electrical": ["electrical","panel","circuit","lighting","switch"],
    "fire_protection": ["sprinkler","fire","alarm","detector","smoke"]
}

# -----------------------------
# Keyword features
# -----------------------------
def keyword_features(text):
    features = {}
    for category in keywords:
        count = 0
        for word in keywords[category]:
            count += text.count(word)
        features[category] = count
    return pd.DataFrame([features])

# -----------------------------
# Helper to handle duplicate column names (Crucial Fix)
# -----------------------------
def handle_duplicate_columns(df):
    cols = pd.Series(df.columns)
    for dupe in cols[cols.duplicated()].unique():
        # This renames duplicates to: name, name.1, name.2...
        cols[cols == dupe] = [f"{dupe}.{i}" if i != 0 else dupe for i in range(sum(cols == dupe))]
    df.columns = cols
    return df

# -----------------------------
# Test PDF
# -----------------------------
pdf_path = "/Users/sandeshkadam/Desktop/vconstruct/Data to be Classified and Redacted/30R5 Toilet Details 02.04.21-PLUMBING DETAILS.pdf"

if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
else:
    print("Reading:", pdf_path)
    raw_text = extract_text(pdf_path)
    clean = clean_text(raw_text)

    # 1. Generate features
    keyword_df = keyword_features(clean)
    tfidf_matrix = vectorizer.transform([clean])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # 2. Combine features
    X = pd.concat([keyword_df, tfidf_df], axis=1)

    # 3. FIX: Make column names unique BEFORE reindexing
    # This transforms duplicate 'architectural' into ['architectural', 'architectural.1']
    X = handle_duplicate_columns(X)

    # 4. Final Alignment
    # Use the model's internal feature list to ensure perfect order
    expected_features = model.feature_names_in_
    X = X.reindex(columns=expected_features, fill_value=0)

    # 5. Prediction
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X).max()

    print("\n" + "="*30)
    print("Predicted Category:", prediction)
    print("Confidence:", round(confidence, 3))
    print("="*30)