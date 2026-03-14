import fitz  # PyMuPDF
import joblib
import pandas as pd
import re
import os

# --- 1. CONFIGURATION ---
BASE_DATASET_DIR = "dataset"  # The folder containing the 6 categories
OUTPUT_CSV = "Final_Classification_Output.csv"

# Load Models
print("Loading classification models...")
model = joblib.load("document_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
expected_features = model.feature_names_in_

# --- 2. HELPERS ---

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def keyword_features(text):
    keywords = {
        "architectural": ["floor","door","window","wall","room","corridor","stair"],
        "structural": ["beam","column","foundation","rebar","slab","concrete"],
        "mechanical": ["duct","hvac","fan","air","pump","chiller"],
        "plumbing": ["pipe","water","drain","toilet","valve","fixture"],
        "electrical": ["electrical","panel","circuit","lighting","switch"],
        "fire_protection": ["sprinkler","fire","alarm","detector","smoke"]
    }
    features = {cat: 0 for cat in keywords}
    for category, words in keywords.items():
        for word in words:
            features[category] += text.count(word)
    return pd.DataFrame([features])

def handle_duplicate_columns(df):
    """Aligns duplicate names to the '.1' format the model expects."""
    cols = pd.Series(df.columns)
    for dupe in cols[cols.duplicated()].unique():
        cols[cols == dupe] = [f"{dupe}.{i}" if i != 0 else dupe for i in range(sum(cols == dupe))]
    df.columns = cols
    return df

def get_sensitive_data_regex(text):
    """Regex-only PII detection (No spaCy required)."""
    found_items = set()
    # Phone, Email, Zip
    found_items.update(re.findall(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text))
    found_items.update(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    found_items.update(re.findall(r'\b\d{5}(?:-\d{4})?\b', text))
    # Firm/Company Suffixes
    company_pattern = r'\b[A-Z][A-Z\s&]+(?:P\.C\.|LLC|INC|LTD|GROUP|ASSOCIATES|CONSULTANTS|ENGINEERING|ENGINEERS)\b'
    for match in re.finditer(company_pattern, text):
        found_items.add(match.group().strip())
    return {item for item in found_items if len(item) > 3}

# --- 3. MAIN DIRECTORY WALK ---

results = []

print(f"Starting traversal of: {BASE_DATASET_DIR}")

# Walk through all subfolders
for root, dirs, files in os.walk(BASE_DATASET_DIR):
    for filename in files:
        # Process only PDFs and skip files already redacted
        if filename.lower().endswith(".pdf") and "_redacted" not in filename.lower():
            pdf_path = os.path.join(root, filename)
            # Folder name serves as the 'True Category' if needed for your own check
            true_folder = os.path.basename(root) 
            
            print(f"Processing [{true_folder}] -> {filename}")
            
            try:
                doc = fitz.open(pdf_path)
                raw_text = ""
                for page in doc:
                    raw_text += page.get_text()
                
                # --- CLASSIFICATION ---
                clean = clean_text(raw_text)
                kw_df = keyword_features(clean)
                tfidf_matrix = vectorizer.transform([clean])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
                
                X = pd.concat([kw_df, tfidf_df], axis=1)
                X = handle_duplicate_columns(X)
                X = X.reindex(columns=expected_features, fill_value=0)
                
                prediction = model.predict(X)[0]
                confidence = model.predict_proba(X).max()
                
                # --- REDACTION ---
                sensitive_items = get_sensitive_data_regex(raw_text)
                for page in doc:
                    for item in sensitive_items:
                        areas = page.search_for(item)
                        for rect in areas:
                            page.add_redaction_annot(rect, fill=(0, 0, 0))
                    page.apply_redactions()
                
                # Save output in the SAME folder as the original
                redacted_name = filename.replace(".pdf", "_Redacted.pdf")
                doc.save(os.path.join(root, redacted_name), garbage=4, deflate=True)
                doc.close()
                
                results.append({
                    "True Folder": true_folder,
                  
                    "file Name": filename,
                   
                    "Predicted Category": prediction,
                  
                    "Confidence Score": round(confidence, 4)
                })

            except Exception as e:
                print(f"   [Error] Could not process {filename}: {e}")

# --- 4. EXPORT RESULTS ---
if results:
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "="*40)
    print(f"SUCCESS: {len(results)} files processed.")
    print(f"CSV saved as: {OUTPUT_CSV}")
    print("="*40)
else:
    print("No PDF files found to process.")