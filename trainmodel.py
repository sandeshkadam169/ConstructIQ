import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -----------------------------
# 1. Load feature dataset
# -----------------------------

df = pd.read_csv("feature_dataset.csv")

# -----------------------------
# 2. Separate features and label
# -----------------------------

cols_to_drop = ['label', 'text', 'clean_text']

# 2. Create your feature matrix X by dropping those columns
# errors='ignore' ensures it doesn't crash if one of these is already missing
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 3. Define your target variable y
y = df['label']

# -----------------------------
# 3. Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 4. Train Random Forest Model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate Model
# -----------------------------

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6. Save Model
# -----------------------------

joblib.dump(model, "document_classifier.pkl")

print("\nModel saved as document_classifier.pkl")