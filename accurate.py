import pandas as pd

# Load CSV
df = pd.read_csv("Final_Classification_Output.csv")

# Compare true label with predicted label
correct_predictions = (df["True Folder"] == df["Predicted Category"]).sum()

# Total samples
total_samples = len(df)

# Accuracy
accuracy = correct_predictions / total_samples

# Percentage accuracy
accuracy_percent = accuracy * 100

print("Total Samples:", total_samples)
print("Correct Predictions:", correct_predictions)
print("Accuracy:", accuracy)
print("Accuracy Percentage:", round(accuracy_percent, 2), "%")