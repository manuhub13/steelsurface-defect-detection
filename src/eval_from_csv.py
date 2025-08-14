import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("outputs/predictions.csv")
y_true = df["TrueLabel"]
y_pred = df["PredictedLabel"]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification report:\n", classification_report(y_true, y_pred))

# Optional: confusion matrix as CSV
cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
pd.DataFrame(cm, index=sorted(y_true.unique()), columns=sorted(y_true.unique())) \
  .to_csv("outputs/confusion_matrix.csv", index=True)
print("\nSaved confusion matrix to outputs/confusion_matrix.csv")
