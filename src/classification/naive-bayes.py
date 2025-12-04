import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

SEED = 42

# 1. Load Dataset
df = pd.read_csv("data/processed/features/dataset.csv")

# Fix LBP -> list
if 'lbp_hist' in df.columns and isinstance(df['lbp_hist'].iloc[0], str):
    df['lbp_hist'] = df['lbp_hist'].apply(ast.literal_eval)
    lbp_df = pd.DataFrame(df['lbp_hist'].tolist(), index=df.index)
    lbp_df.columns = [f"lbp_{i}" for i in range(lbp_df.shape[1])]
    df = pd.concat([df, lbp_df], axis=1)
    df.drop(columns=['lbp_hist'], inplace=True)

# 2. Prepare X and y
drop_cols = ["label", "filename"]
X = df.drop(columns=drop_cols, errors="ignore").values
y = df["label"].values

# 3. Split Dataset (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# 4. Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# 5. Predict
y_pred = nb.predict(X_test)

# 6. Evaluation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n========================")
print("   NAIVE BAYES RESULT   ")
print("========================")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {prec:.2f}")
print(f"Recall   : {rec:.2f}")
print(f"F1-Score : {f1:.2f}")

print("\nConfusion Matrix:")
print(cm)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=["Normal", "TB"],
            yticklabels=["Normal", "TB"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes")
plt.show()


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "TB"]))

y_train_pred = nb.predict(X_train)

acc_train  = accuracy_score(y_train, y_train_pred)
prec_train = precision_score(y_train, y_train_pred)
rec_train  = recall_score(y_train, y_train_pred)
f1_train   = f1_score(y_train, y_train_pred)
cm_train   = confusion_matrix(y_train, y_train_pred)

print("\n=== HASIL DATA LATIH (TRAIN) - NAIVE BAYES ===")
print(f"Accuracy : {acc_train*100:.2f}%")
print(f"Precision: {prec_train:.2f}")
print(f"Recall   : {rec_train:.2f}")
print(f"F1-Score : {f1_train:.2f}")

print("\nConfusion Matrix (Train):")
print(cm_train)

plt.figure(figsize=(4,4))
sns.heatmap(cm_train, annot=True, cmap="Blues", fmt="d",
            xticklabels=["Normal", "TB"],
            yticklabels=["Normal", "TB"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes (Train)")
plt.tight_layout()
plt.show()

print("\nClassification Report (Train):")
print(classification_report(y_train, y_train_pred, target_names=['Normal','TB']))
