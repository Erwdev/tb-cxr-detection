import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

SEED = 42

df = pd.read_csv("data/processed/features/dataset.csv")

# Fix LBP -> list
if 'lbp_hist' in df.columns and isinstance(df['lbp_hist'].iloc[0], str):
    df['lbp_hist'] = df['lbp_hist'].apply(ast.literal_eval)
    lbp_df = pd.DataFrame(df['lbp_hist'].tolist(), index=df.index)
    lbp_df.columns = [f"lbp_{i}" for i in range(lbp_df.shape[1])]
    df = pd.concat([df, lbp_df], axis=1)
    df.drop(columns=['lbp_hist'], inplace=True)

drop_cols = ["label", "filename"]
X = df.drop(columns=drop_cols, errors="ignore").values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

dt = DecisionTreeClassifier(
    max_depth=4,  # moderate depth
    random_state=SEED
)

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n=============================")
print("  DECISION TREE RESULT")
print("=============================")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Precision: {prec:.2f}")
print(f"Recall:    {rec:.2f}")
print(f"F1-Score:  {f1:.2f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'TB']))

# ==== UJI DATA LATIH (TRAIN) ====
y_train_pred = dt.predict(X_train)

acc_train  = accuracy_score(y_train, y_train_pred)
prec_train = precision_score(y_train, y_train_pred)
rec_train  = recall_score(y_train, y_train_pred)
f1_train   = f1_score(y_train, y_train_pred)
cm_train   = confusion_matrix(y_train, y_train_pred)

print("\n=== HASIL DATA LATIH (TRAIN) ===")
print(f"Accuracy : {acc_train*100:.2f}%")
print(f"Precision: {prec_train:.2f}")
print(f"Recall   : {rec_train:.2f}")
print(f"F1-Score : {f1_train:.2f}")
print("\nConfusion Matrix (Train):")
print(cm_train)
print("\nClassification Report (Train):")
print(classification_report(y_train, y_train_pred, target_names=['Normal','TB']))

