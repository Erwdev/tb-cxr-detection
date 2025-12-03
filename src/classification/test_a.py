import pandas as pd
import numpy as np
import ast
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# This must match the seed used to generate the dataset split
SEED = 42

def test_model():
    
    # 1. Load Data
    csv_path = "data/processed/features/dataset.csv"
    if not os.path.exists(csv_path):
        print(" Error: dataset.csv not found.")
        return
    df = pd.read_csv(csv_path)

    # 2. Fix LBP Columns (Parse Strings -> Numbers)
    if 'lbp_hist' in df.columns and isinstance(df['lbp_hist'].iloc[0], str):
        print(" Parsing LBP features...")
        df['lbp_hist'] = df['lbp_hist'].apply(ast.literal_eval)
        lbp_df = pd.DataFrame(df['lbp_hist'].tolist(), index=df.index)
        lbp_df.columns = [f'lbp_{i}' for i in range(lbp_df.shape[1])]
        df = pd.concat([df, lbp_df], axis=1)
        df.drop(columns=['lbp_hist'], inplace=True)

    # 3. Prepare X and y
    drop_cols = ['label', 'filename']
    # Ensure columns are in the correct order (assuming standard extraction order)
    # The SVM is strict about feature order. We select strictly numeric columns.
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Save the feature names to check if they look right
    print(f"   Input Features ({X.shape[1]}): {list(X.columns)}")
    
    X = X.values
    y = df['label'].values
    
    # 4. Re-create the 80/20 Split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    model_path = "models/tb_model_raw.pkl"
    if not os.path.exists(model_path):
        print(f" Error: {model_path} not found.")
        return
        
    print(f"   Loading {model_path}...")
    try:
        model = joblib.load(model_path)
        # If it's a dictionary (like yours), extract model. If not, use as is.
        if isinstance(model, dict) and "model" in model:
            print("   (Detected Dictionary format)")
            model = model["model"]
            # If mask exists, we might need it, but pickle implies raw pipeline
            # We assume model expects ALL features (no Moth Search mask)
        else:
            print("   (Detected Raw Pipeline format)")
            
    except Exception as e:
        print(f" Failed to load model: {e}")
        return

    # 6. Predict
    print(" Predicting...")
    try:
        # We try predicting with X_test directly
        # Note: If used Feature Selection, this might fail with shape mismatch.
        y_pred = model.predict(X_test)
    except ValueError as e:
        print("\n SHAPE MISMATCH ERROR:")
        print(f"   The model expected a different number of features than we provided.")
        print(f"   Error details: {e}")
        print("   Possible reason: used Feature Selection (Moth) but didn't save the mask.")
        return
    
    # 7. Report
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*40)
    print(f"TEST RESULT (SVM)")
    print("="*40)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReport:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "TB"]))

if __name__ == "__main__":
    test_model()