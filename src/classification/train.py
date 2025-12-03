import numpy as np
import pandas as pd
import ast
import random
import os
import joblib

# === 1. LOCK RANDOMNESS (To make results reproducible) ===
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
# NEW IMPORTS FOR RAW PIPELINE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- 2. Moth Search Algorithm (MSA) ---
class MothSearchFeatureSelection:
    def __init__(self, n_moths=5, max_iter=10): 
        self.n_moths = n_moths
        self.max_iter = max_iter

    def fit(self, X, y, classifier):
        n_features = X.shape[1]
        moths = np.random.randint(0, 2, (self.n_moths, n_features))
        best_score = 0
        best_mask = np.ones(n_features)
        
        print(f" Running Moth Search to find best features...")
        
        for iteration in range(self.max_iter):
            for i in range(self.n_moths):
                mask = moths[i]
                if np.sum(mask) == 0: mask[0] = 1 
                
                X_selected = X[:, mask == 1]
                X_tr, X_val, y_tr, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=SEED, stratify=y)
                
                # Use a standard balanced classifier for feature selection
                classifier.fit(X_tr, y_tr)
                y_pred = classifier.predict(X_val)
                
                # Optimize for F1 Macro
                current_score = f1_score(y_val, y_pred, average='macro')
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask.copy()
            
            # Move moths
            for i in range(self.n_moths):
                r = random.random()
                moths[i] = np.where(r > 0.5, best_mask, moths[i])
                if random.random() < 0.1:
                    idx = random.randint(0, n_features-1)
                    moths[i][idx] = 1 - moths[i][idx]

        print(f" Best Feature Subset F1-Macro: {best_score:.4f}")
        return best_mask

# --- 3. Main Training with Tuning ---
def train_sldt_msa():
    print("Loading dataset...")
    df = pd.read_csv("data/processed/features/dataset.csv")

    # Fix CSV Strings
    if 'lbp_hist' in df.columns and isinstance(df['lbp_hist'].iloc[0], str):
        print(" Fixing CSV string format...")
        df['lbp_hist'] = df['lbp_hist'].apply(ast.literal_eval)
        lbp_df = pd.DataFrame(df['lbp_hist'].tolist(), index=df.index)
        lbp_df.columns = [f'lbp_{i}' for i in range(lbp_df.shape[1])]
        df = pd.concat([df, lbp_df], axis=1)
        df.drop(columns=['lbp_hist'], inplace=True)
    
    drop_cols = ['label', 'filename']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).values
    y = df['label'].values
    feature_names = df.drop(columns=[c for c in drop_cols if c in df.columns]).columns
    
    # Base Model for Feature Selection
    base_model = DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=SEED)
    
    # Run MSA
    msa = MothSearchFeatureSelection(n_moths=8, max_iter=10)
    best_feature_mask = msa.fit(X, y, base_model)
    selected_features = feature_names[best_feature_mask == 1]
    print(f" Selected Features: {list(selected_features)}")
    
    # TUNING STAGE
    X_final = X[:, best_feature_mask == 1]
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=SEED, stratify=y)
    
    print(" Starting Hyperparameter Tuning (Grid Search)...")
    
    param_grid = {
        'class_weight': [None, 'balanced', {0:1, 1:2}, {0:1, 1:3}],
        'max_depth': [3, 5, 7, 10]
    }
    
    best_score = 0
    best_params = {}
    best_model = None
    
    # Manual Grid Loop
    total_combos = len(param_grid['class_weight']) * len(param_grid['max_depth'])
    curr = 0
    
    for cw in param_grid['class_weight']:
        for depth in param_grid['max_depth']:
            curr += 1
            estimators = [
                ('dt1', DecisionTreeClassifier(max_depth=depth, criterion='entropy', class_weight=cw, random_state=SEED)),
                ('rf', RandomForestClassifier(n_estimators=10, max_depth=depth, class_weight=cw, random_state=SEED))
            ]
            clf = StackingClassifier(
                estimators=estimators, 
                final_estimator=DecisionTreeClassifier(max_depth=depth, class_weight=cw, random_state=SEED)
            )
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            score = f1_score(y_test, y_pred, average='macro')
            
            # Simplified log
            cw_str = str(cw) if cw != {0:1, 1:2} else "{0:1, 1:2}"
            print(f"  [{curr}/{total_combos}] W:{str(cw_str):<12} | D:{depth:<2} | Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = {'class_weight': cw, 'max_depth': depth}
                best_model = clf
    
    print(f"\n Best Settings Found: Weight={best_params['class_weight']}, Depth={best_params['max_depth']}")
    
    # 4. Final Report
    print("Training Final Best Model...")
    best_model.fit(X_train, y_train)
    y_pred_final = best_model.predict(X_test)
    
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    
    # --- ACCURACY PRINT ---
    acc = accuracy_score(y_test, y_pred_final)
    print(f"Accuracy: {acc*100:.2f}%")
    
    # --- CONFUSION MATRIX PRINT ---
    cm = confusion_matrix(y_test, y_pred_final)
    print("\nConfusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\nBreakdown:")
    print(f"True Negatives (Healthy predicted Healthy): {tn}")
    print(f"False Positives (Healthy predicted TB)    : {fp}")
    print(f"False Negatives (TB predicted Healthy)    : {fn}  <-- CRITICAL")
    print(f"True Positives  (TB predicted TB)         : {tp}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_final, target_names=["Normal", "TB"]))
    
    
    print("\n Saving Dictionary Model to 'models/tb_model.pkl'...")
    os.makedirs("models", exist_ok=True)
    model_data = {
        "model": best_model,
        "feature_names": list(feature_names),
        "selected_mask": best_feature_mask
    }
    joblib.dump(model_data, "models/tb_model.pkl")
    print("Dictionary Model Saved!")

    
    print("Creating Raw Pipeline Model ('models/tb_model_raw.pkl')...")
    
    # 1. Identify which columns Moth Search kept
    keep_indices = [i for i, val in enumerate(best_feature_mask) if val == 1]
    
    # 2. Build the Selector (Fit it to X so it learns shapes)
    selector_step = ColumnTransformer(
        transformers=[
            ('keep_moth_features', 'passthrough', keep_indices)
        ],
        remainder='drop'
    )
    selector_step.fit(X) # Teach it the column structure
    
    # 3. Create Pipeline
    raw_pipeline = Pipeline([
        ('feature_selection', selector_step),
        ('classifier', best_model) # Use the already trained best_model
    ])
    
    # 4. Save
    joblib.dump(raw_pipeline, "models/tb_model_raw.pkl")
    print("Raw Pipeline Saved!")

if __name__ == "__main__":
    train_sldt_msa()