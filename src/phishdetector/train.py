import os, json, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import mlflow, mlflow.sklearn

DATA_PATH = os.environ.get("DATA_PATH", "data/phishing_data.csv")
OUT_DIR   = os.environ.get("OUT_DIR", "model_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Adjust column names if needed; assume "label" is binary {0,1}
    y = df["label"].astype(int)
    X = df.drop(columns=["label"])
    # Only numeric features at first pass (robust & simple)
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    X = X[num_cols].astype(float)
    return X, y, num_cols

def build_pipeline(num_cols):
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(with_mean=False), num_cols)],
        remainder="drop"
    )
    pipe = Pipeline(steps=[
        ("pre", pre),
        ("ros", RandomOverSampler(random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=42
        ))
    ])
    return pipe

def main():
    X, y, num_cols = load_data()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = build_pipeline(num_cols)

    mlflow.set_experiment("phish-detector")
    with mlflow.start_run():
        mlflow.log_params({
            "model": "RandomForest",
            "n_estimators": 400,
            "random_state": 42,
            "scaler_with_mean": False
        })
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        preds = (proba >= 0.5).astype(int)

        roc = roc_auc_score(yte, proba)
        ap  = average_precision_score(yte, proba)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("avg_precision", ap)
        mlflow.sklearn.log_model(pipe, "model")

        report = classification_report(yte, preds, output_dict=True)
        with open(os.path.join(OUT_DIR, "classification_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        # Save pipeline + required feature list for serving
        joblib.dump(
            {"model": pipe, "required_features": num_cols},
            os.path.join(OUT_DIR, "model.joblib")
        )
        print(f"ROC-AUC={roc:.4f}  AP={ap:.4f}")
        print("Saved:", os.path.join(OUT_DIR, "model.joblib"))

if __name__ == "__main__":
    main()
