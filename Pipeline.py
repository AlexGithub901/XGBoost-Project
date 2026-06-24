import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')                # non‑interactive backend for CI
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ============== RELATIVE PATHS (safe for any OS) ==============
RAW_DATA = "data/ai4i2020.csv"
PROCESSED_DIR = "data/processed"
X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.csv")
X_TEST_PATH  = os.path.join(PROCESSED_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.csv")
Y_TEST_PATH  = os.path.join(PROCESSED_DIR, "y_test.csv")
SCALER_PATH  = os.path.join("models", "scaler.joblib")

EXPERIMENT_NAME = "predictive_maintenance_xgboost"
MODEL_REGISTRY_NAME = "failure_predictor_xgb"
N_TRIALS = 30
CV_FOLDS = 5
RANDOM_STATE = 42

def preprocess_and_save():
    """Preprocess raw data and save train/test CSV + scaler."""
    print("Processed data not found. Running preprocessing...")
    df = pd.read_csv(RAW_DATA)

    # Rename columns – remove brackets for XGBoost
    rename_map = {
        "Air temperature [K]": "Air_temperature_K",
        "Process temperature [K]": "Process_temperature_K",
        "Rotational speed [rpm]": "Rotational_speed_rpm",
        "Torque [Nm]": "Torque_Nm",
        "Tool wear [min]": "Tool_wear_min"
    }
    df.rename(columns=rename_map, inplace=True)

    # Drop non‑feature / leaky columns
    drop_cols = ["UDI", "Product ID", "Type", "TWF", "HDF", "PWF", "OSF", "RNF"]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Feature engineering
    df["power"] = df["Torque_Nm"] * df["Rotational_speed_rpm"]

    # Split
    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(X_TRAIN_PATH, index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)
    print("Preprocessing complete. Files saved to", PROCESSED_DIR)

def load_data():
    """Load processed data; create it if missing."""
    needed = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]
    if not all(os.path.exists(p) for p in needed):
        preprocess_and_save()

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test  = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
    y_test  = pd.read_csv(Y_TEST_PATH).values.ravel()

    # Ensure column names are clean (remove any residual brackets/spaces)
    X_train.columns = X_train.columns.str.replace(r"[\[\] ]", "", regex=True)
    X_test.columns  = X_test.columns.str.replace(r"[\[\] ]", "", regex=True)
    return X_train, X_test, y_train, y_test

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }

def plot_confusion_matrix(y_true, y_pred, path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def plot_feature_importance(model, feature_names, path="feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,5))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def objective(trial, X_train, y_train):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state": RANDOM_STATE,
        "use_label_encoder": False
    }
    model = xgb.XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
    return np.mean(scores)

def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Data loaded. Train: {X_train.shape}, Test: {X_test.shape}")

    mlflow.set_tracking_uri("file:///" + os.path.abspath("mlruns").replace("\\", "/"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    if OPTUNA_AVAILABLE:
        print(f"Starting hyperparameter optimization with Optuna ({N_TRIALS} trials)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_TRIALS)
        best_params = study.best_params
        print(f"Best F1 from CV: {study.best_value:.4f}")
        print("Best parameters:", best_params)
    else:
        print("Optuna not installed. Using default hyperparameters.")
        best_params = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1
        }

    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        **best_params
    }
    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    with mlflow.start_run(run_name="tuned_xgboost") as run:
        mlflow.log_params(final_params)
        mlflow.log_metrics(metrics)

        cm_path = plot_confusion_matrix(y_test, y_pred)
        fi_path = plot_feature_importance(model, X_train.columns)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)

        mlflow.xgboost.log_model(model, "xgboost_model")

        model_uri = f"runs:/{run.info.run_id}/xgboost_model"
        mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)

        print(f"\nModel registered as '{MODEL_REGISTRY_NAME}' (version may have increased).")

    os.remove(cm_path)
    os.remove(fi_path)

    print("Training pipeline complete.")
    return metrics

if __name__ == "__main__":
    MIN_F1 = 0.7
    final_metrics = main()

    if final_metrics["f1_score"] < MIN_F1:
        print(f"F1 score {final_metrics['f1_score']:.3f} below threshold {MIN_F1}. Failing CI.")
        sys.exit(1)
    else:
        print(f"F1 score {final_metrics['f1_score']:.3f} passes CI gate. Proceeding.")