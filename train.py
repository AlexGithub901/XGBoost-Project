import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def main():
    X_train = pd.read_csv(r"D:\XGBoost - Project\data\processed\X_train.csv")
    X_test  = pd.read_csv(r"D:\XGBoost - Project\data\processed\X_test.csv")
    y_train = pd.read_csv(r"D:\XGBoost - Project\data\processed\y_train.csv").values.ravel()
    y_test  = pd.read_csv(r"D:\XGBoost - Project\data\processed\y_test.csv").values.ravel()

    def clean_cols(df):
        df.columns = [col.replace("[", "").replace("]", "").replace(" ", "_")
                      for col in df.columns]
        return df
    X_train = clean_cols(X_train)
    X_test  = clean_cols(X_test)


    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }

    mlflow.set_experiment("predictive_maintenance_xgboost")

    with mlflow.start_run():
        mlflow.log_params(params)

        model = xgb.XGBClassifier(enable_categorical=False, **params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        mlflow.xgboost.log_model(model, "xgboost_model")
        model_uri = mlflow.get_artifact_uri("xgboost_model")
        mlflow.register_model(model_uri, "failure_predictor_xgb")

        print(f"Training complete. Accuracy: {acc:.3f}, F1: {f1:.3f}")

if __name__ == "__main__":
    main()