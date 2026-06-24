import pandas as pd
from datetime import datetime
import joblib
import mlflow
import mlflow.xgboost
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

X_train_ref = pd.read_csv(r"D:\XGBoost - Project\data\processed\X_train.csv")
y_train_ref = pd.read_csv(r"D:\XGBoost - Project\data\processed\y_train.csv").values.ravel()

X_curr = pd.read_csv(r"D:\XGBoost - Project\data\processed\X_test.csv")
y_curr = pd.read_csv(r"D:\XGBoost - Project\data\processed\y_test.csv").values.ravel()

column_mapping = ColumnMapping()
column_mapping.numerical_features = X_train_ref.columns.tolist()
column_mapping.target = 'target'  
column_mapping.prediction = None  

drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=X_train_ref, current_data=X_curr, column_mapping=column_mapping)
drift_report.save_html("drift_report.html")

perf_report = Report(metrics=[ClassificationPreset()])
perf_report.run(reference_data=X_train_ref, current_data=X_curr,
                reference_data_y=y_train_ref, current_data_y=y_curr,
                column_mapping=column_mapping)
perf_report.save_html("performance_report.html")
