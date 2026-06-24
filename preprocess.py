# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    df = pd.read_csv(r"D:\XGBoost - Project\data\ai4i2020.csv")

    df = df.drop(columns=[
        "UDI",           
        "Product ID",    
        "Type",          
        "TWF",           
        "HDF",           
        "PWF",           
        "OSF",           
        "RNF"            
    ])

    df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]

    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
        "data/processed/X_train.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
        "data/processed/X_test.csv", index=False
    )
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    joblib.dump(scaler, "models/scaler.joblib")

    print("Preprocessing complete.")
    print(f"X_train shape: {X_train_scaled.shape}")
    print(f"X_test shape:  {X_test_scaled.shape}")
    print("Files saved to data/processed/ and models/")

if __name__ == "__main__":
    main()