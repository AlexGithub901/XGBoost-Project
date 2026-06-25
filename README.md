# XGBoost-MLOps: Predictive Maintenance Pipeline

[![MLOps CI/CD](https://github.com/AlexGithub901/XGBoost-Project/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/AlexGithub901/XGBoost-Project/actions/workflows/ci-cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready **MLOps pipeline** for predictive maintenance, built with XGBoost, Optuna, MLflow, FastAPI, and Docker. The entire machine learning lifecycle—from data preprocessing to a containerized API—is automated, reproducible, and versioned.

## Features

- **Automated preprocessing** – data cleaning, feature engineering, scaling
- **Hyperparameter tuning** with Optuna (or defaults if offline)
- **Experiment tracking** & model registry using MLflow
- **Continuous Integration (CI)** – trains, evaluates, and builds Docker image on every push
- **Quality gate** – only models with F1 > 0.7 are promoted
- **REST API** (FastAPI) for real-time inference
- **Docker** support for portable deployment

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlexGithub901/XGBoost-Project.git
   cd XGBoost-Project
