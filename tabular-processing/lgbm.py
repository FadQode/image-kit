import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd


def objective_lgbm(trial, X_train, y_train, X_val, y_val):
    """Objective function untuk optimasi hyperparameter LGBM dengan Optuna."""

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric="auc",
              early_stopping_rounds=50,
              verbose=False)

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)

    return auc


def tune_lgbm_with_optuna(X_train, y_train, X_val, y_val, n_trials=50):
    """Melakukan tuning hyperparameter LGBM dengan Optuna."""
    study = optuna.create_study(direction="maximize", study_name="LGBM Optimization")
    study.optimize(lambda trial: objective_lgbm(trial, X_train, y_train, X_val, y_val),
                   n_trials=n_trials,
                   n_jobs=-1)
    print(f"✅ Best AUC: {study.best_value:.4f}")
    print("Best params:", study.best_params)
    return study.best_params


def train_best_lgbm(X_train, y_train, X_val, y_val, best_params):
    """Melatih model LGBM dengan parameter terbaik hasil tuning."""
    best_model = LGBMClassifier(
        **best_params,
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        verbosity=-1,
    )
    best_model.fit(X_train, y_train,
                   eval_set=[(X_val, y_val)],
                   eval_metric="auc",
                   early_stopping_rounds=50,
                   verbose=False)
    return best_model


def evaluate_model(model, X_val, y_val):
    """Evaluasi model LGBM dan menampilkan metrik performa."""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1 Score": f1_score(y_val, y_pred),
        "ROC AUC": roc_auc_score(y_val, y_proba)
    }

    print("\n🔍 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))

    return metrics


def train_lgbm_pipeline(X_train, y_train, X_val, y_val, n_trials=50):
    """Pipeline lengkap untuk tuning dan training LGBM."""
    print("🚀 Starting hyperparameter tuning with Optuna...")
    best_params = tune_lgbm_with_optuna(X_train, y_train, X_val, y_val, n_trials)
    print("\n📈 Training final model with best parameters...")
    best_model = train_best_lgbm(X_train, y_train, X_val, y_val, best_params)
    metrics = evaluate_model(best_model, X_val, y_val)
    return best_model, best_params, metrics
