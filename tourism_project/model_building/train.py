"""
Model Building step of the Tourism Package Prediction pipeline.

1. Loads the train/test splits produced by prep.py (downloaded from the
   previous job's workflow artifact).
2. Defines a preprocessing + XGBoost pipeline and a hyperparameter grid.
3. Tunes the model with GridSearchCV, logging EVERY parameter combination as a
   nested MLflow run and the best parameters/metrics on the parent run.
4. Evaluates the best model on train and test data.
5. Saves the best model into tourism_project/deployment/ so the workflow can
   commit it back to the repository for the Streamlit app to load.
"""
import json
import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- MLflow configuration ------------------------------------------------------
# The workflow starts a local MLflow server on port 5000 before running this
# script. MLFLOW_TRACKING_URI can override it (e.g. for a remote server).
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-package-prediction")

# ---- 1. Load the splits from the workflow artifact -----------------------------
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()
print(f"Loaded splits -> train: {Xtrain.shape}, test: {Xtest.shape}")

# ---- 2. Define features, preprocessing and model ------------------------------
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# Handle class imbalance (~81% / 19%) by up-weighting the positive class
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"scale_pos_weight (neg/pos ratio): {class_weight:.3f}")

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter grid (36 combinations x 5 folds). Kept moderate so the job
# finishes in a few minutes on a GitHub Actions runner.
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.7, 1.0],
    "xgbclassifier__subsample": [0.8, 1.0],
}
n_combos = 1
for v in param_grid.values():
    n_combos *= len(v)
print(f"Hyperparameter grid: {n_combos} combinations, 5-fold CV")

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_tourism_model_v1.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---- 3. Tune with GridSearchCV and track in MLflow ------------------------------
with mlflow.start_run(run_name="xgboost-gridsearch") as parent_run:
    # F1 balances precision and recall on the minority (buyer) class, which is
    # what marketing actually cares about: find buyers without wasting calls.
    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0
    )
    grid_search.fit(Xtrain, ytrain)

    # Log every tuned parameter combination as its own nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True, run_name=f"trial-{i:02d}"):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1", results["std_test_score"][i])
            mlflow.log_metric("rank", int(results["rank_test_score"][i]))
    print(f"Logged {len(results['params'])} tuning trials to MLflow")

    # Log the winning parameters on the parent run
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("scale_pos_weight", round(class_weight, 4))
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring", "f1")
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)
    print("Best params:", grid_search.best_params_)
    print(f"Best CV F1 : {grid_search.best_score_:.4f}")

    # ---- 4. Evaluate the best model --------------------------------------------
    best_model = grid_search.best_estimator_
    classification_threshold = 0.5

    y_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_train_pred = (y_train_proba >= classification_threshold).astype(int)
    y_test_pred = (y_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_train_pred, output_dict=True)
    test_report = classification_report(ytest, y_test_pred, output_dict=True)

    metrics = {
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "train_roc_auc": roc_auc_score(ytrain, y_train_proba),
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
        "test_roc_auc": roc_auc_score(ytest, y_test_proba),
    }
    mlflow.log_metrics(metrics)

    print("\nClassification report (test set):")
    print(classification_report(ytest, y_test_pred))
    print("Confusion matrix (test set):")
    print(confusion_matrix(ytest, y_test_pred))
    print("\nSummary metrics:")
    for k, v in metrics.items():
        print(f"  {k:<16}: {v:.4f}")

    # ---- 5. Save the best model for deployment ---------------------------------
    joblib.dump(best_model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(
            {"best_params": grid_search.best_params_, "metrics": metrics,
             "mlflow_run_id": parent_run.info.run_id},
            f, indent=2,
        )
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    mlflow.log_artifact(METRICS_PATH, artifact_path="model")
    print(f"\nModel saved to   {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
