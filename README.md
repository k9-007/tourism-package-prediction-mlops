# Tourism Package Prediction — MLOps Pipeline

End-to-end MLOps pipeline for **"Visit with Us"** that predicts whether a customer will purchase the new
**Wellness Tourism Package**. The pipeline is automated with **GitHub Actions**, experiments are tracked
with **MLflow**, and the model is served through a **Streamlit** app on Streamlit Community Cloud.

## Repository structure

```
.
├── .github/workflows/pipeline.yml      # CI/CD pipeline: register -> prep -> train -> commit model
├── tourism_project/
│   ├── data/tourism.csv                # Registered dataset (lives in the repo)
│   ├── model_building/
│   │   ├── data_register.py            # Validates schema and prints a dataset summary
│   │   ├── prep.py                     # Cleans data and writes train/test splits
│   │   └── train.py                    # GridSearchCV + MLflow tracking, saves best model
│   ├── deployment/
│   │   ├── app.py                      # Streamlit front-end
│   │   ├── requirements.txt            # App dependencies (Streamlit Community Cloud)
│   │   ├── best_tourism_model_v1.joblib# Trained model committed by the pipeline
│   │   └── model_metrics.json          # Metrics/params of the committed model
│   └── requirements.txt                # Pipeline dependencies (GitHub Actions)
└── Tourism_Package_Prediction_MLOps.ipynb  # Project notebook
```

## Pipeline

Every push to `main` (or a manual **Run workflow**) runs three sequential jobs:

1. **Register Dataset** – validates `tourism_project/data/tourism.csv` and uploads it as an artifact.
2. **Data Preparation** – drops identifier columns, fixes categorical typos, removes duplicates,
   splits the data (stratified 80/20) and uploads the splits as the `data-splits` artifact.
3. **Model Training** – downloads the splits, tunes an XGBoost pipeline with `GridSearchCV`, logs every
   trial to MLflow, evaluates the best model and commits it back to `tourism_project/deployment/`.

## Model

`StandardScaler` + `OneHotEncoder` preprocessing feeding an `XGBClassifier` with `scale_pos_weight` to
handle the ~19 % positive class. Hyperparameters are tuned with 5-fold CV optimising F1 on the buyer class.

## Running the app locally

```bash
pip install -r tourism_project/deployment/requirements.txt
streamlit run tourism_project/deployment/app.py
```
