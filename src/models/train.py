import pandas as pd
import numpy as np
import yaml
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_models(config_path='config/config.yaml'):
    """
    Trains classification models and performs student segmentation.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    feature_path = config['paths']['feature_data']
    pipeline_path = config['paths']['pipeline_path']
    
    if not os.path.exists(feature_path):
        logger.error(f"Features file not found at {feature_path}")
        return
    
    df = pd.read_csv(feature_path)
    preprocessor = joblib.load(pipeline_path)
    
    # 1. K-Means Clustering (k=4) for student segmentation
    logger.info("Performing student segmentation...")
    # Use engagement and performance metrics for clustering
    cluster_features = ['engagement_score', 'quiz_scores_avg', 'modules_completed', 'total_logins']
    X_cluster = df[cluster_features]
    # Scaling is good for KMeans
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=config['modeling']['kmeans_clusters'], random_state=config['modeling']['random_state'], n_init=10)
    df['student_segment'] = kmeans.fit_transform(X_cluster_scaled).argmin(axis=1) # Just labels
    df['student_segment'] = kmeans.labels_
    
    # Save segmentation column back to features CSV
    df.to_csv(feature_path, index=False)
    logger.info(f"Student segments saved to {feature_path}")
    
    # 2. Classification setup
    X = df.drop(columns=['dropped_out', 'student_id', 'enrollment_date', 'student_segment'])
    y = df['dropped_out']
    
    # Apply preprocessing pipeline
    X_processed = preprocessor.transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, 
        test_size=config['modeling']['train_test_split'], 
        stratify=y, 
        random_state=config['modeling']['random_state']
    )
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=config['modeling']['smote_random_state'])
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 3. Model Training & MLflow tracking
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config['modeling']['random_state']),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=config['modeling']['random_state']),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=config['modeling']['random_state'])
    }
    
    best_score = 0
    best_model = None
    best_model_name = ""
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            logger.info(f"Training {name}...")
            
            if name == "XGBoost":
                # GridSearchCV for XGBoost
                param_grid = config['modeling']['xgb_params']
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['modeling']['random_state'])
                grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1)
                grid_search.fit(X_train_res, y_train_res)
                model = grid_search.best_estimator_
                mlflow.log_params(grid_search.best_params_)
                # Save XGBoost specifically for SHAP
                joblib.dump(model, 'models/xgb_model.pkl')
            elif name == "Random Forest":
                model.fit(X_train_res, y_train_res)
                joblib.dump(model, 'models/rf_model.pkl')
            else:
                model.fit(X_train_res, y_train_res)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred)
            }
            
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model")
            
            logger.info(f"{name} Metrics: {metrics}")
            
            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                best_model = model
                best_model_name = name
    
    # 4. Save best model
    logger.info(f"Best Model: {best_model_name} (F1: {best_score:.4f})")
    os.makedirs(os.path.dirname(config['paths']['model_path']), exist_ok=True)
    joblib.dump(best_model, config['paths']['model_path'])
    
    # Print final classification report
    y_pred_best = best_model.predict(X_test)
    print("\nFinal Classification Report (Best Model):")
    print(classification_report(y_test, y_pred_best))
    
    # Save the test data for evaluation step
    X_test_df = pd.DataFrame(X_test) # Column names might be lost after transform, but evaluate needs them
    # Better to save as numpy or joblib for evaluate.py
    joblib.dump((X_test, y_test), 'data/processed/test_data.pkl')

if __name__ == "__main__":
    train_models()
