import pandas as pd
import numpy as np
import yaml
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import shap

def evaluate_models(config_path='config/config.yaml'):
    """
    Generates evaluation plots and SHAP analysis.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    reports_dir = config['paths']['reports_dir']
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load data and models
    X_test, y_test = joblib.load('data/processed/test_data.pkl')
    best_model = joblib.load(config['paths']['model_path'])
    preprocessor = joblib.load(config['paths']['pipeline_path'])
    
    # Get feature names for plotting
    # ColumnTransformer can be tricky to get feature names from
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(['course_id'])
    num_feature_names = [
        'age', 'total_logins', 'avg_session_duration_mins', 
        'modules_completed', 'quiz_scores_avg', 'forum_posts', 
        'assignment_submissions', 'video_watch_pct', 
        'days_since_last_login', 'engagement_score', 
        'module_completion_rate', 'interaction_consistency'
    ]
    feature_names = np.concatenate([num_feature_names, cat_feature_names])
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(reports_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 2. ROC-AUC Curve
    plt.figure(figsize=(8, 6))
    y_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(reports_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # 3. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(reports_dir, 'pr_curve.png'), dpi=300)
    plt.close()
    
    # 4. Feature Importance (Random Forest)
    if os.path.exists('models/rf_model.pkl'):
        rf_model = joblib.load('models/rf_model.pkl')
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[-15:] # Top 15
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importances (Random Forest)')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, 'feature_importance.png'), dpi=300)
        plt.close()
    
    # 5. SHAP analysis (XGBoost)
    if os.path.exists('models/xgb_model.pkl'):
        xgb_model = joblib.load('models/xgb_model.pkl')
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        
        # SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot (XGBoost)')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, 'shap_summary.png'), dpi=300)
        plt.close()
        
        # SHAP Bar Plot (Top 10)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", max_display=10, show=False)
        plt.title('Top 10 Features (SHAP Bar Plot)')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, 'shap_bar.png'), dpi=300)
        plt.close()
        
    print(f"Evaluation plots saved to {reports_dir}")

if __name__ == "__main__":
    evaluate_models()
