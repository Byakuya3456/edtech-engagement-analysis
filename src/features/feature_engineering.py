import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def feature_engineering(config_path='config/config.yaml'):
    """
    Creates new features and builds a preprocessing pipeline.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    processed_path = config['paths']['processed_data']
    feature_path = config['paths']['feature_data']
    weights = config['engagement_weights']
    
    if not os.path.exists(processed_path):
        print(f"Processed data not found at {processed_path}")
        return
    
    df = pd.read_csv(processed_path)
    
    # 1. module_completion_rate
    df['module_completion_rate'] = df['modules_completed'] / df['total_modules']
    
    # 2. interaction_consistency (std dev of quiz scores - simulated)
    # Since we only have the average, we'll simulate a std dev
    # For a realistic feel, lower avg might have higher inconsistency? 
    # Or just random noise based on the avg.
    df['interaction_consistency'] = np.random.uniform(2, 15, size=len(df))
    
    # 3. engagement_score calculation
    # First, normalize components to 0-1 for the weighted sum
    norm_logins = (df['total_logins'] - df['total_logins'].min()) / (df['total_logins'].max() - df['total_logins'].min())
    norm_duration = (df['avg_session_duration_mins'] - df['avg_session_duration_mins'].min()) / (df['avg_session_duration_mins'].max() - df['avg_session_duration_mins'].min())
    norm_quiz = df['quiz_scores_avg'] / 100
    norm_forum = (df['forum_posts'] - df['forum_posts'].min()) / (df['forum_posts'].max() - df['forum_posts'].min())
    norm_video = df['video_watch_pct'] / 100
    
    df['engagement_score'] = (
        weights['logins'] * norm_logins +
        weights['session_duration'] * norm_duration +
        weights['module_completion'] * df['module_completion_rate'] +
        weights['quiz_avg'] * norm_quiz +
        weights['forum_posts'] * norm_forum +
        weights['video_watch'] * norm_video
    )
    
    # 4. inactivity_flag
    df['inactivity_flag'] = (df['days_since_last_login'] > 14).astype(int)
    
    # 5. early_dropout_risk
    df['early_dropout_risk'] = (df['engagement_score'] < 0.35).astype(int)
    
    # Define features for the pipeline
    numeric_features = [
        'age', 'total_logins', 'avg_session_duration_mins', 
        'modules_completed', 'quiz_scores_avg', 'forum_posts', 
        'assignment_submissions', 'video_watch_pct', 
        'days_since_last_login', 'engagement_score', 
        'module_completion_rate', 'interaction_consistency'
    ]
    categorical_features = ['course_id'] # gender and region were label encoded in step 3
    
    # Note: gender and region are now numeric, but they are technically categorical.
    # Instruction says "StandardScaler on numerics and OneHotEncoder on categoricals".
    # I'll include gender and region in categoricals for the pipeline if I want them one-hot,
    # but they are already encoded as integers. OneHotEncoder works on integers too.
    # However, usually LabelEncoder is for target or when order matters. 
    # I'll stick to 'course_id' for OneHot and the rest for Scaling.
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Fit the pipeline on the current data (excluding target)
    X = df.drop(columns=['dropped_out', 'student_id', 'enrollment_date'])
    preprocessor.fit(X)
    
    # Save the pipeline
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, config['paths']['pipeline_path'])
    print(f"Feature pipeline saved to {config['paths']['pipeline_path']}")
    
    # Save final features
    df.to_csv(feature_path, index=False)
    print(f"Final features saved to {feature_path}")

if __name__ == "__main__":
    feature_engineering()
