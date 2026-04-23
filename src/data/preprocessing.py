import pandas as pd
import numpy as np
import yaml
import os
import logging
from sklearn.preprocessing import LabelEncoder
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess(config_path='config/config.yaml'):
    """
    Cleans raw data, handles missing values, caps outliers, and encodes categoricals.
    """
    logger.info("Starting preprocessing...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    raw_path = config['paths']['raw_data']
    processed_path = config['paths']['processed_data']
    
    if not os.path.exists(raw_path):
        logger.error(f"Raw data file not found at {raw_path}")
        return
    
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} records.")
    
    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed duplicates. Remaining records: {len(df)}")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            logger.info(f"Filled nulls in {col} with median: {median}")
            
    for col in categorical_cols:
        if df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
            logger.info(f"Filled nulls in {col} with mode: {mode}")
            
    # Outlier detection and capping (IQR method)
    for col in ['total_logins', 'avg_session_duration_mins', 'quiz_scores_avg', 'forum_posts']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        logger.info(f"Capped outliers for {col}")
        
    # Categorical Encoding (LabelEncoder for gender/region)
    # We also need course_id for dropout analysis, but instruction specifies gender/region for LabelEncoder
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    
    le_region = LabelEncoder()
    df['region'] = le_region.fit_transform(df['region'])
    
    # Save encoders for predictor
    os.makedirs('models', exist_ok=True)
    joblib.dump(le_gender, 'models/le_gender.pkl')
    joblib.dump(le_region, 'models/le_region.pkl')
    logger.info("Saved LabelEncoders to models/")
    
    # Save cleaned data
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Cleaned data saved to {processed_path}")

if __name__ == "__main__":
    preprocess()
