import argparse
import sys
from src.data.ingestion import generate_synthetic_data
from src.data.preprocessing import preprocess
from src.features.feature_engineering import feature_engineering
from src.models.train import train_models
from src.models.evaluate import evaluate_models

def main():
    parser = argparse.ArgumentParser(description="EdTech Learning Engagement Analysis Pipeline")
    parser.add_argument(
        "--step", 
        type=str, 
        default="all", 
        choices=["ingestion", "preprocess", "features", "train", "evaluate", "all"],
        help="Pipeline step to run"
    )
    
    args = parser.parse_args()
    
    try:
        if args.step in ["ingestion", "all"]:
            print("\n--- STEP 2: Synthetic Data Generation ---")
            generate_synthetic_data()
            
        if args.step in ["preprocess", "all"]:
            print("\n--- STEP 3: Data Preprocessing ---")
            preprocess()
            
        if args.step in ["features", "all"]:
            print("\n--- STEP 4: Feature Engineering ---")
            feature_engineering()
            
        if args.step in ["train", "all"]:
            print("\n--- STEP 5: Modeling ---")
            train_models()
            
        if args.step in ["evaluate", "all"]:
            print("\n--- STEP 6: Model Evaluation ---")
            evaluate_models()
            
        if args.step == "all":
            print("\nDONE: Pipeline complete. Run: streamlit run dashboard/app.py")
            
    except Exception as e:
        print(f"\nERROR during pipeline execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
