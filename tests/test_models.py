import pytest
import joblib
import yaml
import os

@pytest.fixture
def config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_best_model_loads(config):
    model_path = config['paths']['model_path']
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        assert model is not None
    else:
        pytest.skip("Model file not found")

def test_prediction_output(config):
    model_path = config['paths']['model_path']
    test_data_path = 'data/processed/test_data.pkl'
    
    if os.path.exists(model_path) and os.path.exists(test_data_path):
        model = joblib.load(model_path)
        X_test, _ = joblib.load(test_data_path)
        
        # Test predict
        pred = model.predict(X_test[:5])
        for p in pred:
            assert p in [0, 1]
            
        # Test predict_proba
        proba = model.predict_proba(X_test[:5])
        assert proba.shape[1] == 2
        assert (proba >= 0).all() and (proba <= 1).all()
    else:
        pytest.skip("Model or test data not found")
