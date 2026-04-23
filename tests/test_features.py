import pytest
import pandas as pd
import yaml

@pytest.fixture
def config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def feature_data(config):
    return pd.read_csv(config['paths']['feature_data'])

def test_engagement_score_range(feature_data):
    assert feature_data['engagement_score'].min() >= 0
    assert feature_data['engagement_score'].max() <= 1.0000000000000002 # Float precision

def test_module_completion_rate_range(feature_data):
    assert feature_data['module_completion_rate'].min() >= 0
    assert feature_data['module_completion_rate'].max() <= 1

def test_inactivity_flag_binary(feature_data):
    assert set(feature_data['inactivity_flag'].unique()).issubset({0, 1})
