import pytest
import pandas as pd
import yaml

@pytest.fixture
def config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def cleaned_data(config):
    return pd.read_csv(config['paths']['processed_data'])

def test_no_nulls(cleaned_data):
    assert cleaned_data.isnull().sum().sum() == 0

def test_unique_ids(cleaned_data):
    assert cleaned_data['student_id'].is_unique
