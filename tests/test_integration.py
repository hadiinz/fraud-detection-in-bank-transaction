import pytest
import json

def test_knn_prediction(client):
    mock_data = {
        "data": [{
            'step': 88,
            'customer': 'C583110837',
            'age': 3,
            'gender': 'M',
            'zipcodeOri': '28007',
            'merchant': 'M480139044',
            'zipMerchant': '28007',
            'category': 'es_health',
            'amount': 4449.26,
            'fraud': 1
        }]
    }
    
    response = client.post('/predict/knn', json=mock_data)
    assert response.status_code == 200
    result = response.get_json()
    assert 'prediction' in result[0]
    assert 'actual' in result[0]

def test_rf_prediction(client):
    mock_data = {
        "data": [{
            'step': 88,
            'customer': 'C583110837',
            'age': 3,
            'gender': 'M',
            'zipcodeOri': '28007',
            'merchant': 'M480139044',
            'zipMerchant': '28007',
            'category': 'es_health',
            'amount': 4449.26,
            'fraud': 1
        }]
    }
    
    response = client.post('/predict/randomforest', json=mock_data)
    assert response.status_code == 200
    result = response.get_json()
    assert 'prediction' in result[0]
    assert 'actual' in result[0]

def test_xgb_prediction(client):
    mock_data = {
        "data": [{
            'step': 88,
            'customer': 'C583110837',
            'age': 3,
            'gender': 'M',
            'zipcodeOri': '28007',
            'merchant': 'M480139044',
            'zipMerchant': '28007',
            'category': 'es_health',
            'amount': 4449.26,
            'fraud': 1
        }]
    }
    
    response = client.post('/predict/xgboost', json=mock_data)
    assert response.status_code == 200
    result = response.get_json()
    assert 'prediction' in result[0]
    assert 'actual' in result[0]

def test_ensemble_prediction(client):
    mock_data = {
        "data": [{
            'step': 88,
            'customer': 'C583110837',
            'age': 3,
            'gender': 'M',
            'zipcodeOri': '28007',
            'merchant': 'M480139044',
            'zipMerchant': '28007',
            'category': 'es_health',
            'amount': 4449.26,
            'fraud': 1
        }]
    }
    
    response = client.post('/predict/ensemble', json=mock_data)
    assert response.status_code == 200
    result = response.get_json()
    assert 'prediction' in result[0]
    assert 'actual' in result[0]
