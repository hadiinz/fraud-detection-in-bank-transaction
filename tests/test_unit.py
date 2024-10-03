import pytest
from app import preprocess, knn, rf_clf, XGBoost_CLF, ensemble_model

def test_preprocess():
    data = [{
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
  
    processed_data = preprocess(data)
    # remove unnecessary columns
    assert 'zipMerchant' not in processed_data.columns
    assert 'zipcodeOri' not in processed_data.columns
    assert 'customer' not in processed_data.columns
    # category mapping
    assert processed_data['category'].iloc[0] is not None
    assert processed_data['category'].iloc[0] == 4
    # Rounded amount
    assert processed_data['amount'].iloc[0] == 4450  
    
def test_knn_model():
    data = [
    {
        "step": 88,
        "customer": "C583110837",
        "age": 3,
        "gender": "M",
        "zipcodeOri": "28007",
        "merchant": "M480139044",
        "zipMerchant": "28007",
        "category": "es_health",
        "amount": 4449.26,
        "fraud": 1
    },
   
    {
        "step": 0,
        "customer": "C1093826151",
        "age": 4,
        "gender": "M",
        "zipcodeOri": "28007",
        "merchant": "M348934600",
        "zipMerchant": "28007",
        "category": "es_transportation",
        "amount": 4.55,
        "fraud": 0
    }
]
    
    processed_data = preprocess(data)
    X = processed_data.drop(['fraud'], axis=1)
    prediction = knn.predict(X)
    # assert that it's fraud
    assert prediction[0] == 1
    assert prediction[1] == 0

def test_rf_model():
    data = [
    {
        "step": 88,
        "customer": "C583110837",
        "age": 3,
        "gender": "M",
        "zipcodeOri": "28007",
        "merchant": "M480139044",
        "zipMerchant": "28007",
        "category": "es_health",
        "amount": 4449.26,
        "fraud": 1
    },
   
    {
        "step": 0,
        "customer": "C1093826151",
        "age": 4,
        "gender": "M",
        "zipcodeOri": "28007",
        "merchant": "M348934600",
        "zipMerchant": "28007",
        "category": "es_transportation",
        "amount": 4.55,
        "fraud": 0
    }
]
    
    processed_data = preprocess(data)
    X = processed_data.drop(['fraud'], axis=1)
    prediction = rf_clf.predict(X)
    
    assert prediction[0] == 1
    assert prediction[1] == 0

def test_xgb_model():
    data = [
    {
        "step": 88,
        "customer": "C583110837",
        "age": 3,
        "gender": "M",
        "zipcodeOri": "28007",
        "merchant": "M480139044",
        "zipMerchant": "28007",
        "category": "es_health",
        "amount": 4449.26,
        "fraud": 1
    },
   
    {
        "step": 0,
        "customer": "C1093826151",
        "age": 4,
        "gender": "M",
        "zipcodeOri": "28007",
        "merchant": "M348934600",
        "zipMerchant": "28007",
        "category": "es_transportation",
        "amount": 4.55,
        "fraud": 0
    }
]   
    processed_data = preprocess(data)
    X = processed_data.drop(['fraud'], axis=1)
    prediction = XGBoost_CLF.predict(X)
    
    assert prediction[0] == 1
    assert prediction[1] == 0
