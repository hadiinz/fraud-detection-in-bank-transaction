from flask import Flask, jsonify, request
import pandas as pd
import joblib
import json
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models
knn = joblib.load('models/knn_model.pkl')
rf_clf = joblib.load('models/rf_model.pkl')
XGBoost_CLF = joblib.load('models/xgb_model.pkl')
ensemble_model = joblib.load('models/ensemble_model.pkl')

# Load mappings from JSON file
with open('json/mappings.json', 'r') as f:
    mappings = json.load(f)

category_mapping = mappings['category_mapping']
merchant_mapping = mappings['merchant_mapping']
gender_mapping = mappings['gender_mapping']

def preprocess(data):
    df = pd.DataFrame(data)
    reduced_data = df.drop(['zipMerchant', 'zipcodeOri', 'customer'], axis=1)

    # Load feature names saved during training
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)

    # Clean and convert data
    reduced_data['category'] = reduced_data['category'].str.replace("'", "", regex=False)
    reduced_data['merchant'] = reduced_data['merchant'].str.replace("'", "", regex=False)
    reduced_data['gender'] = reduced_data['gender'].str.replace("'", "", regex=False)
    reduced_data['amount'] = np.ceil(reduced_data['amount']).astype('int64')

    # Apply mappings
    reduced_data['category'] = reduced_data['category'].map(category_mapping)
    reduced_data['merchant'] = reduced_data['merchant'].map(merchant_mapping)
    reduced_data['gender'] = reduced_data['gender'].map(gender_mapping)

    # Reorder the features to match the training order
    features_only = reduced_data.drop(['fraud'], axis=1) 
    features_only = features_only.reindex(columns=feature_names, fill_value=0)

    features_only['fraud'] = reduced_data['fraud']

    return features_only


def make_predictions(model, data):
    processed_data = preprocess(data)
    print(f"Processed data: {processed_data}") 

    X = processed_data.drop(['fraud'], axis=1).astype('float64')  
    actuals = processed_data['fraud'].tolist() 
    predictions = model.predict(X)

    return predictions, actuals

@app.route('/predict/knn', methods=['POST'])
def predict_knn():
    data = request.json['data']
    print(f"Received data for KNN: {data}")  
    predictions, actuals = make_predictions(knn, data)
    
    result = [{'prediction': int(pred), 'actual': int(actual)} for pred, actual in zip(predictions, actuals)]
    print(f"KNN Predictions: {result}") 
    return jsonify(result)


@app.route('/predict/randomforest', methods=['POST'])
def predict_rf():
    data = request.json['data']
    predictions, actuals = make_predictions(rf_clf, data)
    
    result = [{'prediction': int(pred), 'actual': int(actual)} for pred, actual in zip(predictions, actuals)]
    return jsonify(result)

@app.route('/predict/xgboost', methods=['POST'])
def predict_xgb():
    data = request.json['data']
    predictions, actuals = make_predictions(XGBoost_CLF, data)
    
    result = [{'prediction': int(pred), 'actual': int(actual)} for pred, actual in zip(predictions, actuals)]
    return jsonify(result)

@app.route('/predict/ensemble', methods=['POST'])
def predict_ensemble():
    data = request.json['data']
    predictions, actuals = make_predictions(ensemble_model, data)
    
    result = [{'prediction': int(pred), 'actual': int(actual)} for pred, actual in zip(predictions, actuals)]
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
