from flask import Flask, jsonify, request
import pandas as pd
import joblib
import json
import numpy as np

app = Flask(__name__)

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
    # Convert the input data into the format expected by the models (e.g., handle categorical data, scaling, etc.)
    df = pd.DataFrame([data], columns=['step', 'customer', 'age', 'gender', 'zipcodeOri', 'merchant', 'zipMerchant', 'category', 'amount', 'fraud'])
    print(df)

    reduced_data = df.drop(['zipMerchant','zipcodeOri','customer'],axis=1)

    # Load mappings from JSON file
    with open('json/mappings.json', 'r') as f:
        mappings = json.load(f)

    category_mapping = mappings['category_mapping']
    merchant_mapping = mappings['merchant_mapping']
    gender_mapping = mappings['gender_mapping']

    # Remove extra quotes from DataFrame values
    reduced_data['category'] = reduced_data['category'].str.replace("'", "", regex=False)
    reduced_data['merchant'] = reduced_data['merchant'].str.replace("'", "", regex=False)
    reduced_data['gender'] = reduced_data['gender'].str.replace("'", "", regex=False)
    reduced_data['amount'] = np.ceil(reduced_data['amount']).astype('int64')


    # Apply mappings
    reduced_data['category'] = reduced_data['category'].map(category_mapping)
    reduced_data['merchant'] = reduced_data['merchant'].map(merchant_mapping)
    reduced_data['gender'] = reduced_data['gender'].map(gender_mapping)

    return reduced_data

@app.route('/predict/knn', methods=['POST'])
def predict_knn():
    data = request.json['data']
    processed_data = preprocess(data)
    X = processed_data.drop(['fraud'], axis=1).astype('float64')
    y = processed_data['fraud']
    prediction = knn.predict(X)
    print(f"pred:{prediction}\nlabel:{y}")

    return jsonify({'prediction': int(prediction[0])})

@app.route('/predict/randomforest', methods=['POST'])
def predict_rf():
    data = request.json['data']
    processed_data = preprocess(data)
    X = processed_data.drop(['fraud'], axis=1).astype('float64')
    y = processed_data['fraud']
    prediction = rf_clf.predict(X)
    print(f"pred:{prediction}\nlabel:{y}")
    return jsonify({'prediction': int(prediction[0])})


@app.route('/predict/xgboost', methods=['POST'])
def predict_xgb():
    data = request.json['data']
    processed_data = preprocess(data)
    X = processed_data.drop(['fraud'], axis=1).astype('float64')
    y = processed_data['fraud']
    prediction = XGBoost_CLF.predict(X)
    print(f"pred:{prediction}\nlabel:{y}")
    return jsonify({'prediction': int(prediction[0])})

@app.route('/predict/ensemble', methods=['POST'])
def predict_ensemble():
    data = request.json['data']
    processed_data = preprocess(data)
    X = processed_data.drop(['fraud'], axis=1).astype('float64')
    y = processed_data['fraud']
    prediction = ensemble_model.predict(X)
    print(f"pred:{prediction}\nlabel:{y}")
    return jsonify({'prediction': int(prediction[0])})
if __name__ == '__main__':
    app.run(debug=True)
