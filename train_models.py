import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import joblib
from imblearn.over_sampling import SMOTE
import json 
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def remove_first_samples(data):
    # Find first fraud and non-fraud data
    first_fraud = data[data['fraud'] == 1].iloc[[0]]
    first_non_fraud = data[data['fraud'] == 0].iloc[[0]]
    
    # Concatenate and drop them from the dataset
    separated_samples = pd.concat([first_fraud, first_non_fraud])
    data = data.drop(separated_samples.index)
    
    print("Removed first fraud and non-fraud samples:\n", separated_samples)
    return data

def preprocess_data(data, mappings_path):
    data = data.drop(['zipMerchant', 'zipcodeOri', 'customer'], axis=1)
    
    # Load mappings from JSON file
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)

    category_mapping = mappings['category_mapping']
    merchant_mapping = mappings['merchant_mapping']
    gender_mapping = mappings['gender_mapping']
    
    # Clean and convert data
    data['category'] = data['category'].str.replace("'", "", regex=False)
    data['merchant'] = data['merchant'].str.replace("'", "", regex=False)
    data['gender'] = data['gender'].str.replace("'", "", regex=False)
    data['age'] = data['age'].str.replace("'", "", regex=False)
    data['age'] = data['age'].str.replace("U", "7", regex=False)
    data['age'] = data['age'].astype('Int64')
    data['amount'] = np.ceil(data['amount']).astype('int64')

    # Apply mappings
    data['category'] = data['category'].map(category_mapping)
    data['merchant'] = data['merchant'].map(merchant_mapping)
    data['gender'] = data['gender'].map(gender_mapping)
    
    return data

def balance_data(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    y_res = pd.DataFrame(y_res)

    # Convert columns back to int64 after SMOTE
    X_res = X_res.round().astype('int64')
    y_res = y_res.astype('int64')
    
    print("Balanced dataset:\n", y_res['fraud'].value_counts())
    return X_res, y_res


def train_models(X_train, y_train):

     # Save the list of feature names used for training
    feature_names = X_train.columns.tolist()
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
        
    # Train KNN
    print("Training KNN model...\n")
    knn = KNeighborsClassifier(n_neighbors=5, p=1)
    knn.fit(X_train, y_train)
    print("KNN model trained.\n")

    # Train Random Forest
    print("Training RandomForest model...\n")
    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42,
        verbose=1, class_weight="balanced"
    )
    rf_clf.fit(X_train, y_train)
    print("RandomForest model trained.\n")

    # Train XGBoost
    print("Training XGBoost model...\n")
    XGBoost_CLF = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=400,
        objective="binary:hinge", booster='gbtree', n_jobs=-1,
        gamma=0, min_child_weight=1, max_delta_step=0,
        subsample=1, colsample_bytree=1, colsample_bylevel=1,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
        base_score=0.5, random_state=42, verbosity=1  
    )
    XGBoost_CLF.fit(X_train, y_train)
    print("XGBoost model trained.\n")

    # Train Ensemble Model
    print("Training Ensemble model...\n")
    ensemble_model = VotingClassifier(
        estimators=[('knn', knn), ('rf', rf_clf), ('xgb', XGBoost_CLF)],
        voting='soft'
    )
    ensemble_model.fit(X_train, y_train)
    print("Ensemble model trained.\n")
    
    return knn, rf_clf, XGBoost_CLF, ensemble_model

def save_models(knn, rf_clf, XGBoost_CLF, ensemble_model, model_dir='models/'):
    joblib.dump(knn, f'{model_dir}knn_model.pkl')
    joblib.dump(rf_clf, f'{model_dir}rf_model.pkl')
    joblib.dump(XGBoost_CLF, f'{model_dir}xgb_model.pkl')
    joblib.dump(ensemble_model, f'{model_dir}ensemble_model.pkl')
    print("Models have been trained and saved successfully.")

def main():
    # Step 1: Load the dataset
    data = load_data("data/bs140513_032310.csv")

    # Step 2: Remove first fraud and non-fraud samples
    data = remove_first_samples(data)
    
    # Step 3: Preprocess data
    data = preprocess_data(data, 'json/mappings.json')
    
    # Step 4: Split into features and target
    X = data.drop(['fraud'], axis=1).astype('float64')
    y = data['fraud']
    
    # Step 5: Balance the dataset using SMOTE
    X_res, y_res = balance_data(X, y)
    
    # Step 6: Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42, shuffle=True, stratify=y_res
    )
    
    # Step 7: Train the models
    knn, rf_clf, XGBoost_CLF, ensemble_model = train_models(X_train, y_train)
    
    # Step 8: Save the models
    save_models(knn, rf_clf, XGBoost_CLF, ensemble_model)

if __name__ == '__main__':
    main()
