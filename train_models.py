import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import joblib
from imblearn.over_sampling import SMOTE
import json 
import numpy as np

# read the dataset
data_request = pd.read_csv("data/bs140513_032310.csv")


# find first fraud data 
first_fraud = data_request[data_request['fraud'] == 1].iloc[[0]]
# find first nonfraud data 
first_non_fraud = data_request[data_request['fraud'] == 0].iloc[[0]]
separated_samples = pd.concat([first_fraud, first_non_fraud])
# drop from dataset
data_request = data_request.drop(separated_samples.index)
print(separated_samples)



df = pd.DataFrame(data_request)
reduced_data = data_request.drop(['zipMerchant','zipcodeOri','customer'],axis=1)

# Load mappings from JSON file
with open('mappings.json', 'r') as f:
    mappings = json.load(f)

category_mapping = mappings['category_mapping']
merchant_mapping = mappings['merchant_mapping']
gender_mapping = mappings['gender_mapping']

# Remove extra quotes from DataFrame values
reduced_data['category'] = reduced_data['category'].str.replace("'", "", regex=False)
reduced_data['merchant'] = reduced_data['merchant'].str.replace("'", "", regex=False)
reduced_data['gender'] = reduced_data['gender'].str.replace("'", "", regex=False)
reduced_data['age'] = reduced_data['age'].str.replace("'", "", regex=False)
reduced_data['age'] = reduced_data['age'].str.replace("U", "7", regex=False)
reduced_data['age'] = reduced_data['age'].astype('Int64')
reduced_data['amount'] = np.ceil(reduced_data['amount']).astype('int64')


# Apply mappings
reduced_data['category'] = reduced_data['category'].map(category_mapping)
reduced_data['merchant'] = reduced_data['merchant'].map(merchant_mapping)
reduced_data['gender'] = reduced_data['gender'].map(gender_mapping)


X = reduced_data.drop(['fraud'], axis=1).astype('float64')
y = reduced_data['fraud']



sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
y_res = pd.DataFrame(y_res)
# Convert columns back to int64 after SMOTE
X_res = X_res.round().astype('int64')
y_res = y_res.astype('int64')
print(y_res['fraud'].value_counts())


# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.3,random_state=42,shuffle=True,stratify=y_res)

print("going to train knn model\n")
knn = KNeighborsClassifier(n_neighbors=5,p=1)
knn.fit(X_train, y_train)
print("knn model trained\n")

# Train Random Forest
# Initialize the Random Forest classifier with 200 estimators, a maximum depth of 8, and balanced class weights
print("going to train RandomForest model\n")
rf_clf = RandomForestClassifier(n_estimators=100 , max_depth=8, random_state=42,
verbose=1, class_weight="balanced")
rf_clf.fit(X_train, y_train)
print("RandomForest model trained\n")



# Train XGBoost
print("going to train XGB model\n")
XGBoost_CLF = xgb.XGBClassifier(
    max_depth=6, 
    learning_rate=0.05, 
    n_estimators=400, 
    objective="binary:hinge", 
    booster='gbtree', 
    n_jobs=-1, 
    gamma=0, 
    min_child_weight=1, 
    max_delta_step=0, 
    subsample=1, 
    colsample_bytree=1, 
    colsample_bylevel=1, 
    reg_alpha=0, 
    reg_lambda=1, 
    scale_pos_weight=1, 
    base_score=0.5, 
    random_state=42, 
    verbosity=1  
)
XGBoost_CLF.fit(X_train,y_train)
y_prediction = XGBoost_CLF.predict(X_test)
print("XGB model trained\n")

# Create an ensemble model
print("going to train ensemble model\n")
ensemble_model = VotingClassifier(estimators=[('knn', knn), ('rf', rf_clf), ('xgb', XGBoost_CLF)], voting='soft')
ensemble_model.fit(X_train, y_train)
print("ensemble model trained\n")

# Save models
joblib.dump(knn, 'models/knn_model.pkl')
joblib.dump(rf_clf, 'models/rf_model.pkl')
joblib.dump(XGBoost_CLF, 'models/xgb_model.pkl')
joblib.dump(ensemble_model, 'models/ensemble_model.pkl')

print("Models have been trained and saved successfully.")
