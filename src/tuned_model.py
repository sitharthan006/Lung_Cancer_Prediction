import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')



# Load processed data
data = pd.read_csv(
    "data/processed_data/model_data.csv"
)
# Load best parameters
with open('../best_params.json', 'r') as file:
    best_params = json.load(file)



# Sample for evaluation
SAMPLE_SIZE = 50000
data_sampled = data.sample(n=SAMPLE_SIZE, random_state=42, replace=False)


# Encode categorical features
le = LabelEncoder()
categorical_cols = data_sampled.select_dtypes(include='object').columns
for col in categorical_cols:
    data_sampled[col] = le.fit_transform(data_sampled[col])

# Split features and target
X = data_sampled.drop('Final_Prediction', axis=1)
y = data_sampled['Final_Prediction']

# Stratified KFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Tuned Models
tuned_models = {
    'DecisionTree': DecisionTreeClassifier(**best_params['DecisionTree'], random_state=42),
    'RandomForest': RandomForestClassifier(**best_params['RandomForest'], random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(**best_params['GradientBoosting'], random_state=42),
    'XGBoosting': XGBClassifier(**best_params['XGBoosting'], random_state=42, verbosity=0,
                                 use_label_encoder=False, eval_metric='logloss',
                                 device='cuda', tree_method='hist')
}

# Evaluate tuned models
results = {}

for name, model in tuned_models.items():
    print(f"Evaluating {name}...", end=" ", flush=True)

    cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy').mean()
    cv_precision = cross_val_score(model, X, y, cv=skf, scoring='precision_weighted').mean()
    cv_recall = cross_val_score(model, X, y, cv=skf, scoring='recall_weighted').mean()
    cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted').mean()

    results[name] = {
        'Accuracy': cv_accuracy,
        'Precision': cv_precision,
        'Recall': cv_recall,
        'F1': cv_f1
    }



# Train on full sampled data and save tuned models
os.makedirs('models/tuned_models', exist_ok=True)

for name, model in tuned_models.items():
    model.fit(X, y)
    with open(f'models/tuned_models/{name}.pkl', 'wb') as file:
        pickle.dump(model, file)


# Best model
best_model = max(results, key=lambda k: results[k]['F1'])
print(f"\n[Best Tuned Model] Best Tuned Model: {best_model}")
print(f"   Accuracy:  {results[best_model]['Accuracy']:.6f}")
print(f"   Precision: {results[best_model]['Precision']:.6f}")
print(f"   Recall:    {results[best_model]['Recall']:.6f}")
print(f"   F1-Score:  {results[best_model]['F1']:.6f}")
