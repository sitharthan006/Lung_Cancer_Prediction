import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


# Load processed data
data = pd.read_csv(
    "data/processed_data/model_data.csv"
)


# Sample data for training (460K is too large for GridSearchCV later)
SAMPLE_SIZE = 50000
data_sampled = data.sample(n=SAMPLE_SIZE, random_state=42, replace=False)


# Encode categorical features
label_encoders = {}
categorical_cols = data_sampled.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    data_sampled[col] = le.fit_transform(data_sampled[col])
    label_encoders[col] = le

# Split features and target
X = data_sampled.drop('Final_Prediction', axis=1)
y = data_sampled['Final_Prediction']



# Models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RidgeClassifier': RidgeClassifier(alpha=1.0, random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoostClassifier': XGBClassifier(n_estimators=100, random_state=42, verbosity=0,
                                        use_label_encoder=False, eval_metric='logloss',
                                        device='cuda', tree_method='hist'),
    'LinearSVC': LinearSVC(max_iter=2000, random_state=42)
}

# Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in models.items():
    print(f"Training {name}...", end=" ", flush=True)

    # Cross-validated Accuracy
    cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy').mean()

    # Cross-validated Precision
    cv_precision = cross_val_score(model, X, y, cv=skf, scoring='precision_weighted').mean()

    # Cross-validated Recall
    cv_recall = cross_val_score(model, X, y, cv=skf, scoring='recall_weighted').mean()

    # Cross-validated F1
    cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted').mean()

    results[name] = {
        'Accuracy': cv_accuracy,
        'Precision': cv_precision,
        'Recall': cv_recall,
        'F1': cv_f1
    }



# Train on full sampled data and save models
os.makedirs('../models/full_data_models', exist_ok=True)

for name, model in models.items():
    model.fit(X, y)
    with open(f'../models/full_data_models/{name}Model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Save label encoders for later use
with open('../models/full_data_models/label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)


# Print best model
best_model = max(results, key=lambda k: results[k]['F1'])
print(f"\n[Best Model] Best Model: {best_model}")
print(f"   Accuracy:  {results[best_model]['Accuracy']:.6f}")
print(f"   Precision: {results[best_model]['Precision']:.6f}")
print(f"   Recall:    {results[best_model]['Recall']:.6f}")
print(f"   F1-Score:  {results[best_model]['F1']:.6f}")
