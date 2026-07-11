import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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



# Sample for tuning
SAMPLE_SIZE = 30000
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

# Define parameter grids
param_grids = {
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 1.0]
        }
    },
    'XGBoosting': {
        'model': XGBClassifier(random_state=42, verbosity=0,
                               use_label_encoder=False, eval_metric='logloss',
                               device='cuda', tree_method='hist'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    }
}

# Hyperparameter Tuning
best_params = {}



for name, config in param_grids.items():
    print(f"\nTuning {name}...")
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=skf,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X, y)

    best_f1 = grid_search.best_score_
    best_params[name] = grid_search.best_params_





# Save best parameters
with open('best_params.json', 'w') as file:
    json.dump(best_params, file, indent=4)


