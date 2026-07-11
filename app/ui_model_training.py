import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LUNG CANCER PREDICTION — UI Model Training")
print("=" * 80)

# Load processed data
data = pd.read_csv(
    "data/processed_data/model_data.csv"
)

print(f"\nData loaded. Shape: {data.shape}")

# Select UI-friendly features
# We want features that a user can easily input in a form
ui_features = [
    'Age', 'Gender', 'Smoking_Status', 'Air_Pollution_Exposure', 
    'Family_History', 'Occupation_Exposure', 'Healthcare_Access', 
    'Stage_at_Diagnosis', 'Cancer_Type', 'Mortality_Risk', 
    '5_Year_Survival_Probability'
]

# Extract feature matrix X and target y from a 50k sample for fast UI model training
SAMPLE_SIZE = 50000
data_sampled = data.sample(n=SAMPLE_SIZE, random_state=42, replace=False)
X = data_sampled[ui_features]
y = data_sampled['Final_Prediction']

# Encode target y to 0 and 1
y_encoded = (y == 'Yes').astype(int)

# Encode categorical features and saveencoders for UI mapping
label_encoders = {}
categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"Encoded {col}: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

# Train UI models
models = {
    'LogisticRegression_ui': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree_ui': DecisionTreeClassifier(max_depth=10, random_state=42),
    'RandomForest_ui': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    'GradientBoosting_ui': GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42),
    'XGBoosting_ui': XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss', device='cuda', tree_method='hist')
}

# Save UI Models
os.makedirs('models/ui_models', exist_ok=True)

for name, model in models.items():
    print(f"Training {name}...", end=" ", flush=True)
    model.fit(X, y_encoded)
    with open(f'models/ui_models/{name}.pkl', 'wb') as file:
        pickle.dump(model, file)
   

# Save label encoders for UI use
with open('models/ui_models/ui_label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

# Save the feature columns list to verify order in prediction
with open('models/ui_models/ui_features.pkl', 'wb') as file:
    pickle.dump(ui_features, file)


