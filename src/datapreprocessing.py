import pandas as pd
import numpy as np
import pickle
import os



# Load Data
data = pd.read_csv('data/raw_data/lung_cancer_prediction.csv')



# Check for null values
null_counts = data.isnull().sum()

# Drop columns with more than 30% missing values
null_pct = data.isnull().mean()
columns_to_drop = null_pct[null_pct > 0.30].index.tolist()

if columns_to_drop:
    data.drop(columns=columns_to_drop, inplace=True)
    


# Numerical columns
num_cols = data.select_dtypes(include=np.number).columns

# Categorical columns
cat_cols = data.select_dtypes(exclude=np.number).columns

# Fill numerical columns with median
for col in num_cols:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())

# Fill categorical columns with mode
for col in cat_cols:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].mode()[0])


# Exposure Count
exposure_cols = [
    'Second_Hand_Smoke',
    'Occupation_Exposure',
    'Indoor_Smoke_Exposure',
    'Tobacco_Marketing_Exposure',
    'Family_History'
]

data['Exposure_Count'] = data[exposure_cols].apply(
    lambda row: (row == 'Yes').sum(),
    axis=1
)

# Smoking Risk
smoking_map = {
    'Never Smoker': 0,
    'Former Smoker': 1,
    'Current Smoker': 2,
    'Passive Smoker': 1
}

data['Smoking_Risk'] = data['Smoking_Status'].map(smoking_map).fillna(0)

# Pollution Risk
pollution_map = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}

data['Pollution_Risk'] = data['Air_Pollution_Exposure'].map(pollution_map).fillna(0)

# Risk Score
data['Risk_Score'] = (
    data['Smoking_Risk'] * 2 +
    data['Pollution_Risk'] +
    data['Exposure_Count'] +
    data['Mortality_Risk'] * 5
)

# Healthcare Score
healthcare_map = {
    'Poor': 0,
    'Limited': 1,
    'Good': 2
}

data['Healthcare_Score'] = data['Healthcare_Access'].map(healthcare_map).fillna(1)

# Treatment Score
treatment_map = {
    'None': 0,
    'Partial': 1,
    'Full': 2
}

data['Treatment_Score'] = data['Treatment_Access'].map(treatment_map).fillna(1)

# Access Score
data['Access_Score'] = (
    data['Healthcare_Score'] +
    data['Treatment_Score'] +
    (data['Insurance_Coverage'] == 'Yes').astype(int) +
    (data['Screening_Availability'] == 'Yes').astype(int)
)

# Stage Numeric
stage_map = {
    'I': 1,
    'II': 2,
    'III': 3,
    'IV': 4
}

data['Stage_Numeric'] = data['Stage_at_Diagnosis'].map(stage_map).fillna(2)

# Drop Country if present
if 'Country' in data.columns:
    data.drop(columns=['Country'], inplace=True)

# Drop intermediate engineered columns
cols_to_drop = [
    'Smoking_Risk',
    'Pollution_Risk',
    'Healthcare_Score',
    'Treatment_Score'
]

data.drop(columns=cols_to_drop, inplace=True)


# Target Distribution

# ============================
# Save Processed Data
# ============================

os.makedirs("data/processed_data", exist_ok=True)
data.to_csv(
    "data/processed_data/model_data.csv",
    index=False
)
