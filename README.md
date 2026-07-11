# 🫁 Lung Cancer Prediction Project

This is a comprehensive machine learning classification pipeline designed to predict the likelihood of lung cancer in patients using various demographic, behavioral, and clinical features.

It is structured and styled matching the House Price prediction project patterns.

## 📁 Project Structure

```
disease_prediction/
│
├── app/
│   ├── streamlit_app.py              # Streamlit UI for predictions
│   └── ui_model_training.py          # Train models for UI (simplified features)
│
├── data/
│   ├── processed_data/
│   │   └── model_data.pkl            # Preprocessed data pickle
│   └── raw_data/
│       └── lung_cancer_prediction.csv # Original dataset
│
├── images/
│   ├── eda_images/                   # 21 EDA visualization plots
│   └── ui_images/                    # UI screenshots and visual output
│
├── models/
│   ├── full_data_models/             # Classifiers trained on 50k sampled dataset
│   ├── tuned_models/                 # Hyperparameter-tuned models
│   └── ui_models/                    # Simplified models optimized for UI inputs
│
├── src/
│   ├── datapreprocessing.py          # Data cleaning & feature engineering
│   ├── EDA.py                        # Exploratory Data Analysis & plots generator
│   ├── model_training.py             # Stratified CV evaluate of 7 classifiers
│   ├── hyperparameter_tuning.py      # GridSearchCV hyperparameter searches
│   └── tuned_model.py                # Re-evaluating optimal tuned models
│
├── best_params.json                  # Tuned hyperparameters
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Git exclusion rules
```

## 🔧 Machine Learning Classifiers

The project trains and compares the following classification models:
1. **Logistic Regression**
2. **Ridge Classifier**
3. **Decision Tree Classifier**
4. **Random Forest Classifier**
5. **Gradient Boosting Classifier**
6. **XGBoost Classifier**
7. **Linear Support Vector Classifier (LinearSVC)**

Evaluation metrics used: **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess the Data
```bash
cd src
python datapreprocessing.py
```

### 3. Generate Exploratory Data Analysis Plots
```bash
python EDA.py
```

### 4. Train and Compare Models
```bash
python model_training.py
```

### 5. Tune Hyperparameters
```bash
python hyperparameter_tuning.py
```

### 6. Evaluate and Train Tuned Models
```bash
python tuned_model.py
```

### 7. Setup UI and Launch Web Dashboard
```bash
cd ../app
python ui_model_training.py
streamlit run streamlit_app.py
```
