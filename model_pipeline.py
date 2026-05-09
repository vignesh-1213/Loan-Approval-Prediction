import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

def main():
    print("Loading data...")
    df = pd.read_csv('loan_prediction.csv')
    
    print("Initial class distribution:")
    print(df['Loan_Status'].value_counts())
    
    # 1. Data Preprocessing
    print("\nHandling missing values...")
    # Drop Loan_ID
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
        
    # Impute categorical with mode
    cat_cols_missing = ['Gender', 'Married', 'Dependents', 'Self_Employed']
    for col in cat_cols_missing:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # Impute numerical with median
    num_cols_missing = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in num_cols_missing:
        df[col] = df[col].fillna(df[col].median())
        
    print("Encoding categorical variables...")
    # Dependents has '3+', replace with 3
    df['Dependents'] = df['Dependents'].astype(str).str.replace('+', '').astype(int)
    
    # Encode categorical features
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Encode target
    df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})
    
    # Split data
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # 3. Handle class imbalance
    print("Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"Resampled class distribution:\n{y_train_resampled.value_counts()}")
    
    # 4. Compare models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
    }
    
    results = []
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
        print(f"\n--- {name} ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        # Save model
        joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")
        
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df.to_string())
    
    # Save processed data for Streamlit
    df.to_csv('processed_data.csv', index=False)
    
    # Save the feature names for inference
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print("\nPipeline completed successfully! Models and scaler saved in 'models/' directory.")

if __name__ == '__main__':
    main()
