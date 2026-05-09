import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Loan Approval Prediction\n",
            "This notebook covers the full end-to-end process of building a machine learning model for loan approval prediction."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Import Libraries"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix\n",
            "from imblearn.over_sampling import SMOTE\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Load Data"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = pd.read_csv('loan_prediction.csv')\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Exploratory Data Analysis & Preprocessing"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(df.isnull().sum())"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Drop Loan_ID\n",
            "if 'Loan_ID' in df.columns:\n",
            "    df = df.drop('Loan_ID', axis=1)\n",
            "\n",
            "# Impute missing values\n",
            "cat_cols_missing = ['Gender', 'Married', 'Dependents', 'Self_Employed']\n",
            "for col in cat_cols_missing:\n",
            "    df[col] = df[col].fillna(df[col].mode()[0])\n",
            "\n",
            "num_cols_missing = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']\n",
            "for col in num_cols_missing:\n",
            "    df[col] = df[col].fillna(df[col].median())"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Encoding Dependents (3+ to 3)\n",
            "df['Dependents'] = df['Dependents'].astype(str).str.replace('+', '').astype(int)\n",
            "\n",
            "# Encoding Categorical Variables\n",
            "cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']\n",
            "df = pd.get_dummies(df, columns=cat_cols, drop_first=True)\n",
            "\n",
            "# Target Encoding\n",
            "df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Train-Test Split & Scaling"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X = df.drop('Loan_Status', axis=1)\n",
            "y = df['Loan_Status']\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
            "\n",
            "scaler = StandardScaler()\n",
            "X_train_scaled = scaler.fit_transform(X_train)\n",
            "X_test_scaled = scaler.transform(X_test)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Handle Class Imbalance with SMOTE"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Before SMOTE:\")\n",
            "print(y_train.value_counts())\n",
            "\n",
            "smote = SMOTE(random_state=42)\n",
            "X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)\n",
            "\n",
            "print(\"\\nAfter SMOTE:\")\n",
            "print(y_train_resampled.value_counts())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Model Training and Comparison"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "models = {\n",
            "    'Logistic Regression': LogisticRegression(random_state=42),\n",
            "    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),\n",
            "    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)\n",
            "}\n",
            "\n",
            "results = []\n",
            "for name, model in models.items():\n",
            "    model.fit(X_train_resampled, y_train_resampled)\n",
            "    y_pred = model.predict(X_test_scaled)\n",
            "    y_prob = model.predict_proba(X_test_scaled)[:, 1]\n",
            "    \n",
            "    precision = precision_score(y_test, y_pred)\n",
            "    recall = recall_score(y_test, y_pred)\n",
            "    f1 = f1_score(y_test, y_pred)\n",
            "    roc_auc = roc_auc_score(y_test, y_prob)\n",
            "    \n",
            "    results.append({\n",
            "        'Model': name,\n",
            "        'Precision': precision,\n",
            "        'Recall': recall,\n",
            "        'F1-Score': f1,\n",
            "        'ROC-AUC': roc_auc\n",
            "    })\n",
            "\n",
            "results_df = pd.DataFrame(results)\n",
            "results_df"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 6))\n",
            "sns.barplot(data=results_df.melt(id_vars='Model'), x='Model', y='value', hue='variable')\n",
            "plt.title('Model Comparison Metrics')\n",
            "plt.ylim(0, 1)\n",
            "plt.legend(loc='lower right')\n",
            "plt.show()"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('loan_prediction_notebook.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully.")
