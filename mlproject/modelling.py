
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setup MLflow dengan autologging
mlflow.set_experiment("Diabetes_Prediction_CI")
mlflow.autolog()

def load_preprocessed_data():
    """Load data yang sudah dipreprocess"""
    print("=" * 70)
    print("LOADING PREPROCESSED DATA")
    print("=" * 70)
    
    # Load preprocessed data
    df = pd.read_csv('diabetes_preprocessing/preprocessed_diabetes.csv')
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model dan return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

def train_model(model_name, model, X_train, X_test, y_train, y_test):
    """Train model dengan MLflow autologging"""
    
    with mlflow.start_run(run_name=model_name):
        print(f"\n{'='*70}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*70}")
        
        # Train model - autolog akan otomatis mencatat parameters, metrics, dan model
        model.fit(X_train, y_train)
        print(f"✓ Model trained successfully")
        
        # Evaluate untuk display (autolog sudah mencatat metrics)
        metrics = evaluate_model(model, X_test, y_test)
        
        print(f"  Model Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        print(f"✓ MLflow autologging completed for {model_name}")
        
        return metrics

def main():
    """Main function untuk training models"""
    
    print("\n" + "="*70)
    print("DIABETES PREDICTION - MODEL TRAINING (CI/CD)")
    print("MLflow Tracking: AUTOLOGGING")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Define models
    models = {
        'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision_Tree': DecisionTreeClassifier(random_state=42),
        'Random_Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    # Train all models
    results = {}
    for model_name, model in models.items():
        metrics = train_model(model_name, model, X_train, X_test, y_train, y_test)
        results[model_name] = metrics
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    # Best model
    best_model = results_df['accuracy'].idxmax()
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Accuracy: {results_df.loc[best_model, 'accuracy']:.4f}")
    
    print("\n✅ CI/CD Training completed successfully with MLflow autologging!")

if __name__ == "__main__":
    main()