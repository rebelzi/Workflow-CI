
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

# Setup MLflow
mlflow.set_experiment("Diabetes_Prediction_CI")

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
    """Train model dengan MLflow tracking"""
    
    with mlflow.start_run(run_name=model_name):
        print(f"\n{'='*70}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*70}")
        
        # Train model
        model.fit(X_train, y_train)
        print(f"✓ Model trained successfully")
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Create and log confusion matrix
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Diabetes', 'Diabetes'],
                    yticklabels=['Non-Diabetes', 'Diabetes'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        # Save confusion matrix
        cm_filename = f'confusion_matrix_{model_name}.png'
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)
        plt.close()
        
        print(f"✓ MLflow tracking completed for {model_name}")
        
        return metrics

def main():
    """Main function untuk training models"""
    
    print("\n" + "="*70)
    print("DIABETES PREDICTION - MODEL TRAINING (CI/CD)")
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
    
    print("\n✅ CI/CD Training completed successfully!")

if __name__ == "__main__":
    main()