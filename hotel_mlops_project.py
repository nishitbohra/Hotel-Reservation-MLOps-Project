# hotel_mlops_project.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
import mlflow
import mlflow.sklearn
import joblib
import os
from datetime import datetime

# Set up MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Hotel_Reservation_Prediction"
mlflow.set_experiment(experiment_name)

# Create directories for saving models and visualizations
os.makedirs("models", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# Load the dataset
def load_data(filepath):
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded data with shape: {df.shape}")
    return df

# Exploratory Data Analysis
def perform_eda(df):
    print("Performing Exploratory Data Analysis...")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Target distribution
    plt.figure(figsize=(10, 6))
    # Using booking_status instead of is_canceled
    target_counts = df['booking_status'].value_counts()
    sns.barplot(x=target_counts.index, y=target_counts.values)
    plt.title('Distribution of Target Variable (booking_status)')
    plt.xlabel('Booking Status')
    plt.ylabel('Count')
    plt.savefig('visualizations/target_distribution.png')
    
    # Correlation heatmap for numerical features
    plt.figure(figsize=(14, 10))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    
    # Lead time vs. booking status
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='booking_status', y='lead_time', data=df)
    plt.title('Lead Time vs. Booking Status')
    plt.savefig('visualizations/lead_time_vs_booking.png')
    
    # Average price per room vs. booking status
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='booking_status', y='avg_price_per_room', data=df)
    plt.title('Average Price Per Room vs. Booking Status')
    plt.savefig('visualizations/price_vs_booking.png')
    
    # Market segment type vs. booking status
    plt.figure(figsize=(12, 6))
    market_booking = pd.crosstab(df['market_segment_type'], df['booking_status'], normalize='index')
    market_booking.plot(kind='bar', stacked=True)
    plt.title('Booking Status by Market Segment')
    plt.xlabel('Market Segment Type')
    plt.ylabel('Percentage')
    plt.savefig('visualizations/market_segment_booking.png')
    
    # Room type vs. booking status
    plt.figure(figsize=(12, 6))
    room_booking = pd.crosstab(df['room_type_reserved'], df['booking_status'], normalize='index')
    room_booking.plot(kind='bar', stacked=True)
    plt.title('Booking Status by Room Type')
    plt.xlabel('Room Type')
    plt.ylabel('Percentage')
    plt.savefig('visualizations/room_type_booking.png')
    
    return None

# Data preprocessing
def preprocess_data(df):
    print("Preprocessing data...")
    
    # Convert booking_status to binary (assuming 'Canceled' is the positive class)
    # Create a binary target variable for classification
    df['is_canceled'] = df['booking_status'].apply(lambda x: 1 if x == 'Canceled' else 0)
    
    # Identify feature types
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target variables from features
    if 'booking_status' in categorical_features:
        categorical_features.remove('booking_status')
    if 'is_canceled' in numerical_features:
        numerical_features.remove('is_canceled')
    
    # Define preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Prepare the data
    X = df.drop(['is_canceled', 'booking_status'], axis=1)
    y = df['is_canceled']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features

# Feature importance visualization
def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Rearrange feature names so they match the sorted feature importances
        names = [feature_names[i] for i in indices]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance for {model_name}')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), names, rotation=90)
        plt.tight_layout()
        plt.savefig(f'visualizations/feature_importance_{model_name}.png')
        
        return importances[indices], names
    return None, None

# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features):
    print("Training and evaluating models...")
    
    # Define models to train
    models = {
        'LogisticRegression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'DecisionTree': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ]),
        'RandomForest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'GradientBoosting': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    # Dictionary to store results
    results = {}
    
    # Train, evaluate and log each model
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            print(f"\nTraining {model_name}...")
            
            # Log model parameters
            for param_name, param_value in model.get_params().items():
                if param_name.startswith('classifier__'):
                    mlflow.log_param(param_name, param_value)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Save model
            joblib.dump(model, f"models/{model_name}_model.pkl")
            mlflow.sklearn.log_model(model, f"{model_name}_model")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_prob': y_prob
            }
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            cm_path = f'visualizations/confusion_matrix_{model_name}.png'
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            
            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_path = f'visualizations/roc_curve_{model_name}.png'
            plt.savefig(roc_path)
            mlflow.log_artifact(roc_path)
            
            # Get feature names for feature importance plots
            # This is tricky with Pipeline and ColumnTransformer
            try:
                # For models with feature_importances_ attribute
                if hasattr(model[-1], 'feature_importances_'):
                    # Try to get feature names from preprocessor
                    all_feature_names = []
                    for name, transformer, features in preprocessor.transformers_:
                        if name == 'cat' and hasattr(transformer[-1], 'get_feature_names_out'):
                            # Get categorical feature names after one-hot encoding
                            cat_features = transformer[-1].get_feature_names_out(features)
                            all_feature_names.extend(cat_features)
                        else:
                            # Add numerical feature names directly
                            all_feature_names.extend(features)
                    
                    # Plot feature importance
                    importances, names = plot_feature_importance(model[-1], all_feature_names, model_name)
                    
                    # Log feature importances
                    if importances is not None:
                        for idx, (importance, feature_name) in enumerate(zip(importances, names)):
                            mlflow.log_metric(f"importance_{idx}_{feature_name}", importance)
                        
                        mlflow.log_artifact(f'visualizations/feature_importance_{model_name}.png')
            except Exception as e:
                print(f"Could not plot feature importance for {model_name}: {e}")
    
    # Compare all models ROC curves
    plt.figure(figsize=(10, 8))
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.tight_layout()
    compare_roc_path = 'visualizations/roc_curves_comparison.png'
    plt.savefig(compare_roc_path)
    
    # Compare model metrics
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        values = [results[model][metric] for model in model_names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        
        bars = plt.bar(model_names, values, color=colors)
        plt.title(f'Model Comparison - {metric.capitalize()}')
        plt.ylabel(metric.capitalize())
        plt.ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'visualizations/model_comparison_{metric}.png')
    
    # Find best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = models[best_model_name]
    
    print(f"\nBest Model: {best_model_name} with F1 Score: {results[best_model_name]['f1']:.4f}")
    
    # Return best model for deployment
    return best_model, best_model_name

# Hyperparameter tuning for best model
def tune_best_model(X_train, X_test, y_train, y_test, best_model_name):
    print(f"\nTuning hyperparameters for {best_model_name}...")
    
    # Define parameter grid based on model
    if best_model_name == 'LogisticRegression':
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__penalty': ['l1', 'l2']
        }
    elif best_model_name == 'DecisionTree':
        param_grid = {
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'RandomForest':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
    elif best_model_name == 'GradientBoosting':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    elif best_model_name == 'XGBoost':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.3],
            'classifier__max_depth': [3, 5, 7]
        }
    else:
        print(f"No predefined parameter grid for {best_model_name}")
        return None
    
    # Create pipeline with the best model (reusing preprocessor)
    model = joblib.load(f"models/{best_model_name}_model.pkl")
    
    # Create GridSearchCV
    with mlflow.start_run(run_name=f"{best_model_name}_Tuning"):
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)
        
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Get best model
        best_tuned_model = grid_search.best_estimator_
        
        # Evaluate best model
        y_pred = best_tuned_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("tuned_accuracy", accuracy)
        mlflow.log_metric("tuned_precision", precision)
        mlflow.log_metric("tuned_recall", recall)
        mlflow.log_metric("tuned_f1_score", f1)
        
        print(f"Tuned {best_model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Save tuned model
        joblib.dump(best_tuned_model, f"models/{best_model_name}_tuned_model.pkl")
        mlflow.sklearn.log_model(best_tuned_model, f"{best_model_name}_tuned_model")
        
        return best_tuned_model

# Main function
def main(data_path):
    # Load data
    df = load_data(data_path)
    
    # Perform EDA
    perform_eda(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = preprocess_data(df)
    
    # Train and evaluate models
    best_model, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test, 
                                                          preprocessor, numerical_features, categorical_features)
    
    # Tune best model
    tuned_model = tune_best_model(X_train, X_test, y_train, y_test, best_model_name)
    
    print("\nMLOps workflow completed successfully!")
    print(f"Model artifacts saved in 'models/' directory")
    print(f"Visualizations saved in 'visualizations/' directory")
    print(f"MLflow tracking available at http://localhost:5000")

if __name__ == "__main__":
    # Replace with your dataset path
    data_path = "Hotel_Reservations.csv"
    main(data_path)