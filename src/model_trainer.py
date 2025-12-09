import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.compose import ColumnTransformer

class ModelTrainer:
    """
    Manages the training, prediction, and evaluation of all required models
    (Regression for Severity and Classification for Frequency).
    """
    
    def __init__(self, preprocessor: ColumnTransformer):
        self.preprocessor = preprocessor
        self.models = {}

    def train_regression_models(self, X_train, y_train):
        """Trains Linear Regression, Random Forest, and XGBoost models for Claim Severity."""
        
        model_defs = {
            'Linear_Regression': LinearRegression(),
            'RandomForest_Regressor': RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=10),
            'XGBoost_Regressor': XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')
        }
        
        for name, model in model_defs.items():
            # Build the pipeline: Preprocessor -> Model
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('model', model)])
            pipeline.fit(X_train, y_train)
            self.models[name] = pipeline
        
        print("Regression Models Trained:", list(self.models.keys()))

    def evaluate_regression(self, X_test, y_test_orig):
        """
        Evaluates regression models using RMSE and R^2. 
        Note: Predictions must be transformed back to the original scale (np.expm1) 
        before calculating RMSE/R^2.
        """
        
        metrics = []
        for name, model in self.models.items():
            # 1. Predict on log scale
            y_pred_log = model.predict(X_test)
            
            # 2. Transform prediction back to original scale (un-log)
            y_pred_orig = np.expm1(y_pred_log) 
            
            # 3. Calculate metrics on original scale
            mse = mean_squared_error(y_test_orig, y_pred_orig)
            rmse = np.sqrt(mse)            
            
            r2 = r2_score(y_test_orig, y_pred_orig)

            metrics.append({'Model': name, 'RMSE (Original Scale)': rmse, 'R-squared': r2})

        return pd.DataFrame(metrics).sort_values(by='RMSE (Original Scale)')

    def train_classification_models(self, X_train, y_train):
        """Trains Logistic Regression, Random Forest, and XGBoost models for Claim Frequency."""
        
        model_defs = {
            'Logistic_Regression': LogisticRegression(random_state=42, n_jobs=-1, solver='liblinear'),
            'RandomForest_Classifier': RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10),
            'XGBoost_Classifier': XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
        }
        
        for name, model in model_defs.items():
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('model', model)])
            pipeline.fit(X_train, y_train)
            self.models[name] = pipeline

        print("Classification Models Trained:", list(self.models.keys()))

    def evaluate_classification(self, X_test, y_test):
        """Evaluates classification models using AUC-ROC, Precision, and Recall."""
        
        metrics = []
        for name, model in self.models.items():
            # Use predict_proba for AUC-ROC calculation
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            # Use predict for Precision/Recall/F1 (binary threshold)
            y_pred = model.predict(X_test)
            
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            metrics.append({'Model': name, 'AUC-ROC': auc_roc, 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

        return pd.DataFrame(metrics).sort_values(by='AUC-ROC', ascending=False)