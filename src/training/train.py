import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report
)
from xgboost import XGBClassifier
import mlflow
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = '/Users/shefalisaini/my_git_work/mlops-predictive-maintenance'
DATA_PATH    = os.path.join(PROJECT_ROOT, 'data/raw/ai4i2020.csv')
MLFLOW_URI   = 'http://localhost:5001'
EXPERIMENT   = 'predictive-maintenance-xgboost'

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Failure rate: {df['Machine failure'].mean():.2%}")
    return df

def prepare_features(df):
    drop_cols = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['torque_speed'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    X = df.drop(columns=['Machine failure'])
    y = df['Machine failure']
    return X, y

def train(n_estimators=100, max_depth=6, learning_rate=0.1, scale_pos_weight=9):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    with mlflow.start_run():
        params = {
            'n_estimators':     n_estimators,
            'max_depth':        max_depth,
            'learning_rate':    learning_rate,
            'scale_pos_weight': scale_pos_weight,
            'test_size':        0.2,
            'random_state':     42
        }
        mlflow.log_params(params)

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_train_scaled, y_train)

        y_pred  = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            'f1':        round(f1_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred), 4),
            'recall':    round(recall_score(y_test, y_pred), 4),
            'auc_roc':   round(roc_auc_score(y_test, y_proba), 4)
        }
        mlflow.log_metrics(metrics)

        # Save model locally and log as artifact
        # os.makedirs(os.path.join(PROJECT_ROOT, 'models/experiments'), exist_ok=True)
        # model_path = os.path.join(PROJECT_ROOT,
        #     f'models/experiments/xgboost_{n_estimators}_{max_depth}.pkl')
        # joblib.dump(model, model_path)
        # mlflow.log_artifact(model_path)

        print("\n── Results ──────────────────────────────")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v}")
        print(classification_report(y_test, y_pred,
              target_names=['No Failure', 'Failure']))

        return metrics, model

if __name__ == '__main__':
    print("=" * 50)
    print("Run 1: Default parameters")
    print("=" * 50)
    train(n_estimators=100, max_depth=6, learning_rate=0.1)

    print("\n" + "=" * 50)
    print("Run 2: More estimators, deeper trees")
    print("=" * 50)
    train(n_estimators=200, max_depth=8, learning_rate=0.05)

    print("\n" + "=" * 50)
    print("Run 3: Fast learning rate")
    print("=" * 50)
    train(n_estimators=150, max_depth=5, learning_rate=0.2)
