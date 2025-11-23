import os
import pickle
import yaml
import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from mlflow.models import infer_signature
from urllib.parse import urlparse


# MLflow authentication
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/tahaanik729/new_pipel.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "tahaanik729"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "110eb2aa6bba9ba713078a897e0c8d82a7db7f04"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("new_pipel")


def hyper(x_train, y_train, param_grid):
    """Run hyperparameter tuning."""
    rf = RandomForestClassifier()
    grid = GridSearchCV(rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid.fit(x_train, y_train)
    return grid


# Load params
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)['train']


def train(data_path, model, random, n, depth):
    """Main training function."""
    
    data = pd.read_csv(data_path)
    x = data.drop(columns=['Outcome'])
    y = data['Outcome']

    with mlflow.start_run():

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=random
        )

        signature = infer_signature(x_train, y_train)

        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        # Run hyperparameter tuning
        grid_search = hyper(x_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        # Evaluate
        y_pred = best_model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        # Log to MLflow
        mlflow.log_metric("accuracy", acc)

        mlflow.log_param('best_n_estimators', best_model.n_estimators)
        mlflow.log_param('max_depth', best_model.max_depth)
        mlflow.log_param('min_samples_split', best_model.min_samples_split)
        mlflow.log_param('min_samples_leaf', best_model.min_samples_leaf)

        cm = confusion_matrix(y_test, y_pred)
        clr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), 'confusion_matrix.txt')
        mlflow.log_text(clr, 'classification_report.txt')

        # Track model
        tracking_url = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url != 'file':
            mlflow.sklearn.log_model(best_model, 'model', registered_model_name='Best Model')
        else:
            mlflow.sklearn.log_model(best_model, 'model', signature=signature)

        # Save pickle
        os.makedirs(os.path.dirname(model), exist_ok=True)
        pickle.dump(best_model, open(model, 'wb'))

        print(f"Model saved to {model}")


if __name__ == '__main__':
    train(
        params['data'],
        params['model'],
        params['random_state'],
        params['n_estimator'],
        params['max_depth']
    )
