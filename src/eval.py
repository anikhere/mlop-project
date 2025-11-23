
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import yaml
import mlflow
from mlflow.models import infer_signature

import mlflow
import pickle
import os 
from urllib.parse import urlparse
import os
import mlflow
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/tahaanik729/new_pipel.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "tahaanik729"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "110eb2aa6bba9ba713078a897e0c8d82a7db7f04"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("new_pipel")
params= yaml.safe_load(open('params.yaml'))['train']
def evaluate(data_path,model):
    data= pd.read_csv(data_path)
    x = data.drop(columns=['Outcome'])
    y = data['Outcome']
    mlflow.set_tracking_uri('https://dagshub.com/tahaanik729/new_pipel.mlflow')
    model= pickle.load(open(model,'rb'))
    with mlflow.start_run():
        y_pred = model.predict(x)
        acc = accuracy_score(y,y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("model", "RandomForestClassifier")
        
if __name__ == "__main__":
    evaluate(params['data'],params['model'])



