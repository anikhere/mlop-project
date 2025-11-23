import pandas as pd 
import numpy as np
import os 
import sys
import yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["preprocess"]
def preprocess(inp,out):
    df = pd.read_csv(inp,header=None)
    os.makedirs(os.path.dirname(out),exist_ok=True)
    df.to_csv(out,header=None,index=False)
    print(f'the data is stored to {out}path')
if __name__ == '__main__':
    preprocess(params['input'],params['output'])
