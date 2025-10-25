import pickle
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

data = load_diabetes()
x,y = data.data, data.target
model = LinearRegression()
model.fit(x, y)
with open('model.pkl','wb') as f:
    pickle.dump(model,f)
print(f'the model dump is done')