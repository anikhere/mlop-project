import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

data = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
x,y = data.data, data.target
model = Lasso()
model.fit(x_train, y_train)
model.predict(x_test)
with open('model.pkl','wb') as f:
    pickle.dump(model,f)
print(f'the model dump is done')