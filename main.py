from typing import Union
import uvicorn
from fastapi import FastAPI
import pickle
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score, mean_absolute_percentage_error, \
    mean_squared_error

app = FastAPI()

# load
model = xgb.XGBRegressor()
model.load_model('xgb_reg.txt')


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/predict")
def predict():
    # read dataset
    dataset = pd.read_csv("dataset_cleaned_V3.csv", index_col='DATETIME')

    # split data into X and y
    X = dataset.copy()
    y = X.pop('PWEA-T')

    # split data into train and test set with test size 0.2
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    predictions = model.predict(X)
    # metrics = {
        # 'MSE score': mean_squared_error(y_test, predictions),
        # 'R2 score': r2_score(y_test, predictions)
    # }
    return {
        "Predictions": predictions.tolist(),
        # "Metrics": metrics
    }


if __name__ == '__main__':
    uvicorn.run('main:app')
