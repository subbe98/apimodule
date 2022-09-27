from typing import Union
import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/predict")
def predict():
    return {
        "Prediction": "Complete"
    }


if __name__ == '__main__':
    uvicorn.run('main:app')
