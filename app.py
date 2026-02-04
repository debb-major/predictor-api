import numpy as np

from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


app = FastAPI()

# Simple training (usually I'd load a saved model)
iris = load_iris()
model = LogisticRegression(max_iter=200).fit(iris.data, iris.target)

@app.get("/")
def read_root():
    return {"message": "Iris Prediction API is running!"}

@app.post("/predict")
def predict(sepal_l: float, sepal_w: float, petal_l: float, petal_w: float):
    features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    prediction = model.predict(features)
    return {"class_id": int(prediction[0]), "class_name": iris.target_names[prediction[0]]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    