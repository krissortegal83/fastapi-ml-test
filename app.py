# app.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

app = FastAPI(title="Mini API Predict")

# Modelo de juguete al iniciar
X, y = make_classification(
    n_samples=600, n_features=6, n_informative=4, n_redundant=0,
    random_state=7, class_sep=1.1
)
model = LogisticRegression(max_iter=200).fit(X, y)

class Features(BaseModel):
    # exactamente 6 floats
    values: List[float] = Field(..., min_length=6, max_length=6)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Features):
    arr = np.array(payload.values).reshape(1, -1)
    pred = int(model.predict(arr)[0])
    proba = float(model.predict_proba(arr)[0][pred])
    return {"prediction": pred, "confidence": round(proba, 4)}


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hola, FastAPI ya está funcionando"}

@app.post("/predict")
def predict(data: Features):
    # simulamos un modelo: sumamos los valores
    resultado = sum(data.values)
    return {"input": data.values, "prediction": resultado}