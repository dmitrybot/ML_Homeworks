from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

import pandas as pd
import joblib
import re
import numpy as np

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

model = joblib.load('model.pkl')
scaler = joblib.load("scaler.pkl")

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    processed_data = preprocess_item(item)
    prediction = model.predict(processed_data)
    return float(prediction[0])
    
@app.post("/predict_items")
def predict_items_csv(file: UploadFile = File(...)) -> str:
    df = pd.read_csv(file.file)
    items = [Item(**row) for _, row in df.iterrows()]
    predictions = predict_items(items)
    
    df['predicted_price'] = predictions
    df.to_csv('predicted_results.csv', index=False)
    return 'predicted_results.csv'

def predict_items(items: List[Item]) -> List[float]:
    processed_data = pd.concat([preprocess_item(item) for item in items], axis=0)
    return model.predict(processed_data)

def preprocess_item(item: Item) -> pd.DataFrame:
    df = pd.DataFrame([item.dict()])

    df['mileage'] = df['mileage'].apply(extract_number)
    df['engine'] = df['engine'].apply(extract_number)
    df['max_power'] = df['max_power'].apply(extract_number)
    df = df.drop('torque', axis=1)

    for column in df.columns[df.isna().any()]:
        df[column].fillna(df[column].median(), inplace=True)

    df[['mileage', 'engine', 'max_power']] = df[['mileage', 'engine', 'max_power']].astype(float)
    df[['engine', 'seats']] = df[['engine', 'seats']].astype(int)

    df = df.drop('selling_price', axis=1)

    df_scaled = scaler.transform(df.select_dtypes(include=['int', 'float']))
    return df_scaled

def extract_number(value):
    if pd.isnull(value):
        return np.nan
    match = re.search(r'[\d.]+', str(value))
    if match:
        return float(match.group(0))
    return np.nan


{
    "name": "Tata Indica Vista Aura 1.2 Safire BSIV",
    "year": 2011,
    "selling_price": 130000,
    "km_driven": 70000,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "Second Owner",
    "mileage": "16.5 kmpl",
    "engine": "1172 CC",
    "max_power": "65 bhp",
    "torque": "96  Nm at 3000  rpm ",
    "seats": 5.0
}