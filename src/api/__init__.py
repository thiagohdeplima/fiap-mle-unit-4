import os
import pickle
import tensorflow as tf

import pandas as pd
import numpy as np
import yfinance as yf

import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Share API", description="Allow the management of shares", version="0.1.0")

@app.get("/shares/{symbol}", tags=["shares"])
async def get_share(symbol: str):
  """
  Retrieve share information for a given symbol.
  """

  if not os.path.exists(f"./models/{symbol}_model.h5"):
    return HTTPException(status_code=404, detail="model for this symbol not found")

  if not os.path.exists(f"./models/{symbol}_scaler.pkl"):
    return HTTPException(status_code=404, detail="scaler for this symbol not found")

  data = __get_data(symbol)
  model, scaler = __load_model_and_scaler(symbol)

  ###
  data_last = data["Close"].values.reshape(-1, 1)
  data_scaled = scaler.transform(data_last)
  X_input = np.expand_dims(data_scaled, axis=0)

  y_pred_scaled = model.predict(X_input)
  y_pred = scaler.inverse_transform(y_pred_scaled)
  price_pred = float(y_pred.flatten()[0])
  ###

  return {"symbol": symbol, "price": price_pred, "currency": "BRL"}

def main():
  uvicorn.run(app, host="0.0.0.0", port=8000)

def __get_data(symbol, lookback: int = 60) -> pd.DataFrame:
  lookback = 60

  if os.path.exists(f"./models/{symbol}_dataset.pkl"):
    lookback = min(lookback, len(pd.read_pickle(f"./models/{symbol}_dataset.pkl")))

  return yf.download(symbol, period="5y", interval="1d")[["Close"]].tail(lookback)

def __load_model_and_scaler(symbol: str):
  model_path = f"./models/{symbol}_model.h5"
  scaler_path = f"./models/{symbol}_scaler.pkl"

  if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Model or scaler for {symbol} not found.")

  model = tf.keras.models.load_model(model_path, compile=False)
  with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

  return model, scaler

