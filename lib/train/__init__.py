import os
import pickle

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error

def train(data: dict[str, pd.DataFrame], lookback=60) -> Sequential:
  model = __get_model(lookback)

  early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

  model.fit(
    data["X_train"], data["y_train"],
    epochs=50,
    batch_size=32,
    validation_data=(data["X_val"], data["y_val"]),
    callbacks=[early_stop],
    verbose=1
  )

  return model

def evaluate_model(model, data: dict[str, pd.DataFrame], scaler):
  pred = model.predict(data["X_test"])

  y_true = scaler.inverse_transform(data["y_test"].reshape(-1, 1)).flatten()
  y_pred = scaler.inverse_transform(pred).flatten()

  mae = mean_absolute_error(y_true, y_pred)
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

  return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def __get_model(lookback=60):
  model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
  ])

  model.compile(optimizer='adam', loss='mse')

  return model

def save_model(symbol, model, scaler, dataset):
  output_dir = "models"
  os.makedirs(output_dir, exist_ok=True)

  model_path = os.path.join(output_dir, f"{symbol}_model.h5")
  scaler_path = os.path.join(output_dir, f"{symbol}_scaler.pkl")
  dataset_path = os.path.join(output_dir, f"{symbol}_dataset.pkl")

  model.save(model_path)

  with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

  with open(dataset_path, "wb") as f:
    pickle.dump(dataset, f)

  return {"model_path": model_path, "scaler_path": scaler_path, "dataset_path": dataset_path}
