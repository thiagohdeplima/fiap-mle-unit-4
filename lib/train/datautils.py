from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import yfinance as yf

from sklearn.preprocessing import MinMaxScaler

class Data:
  def __init__(self, df: pd.DataFrame):
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.2)

    self.train = df.iloc[:train_size]
    self.val = df.iloc[train_size:train_size+val_size]
    self.test = df.iloc[train_size+val_size:]

  @staticmethod
  def from_ticker(ticker, **kwargs):
    start_date = kwargs.get('start_date')
    end_date = kwargs.get('end_date')

    if start_date is None:
      start_date = (datetime.now() - timedelta(365 * 5)).strftime('%Y-%m-%d')

    if end_date is None:
      end_date = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

    dataframe = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    dataframe = dataframe[['Close']]

    return Data(dataframe)


def normalize_and_split(data: Data, **kwargs):
  scaler = MinMaxScaler()
  lookback = kwargs.get('lookback', 60)

  train_scaled = scaler.fit_transform(data.train)
  val_scaled = scaler.transform(data.val)
  test_scaled = scaler.transform(data.test)

  lookback = 60
  X_train, y_train = create_sequences(train_scaled, lookback)
  X_val, y_val = create_sequences(val_scaled, lookback)
  X_test, y_test = create_sequences(test_scaled, lookback)

  X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
  X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
  X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

  return {
    "X_train": X_train,
    "y_train": y_train,
    "X_val": X_val,
    "y_val": y_val,
    "X_test": X_test,
    "y_test": y_test
  }, scaler


def create_sequences(data, lookback=60) -> tuple[np.ndarray, np.ndarray]:
  X, y = [], []

  for i in range(lookback, len(data)):
    X.append(data[i-lookback:i, 0])
    y.append(data[i, 0])

  return np.array(X), np.array(y)
