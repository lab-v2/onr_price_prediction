import models
import utils
from constants import TARGET_COLUMN
import sys
from datetime import datetime
from build_data import get_data
import pandas as pd
import pmdarima as pm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Commandline arg
COMMODITY = sys.argv[1:][0]

VOLZA_FILE_PATH = f"../volza/{COMMODITY}/{COMMODITY}.csv"
PRICE_FILE_PATH = f"../volza/{COMMODITY}/{COMMODITY}_prices.csv"

# Get the data
aggregated_df = get_data(VOLZA_FILE_PATH, PRICE_FILE_PATH)

# Clean up before passing to Arima
initial_row_count = aggregated_df.shape[0]

columns_of_interest = ['Price', 'spikes']  # Add other columns as necessary

aggregated_df = aggregated_df.dropna(subset=columns_of_interest)

rows_dropped = initial_row_count - aggregated_df.shape[0]

print(f"Rows dropped due to NaN values: {rows_dropped}")

# Fit an Auto ARIMA model to the 'Price' series
model = pm.auto_arima(aggregated_df['Price'], seasonal=True, m=12, suppress_warnings=True, stepwise=True, error_action='ignore')

# Forecast the series using the model (in-sample prediction)
forecast = model.predict_in_sample()

# Calculate residuals (difference between actual and forecasted values)
residuals = aggregated_df[TARGET_COLUMN] - forecast

# Append residuals to DataFrame as a new feature (using residuals as a way to detect spike / anomaly)
aggregated_df = aggregated_df.copy()
aggregated_df['ARIMA_Residuals'] = residuals

FEATURE_COLUMNS = [TARGET_COLUMN, 'ARIMA_Residuals']  # Add other feature columns as necessary
# FEATURE_COLUMNS = [TARGET_COLUMN]  # Add other feature columns as necessary
X = aggregated_df[FEATURE_COLUMNS].values
y = aggregated_df['spikes'].values

# Feature scaling using StandardScaler, redoing this so we scale train / test separately to avoid leakage
scaler = StandardScaler()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, shuffle=False)

X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.fit_transform(X_test_raw)

# Define the window size for creating sequences
SPIKES_WINDOW_SIZE = 20

# Function to create sequences from the scaled data
def create_sequences(X_scaled, y, window_size):
    X_sequences, y_sequences = [], []
    for i in range(len(X_scaled) - window_size + 1):
        X_sequences.append(X_scaled[i:i + window_size, :])
        y_sequences.append(y[i + window_size - 1])
    return np.array(X_sequences), np.array(y_sequences)

# Create sequences for training and test data
X_train, y_train = create_sequences(X_train_scaled, y_train, SPIKES_WINDOW_SIZE)
X_test, y_test = create_sequences(X_test_scaled, y_test, SPIKES_WINDOW_SIZE)

output_dicts = []

# Train and test the model

#LSTM Model
for layers in [250, 200, 100, 50]:
    y_pred, output_dict = models.evaluate_lstm(layers, X_train, y_train, X_test, y_test)
    output_dicts.append(output_dict)
    
#CNN w Attention Model
for filter in [32, 64, 128, 256]:
    for kernel in [7,5,3]:
        y_pred, output_dict = models.evaluate_attention_cnn(filter, kernel, X_train, y_train, X_test, y_test)
        output_dicts.append(output_dict)
        
#RNN Model
for units in [200,150,100,50]:
    y_pred, output_dict = models.evaluate_rnn(units,  X_train, y_train, X_test, y_test)
    output_dicts.append(output_dict)
    
#CNN Model
for filter in [32, 64, 128, 256]:
    for kernel in [7,5,3]:
        y_pred, output_dict = models.evaluate_cnn(filter, kernel,  X_train, y_train, X_test, y_test)
        output_dicts.append(output_dict)
        
output_dicts = pd.DataFrame(output_dicts)
output_dicts.to_csv("results.csv")