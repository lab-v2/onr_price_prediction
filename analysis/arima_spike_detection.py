import models
import utils
import data_processing
from constants import TARGET_COLUMN, RANDOM_STATE
import sys
from datetime import datetime
from build_data import get_data
import pmdarima as pm

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

# Define the window size for creating sequences
SPIKES_WINDOW_SIZE = 20

# Prepare features and target
FEATURE_COLUMNS = [TARGET_COLUMN, 'ARIMA_Residuals']  # Adjust as needed
X, y = data_processing.prepare_features_and_target(aggregated_df, FEATURE_COLUMNS, 'spikes')

# Split data (using sklearn's train_test_split)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=False)

# Scale features
X_train_scaled, X_test_scaled = data_processing.scale_features(X_train_raw, X_test_raw)

# Create sequences
X_train, y_train = data_processing.create_sequences(X_train_scaled, y_train, SPIKES_WINDOW_SIZE)
X_test, y_test = data_processing.create_sequences(X_test_scaled, y_test, SPIKES_WINDOW_SIZE)

# Evaluate all models & save in file
output_file_path = f'{COMMODITY}/results.csv'
models.evaluate_all(X_train, y_train, X_test, y_test, output_file_path)