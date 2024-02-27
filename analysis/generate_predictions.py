import models
import utils
import data_processing
from constants import (
    VALUE_COLUMN,
    UNIT_RATE_COLUMN,
    QUANTITY_COLUMN,
    GROSS_WEIGHT_COLUMN,
    SHIP_COUNT_COLUMN,
    PORT_COUNT_COLUMN,
    FILL_METHOD,
    TARGET_COLUMN,
    ARIMA_RESIDUAL_COLUMN
)
import pandas as pd
import constants
import sys
from datetime import datetime
from build_data import get_data
from sklearn.model_selection import train_test_split
import argparse

feature_choices = ["VOLZA", "OIL", 'PRICE', 'AIS', 'ARIMA']

parser = argparse.ArgumentParser(
    prog="Generate Predictions",
)

parser.add_argument("commodity")
parser.add_argument(
    "-f", "--features", nargs="*", choices= feature_choices
)  
parser.add_argument("-c", "--centre", action="store_true")  # option that takes a value

args = parser.parse_args()
# print(args)
COMMODITY = args.commodity
VOLZA_FILE_PATH = f"volza/{COMMODITY}/{COMMODITY}.csv"
PRICE_FILE_PATH = f"volza/{COMMODITY}/{COMMODITY}_prices.csv"
PETROL_FILE_PATH = 'volza/petroleum/petrol_crude_oil_spot_price.csv'
AIS_POPULAR_FILE_PATH = f'ais/ais_ml_features.csv' 

feature_map = {
    'VOLZA': [VALUE_COLUMN, UNIT_RATE_COLUMN, QUANTITY_COLUMN, GROSS_WEIGHT_COLUMN],
    'OIL': [constants.BRENT_OIL_COLUMN, constants.WTI_OIL_COLUMN],
    'PRICE': [TARGET_COLUMN],
    'AIS': [SHIP_COUNT_COLUMN, PORT_COUNT_COLUMN],
    'ARIMA': [ARIMA_RESIDUAL_COLUMN]
}

features = []
for feature in args.features:
    features.extend(feature_map[feature])
    
SPIKE_WINDOW_SIZES = [10, 20, 30, 40]
results_dfs = []
NAME_SPACE = '_'.join(args.features)
print(NAME_SPACE)
for window_size in SPIKE_WINDOW_SIZES:
    SPIKES_WINDOW_SIZE = window_size
    aggregated_df = get_data(VOLZA_FILE_PATH, PRICE_FILE_PATH, AIS_POPULAR_FILE_PATH, PETROL_FILE_PATH, SPIKES_WINDOW_SIZE, args.centre)
    X, y = data_processing.prepare_features_and_target(aggregated_df, features, 'spikes')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_test = data_processing.scale_features(X_train, X_test)

    X_train, y_train = data_processing.create_sequences(X_train, y_train, SPIKES_WINDOW_SIZE)
    X_test, y_test = data_processing.create_sequences(X_test, y_test, SPIKES_WINDOW_SIZE)

    print(f'X train shape: {X_train.shape}')

    output_file_path = f'{COMMODITY}/{NAME_SPACE}/results_{window_size}.csv'
    pred_file_path = f'{COMMODITY}/{NAME_SPACE}/predictions/{window_size}'
    print(pred_file_path)
    results_df = models.evaluate_all(X_train, y_train, X_test, y_test, output_file_path, pred_file_path)
    results_df['Window Size'] = window_size
    results_dfs.append(results_df)
