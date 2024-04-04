from constants import VALUE_COLUMN, QUANTITY_COLUMN, UNIT_RATE_COLUMN, SPIKES_THRESHOLD
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from collections import deque

# Only keep rows where we have usable quantity units (kg, ton) and standardizing it.
def convert_to_kg(df, quantity_col='Std. Quantity', unit_col='Std. Unit'):
    converstion_factors = {
        'TON': 907.185,
        'TNE': 1000,
        'KGS': 1,
        'Kgs': 1,
    }

    df_filtered = df[df[unit_col].isin(converstion_factors.keys())]

    def convert(row):
        unit = row[unit_col]
        quantity = row[quantity_col]
        return quantity * converstion_factors.get(unit,1)
    
    df_filtered = df_filtered[df_filtered[VALUE_COLUMN] != 0]
    df_filtered[QUANTITY_COLUMN] = df_filtered.apply(convert, axis=1)
    df_filtered = df_filtered[df_filtered[QUANTITY_COLUMN] != 0]

    df_filtered[UNIT_RATE_COLUMN] = df_filtered[VALUE_COLUMN] / df_filtered[QUANTITY_COLUMN]

    return df_filtered

# Only keep rows where we have usable quantity units (kg, ton) and standardizing it. For VOLZA only.
def convert_to_pound(df, quantity_col='Std. Quantity', unit_col='Std. Unit'):
    converstion_factors = {
        'TON': 907.185 * 2.20462,  
        'TNE': 1000 * 2.20462,     
        'KGS': 2.20462,            
        'Kgs': 2.20462,
    }

    df_filtered = df[df[unit_col].isin(converstion_factors.keys())]

    def convert(row):
        unit = row[unit_col]
        quantity = row[quantity_col]
        return quantity * converstion_factors.get(unit,1)
    
    df_filtered = df_filtered[df_filtered[VALUE_COLUMN] != 0]
    df_filtered[QUANTITY_COLUMN] = df_filtered.apply(convert, axis=1)
    df_filtered = df_filtered[df_filtered[QUANTITY_COLUMN] != 0]

    df_filtered[UNIT_RATE_COLUMN] = df_filtered[VALUE_COLUMN] / df_filtered[QUANTITY_COLUMN]

    return df_filtered

def detect_spikes(df, column, window_size, center=False):
    ## Detecting spikes
    print("Detecting spikes...",window_size)
    moving_avg = df[column].rolling(window=window_size, center=center).mean()
    std_dev = df[column].rolling(window=window_size, center=center).std()

    # Set a threshold to identify spikes
    return (abs(df[column] - moving_avg) > SPIKES_THRESHOLD * std_dev).astype(int)

def detect_spikes_new(df, column, window_size, center=False):
    df = df.dropna()
    spikes = np.zeros(len(df), dtype=float)
    feature = []
    labels = []
    if isinstance(column, str):
        column = [column]
    
    for col in column:
        series = df[col].tolist()
        # Calculate rolling mean and std deviation for the current column
        rolling_mean = df[col].rolling(window=window_size, center=center).mean()
        rolling_std = df[col].rolling(window=window_size, center=center).std()
        
        # Calculate if a spike occurred based on threshold
        for i in range(window_size, len(df)):  # Start from window_size - 1 to have a full window
            if abs(df[col].iloc[i] - rolling_mean.iloc[i-1]) > SPIKES_THRESHOLD * rolling_std.iloc[i-1]:
                spikes[i] = 1
            feature.append(np.array(series[i - window_size : i]))
            labels.append(int(spikes[i]))

    spikes = spikes.astype(int)
    
    return spikes, feature, labels

def detect_spikes_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    
    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    spikes = outliers.astype(int)
    
    return spikes

def detect_spikes_cma(df, column):
    moving_avg = df[column].expanding(min_periods=1).mean()
    std_dev = df[column].expanding(min_periods=1).std()

    std_dev_filled = std_dev.fillna(1) 
    return (abs(df[column] - moving_avg) > SPIKES_THRESHOLD * std_dev_filled).astype(int)

def detect_spikes_if(df, columns, n_estimators=100, contamination=0.1):
    X = df[columns].values

    # Initialize and fit the Isolation Forest model
    clf = IsolationForest(n_estimators=n_estimators, contamination=contamination)
    clf.fit(X)

    # Predict anomalies (-1 for anomalies, 1 for normal)
    is_anomaly = clf.predict(X)

    # Convert anomaly labels to binary labels (1 for spike, 0 for no spike)
    spikes = pd.Series(is_anomaly == -1, index=df.index).astype(int)
    
    return spikes

# ================= Time Series related features =========================== (doesnt work too well)
def add_lagged_features(df, column, n_lags):
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df[column].shift(i)
    return df

def add_moving_averages(df, column, windows):
    for window in windows:
        df[f'ma_{window}'] = df[column].rolling(window=window).mean()
    return df

def add_returns(df, column):
    df['returns'] = df[column].pct_change()
    return df

def add_volatility(df, column, windows):
    for window in windows:
        df[f'volatility_{window}'] = df[column].rolling(window=window).std()
    return df

# =====================================================================

def plot_prices(df, column='spikes', title='Price Spike chart'):
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(df.index, df['Price'], label='Price', color='blue')

    # Highlighting spikes
    spike_indices = df[df[column] == 1].index
    spike_prices = df.loc[spike_indices, 'Price']
    plt.scatter(spike_indices, spike_prices, color='red', marker='^', label='Spikes')

    # Calculate the percentage of spikes
    total_spikes = len(spike_indices)
    total_data_points = len(df)
    spike_percentage = (total_spikes / total_data_points) * 100

    # Print or display the spike percentage
    print(f"Spike Percentage: {spike_percentage:.2f}%")

    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Data with Spikes')
    plt.legend()

    # Display the plot
    plt.show()


# ============================= spike labelling shift by 1 ============================== #
def detect_spikes_shift(df, column, window_size, SPIKES_THRESHOLD=2, present=False):
    spikes = []
    
    # If not including the present, shift the column values down by one
    if not present:
        shifted_column = df[column].shift(1)
    else:
        shifted_column = df[column]

    # Calculate rolling stats
    moving_avg = shifted_column.rolling(window=window_size, min_periods=1).mean()
    std_dev = shifted_column.rolling(window=window_size, min_periods=1).std()

    # Detect spikes
    for actual_value, mean, sd in zip(df[column], moving_avg, std_dev):
        if sd == 0:  # Avoid division by zero
            spikes.append(0)
        else:
            spikes.append(int(abs(actual_value - mean) > SPIKES_THRESHOLD * sd))
    
    return pd.Series(spikes, index=df.index)

