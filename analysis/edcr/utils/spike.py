import pandas as pd

SPIKES_THRESHOLD = 2

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

# ============================= spike labelling shift by 1 ============================== #
def detect_spikes_shift(df, column, window_size, SPIKES_THRESHOLD=2, present=False):
    spikes = []
    
    # If not including the present, shift the column values down by 1
    if not present:
        shifted_column = df[column].shift(1)
    else:
        shifted_column = df[column]

    # Calculate rolling stats
    moving_avg = shifted_column.rolling(window=window_size, min_periods=1).mean()
    std_dev = shifted_column.rolling(window=window_size, min_periods=1).std()

    # Detect spikes
    for actual_value, mean, sd in zip(df[column], moving_avg, std_dev):
        if sd == 0:  
            spikes.append(0)
        else:
            spikes.append(int(abs(actual_value - mean) > SPIKES_THRESHOLD * sd))
    
    return pd.Series(spikes, index=df.index)