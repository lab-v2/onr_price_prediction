from constants import VALUE_COLUMN, QUANTITY_COLUMN, UNIT_RATE_COLUMN, SPIKES_THRESHOLD
import matplotlib.pyplot as plt
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

def detect_spikes(df, column, window_size):
    ## Detecting spikes
    print("Detecting spikes...",window_size)
    moving_avg = df[column].rolling(window=window_size, center=True).mean()
    std_dev = df[column].rolling(window=window_size, center=True).std()

    # Set a threshold to identify spikes
    return (abs(df[column] - moving_avg) > SPIKES_THRESHOLD * std_dev).astype(int)

def plot_prices(df, title):
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(df.index, df['Price'], label='Price', color='blue')

    # Highlighting spikes
    spike_indices = df[df['spikes'] == 1].index
    spike_prices = df.loc[spike_indices, 'Price']
    plt.scatter(spike_indices, spike_prices, color='red', marker='^', label='Spikes')

    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Data with Spikes')
    plt.legend()

    # Display the plot
    plt.show()