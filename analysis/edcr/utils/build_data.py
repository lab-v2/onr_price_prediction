import pandas as pd
from datetime import datetime
import pmdarima as pm
from .constants import (
    VALUE_COLUMN,
    QUANTITY_COLUMN,
    UNIT_RATE_COLUMN,
    PETROL_FILE_PATH,
    DATE_COLUMN,
    BRENT_OIL_COLUMN,
    WTI_OIL_COLUMN,
    FILL_METHOD,
    AIS_POPULAR_FILE_PATH
)

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

def get_data(volza_file_path, price_file_path, window_size,center):
    print("Building data...",)
    # Formatting the date and price for Volza data
    volza_pd = pd.read_csv(volza_file_path)
    volza_pd = volza_pd[
        (volza_pd["Country of Origin"].notnull())
        & (volza_pd["Country of Destination"].notnull())
    ]
    volza_pd = volza_pd.rename(columns={"Unnamed: 0": "ID"})
    volza_pd["Date"] = volza_pd["Date"].apply(lambda x: x.split(" ")[0])
    volza_pd["Date"] = pd.to_datetime(volza_pd["Date"], errors="raise", format="%Y-%m-%d")
    volza_pd = convert_to_kg(volza_pd)

    # Preprocessing the AIS data
    ais_popular_pd = pd.read_csv(AIS_POPULAR_FILE_PATH)
    ais_popular_pd["Date"] = pd.to_datetime(ais_popular_pd["Date"])

    # Preprocessing the price data
    prices_pd = pd.read_csv(price_file_path)
    prices_pd["Date"] = prices_pd["Date"].apply(
        lambda x: datetime.strptime(x, "%b %d, %Y").strftime("%Y-%m-%d")
    )
    prices_pd["Date"] = pd.to_datetime(prices_pd["Date"], errors="raise", format="%Y-%m-%d")
    
    #Check if Price column is a a string and convert to float
    if prices_pd["Price"].dtype == "O":
        prices_pd["Price"] = prices_pd["Price"].str.replace(",", "").astype(float)
    prices_pd = prices_pd[["Date", "Price"]]

    # Aggregate volza data by day
    date_wise_volza = volza_pd.groupby("Date")[
        [VALUE_COLUMN, QUANTITY_COLUMN, "Gross Weight"]
    ].sum()

    # Avg of Commodity Price in Volza
    avg_price_volza = volza_pd.groupby("Date")[UNIT_RATE_COLUMN].mean()
    date_wise_volza = date_wise_volza.join(avg_price_volza, how="left")

    # Petroleum data prep
    petrol_df = pd.read_csv(PETROL_FILE_PATH, delimiter=";", on_bad_lines="warn")
    petrol_df["Date"] = pd.to_datetime(petrol_df["Date"])

    # Split based on types of oil
    brent_df = petrol_df[petrol_df["product-name"] == "UK Brent Crude Oil"]
    wti_df = petrol_df[petrol_df["product-name"] == "WTI Crude Oil"]

    brent_df.rename(columns={"Value": "Brent Oil Value"}, inplace=True)
    wti_df.rename(columns={"Value": "WTI Oil Value"}, inplace=True)

    # Combining dataframes
    prices_pd = prices_pd.set_index("Date")
    ais_popular_pd = ais_popular_pd.set_index("Date")
    
    # date_wise_volza = date_wise_volza.join(ais_popular_pd, how="left").fillna(
    #     method=FILL_METHOD
    # )
    aggregated_df = date_wise_volza.join(prices_pd, how="left").fillna(method=FILL_METHOD)
    aggregated_df = aggregated_df.merge(
        brent_df[[DATE_COLUMN, BRENT_OIL_COLUMN]], on="Date", how="left"
    ).fillna(method=FILL_METHOD)
    aggregated_df = aggregated_df.merge(
        wti_df[[DATE_COLUMN, WTI_OIL_COLUMN]], on="Date", how="left"
    ).fillna(method=FILL_METHOD)

    # Clean up before passing to Arima
    initial_row_count = aggregated_df.shape[0]
    columns_of_interest = ['Price']  # Add other columns as necessary
    aggregated_df = aggregated_df.dropna(subset=columns_of_interest)
    rows_dropped = initial_row_count - aggregated_df.shape[0]
    print(f"Rows dropped due to NaN values: {rows_dropped}")

    train_size = int(len(aggregated_df) * 0.8)
    train_df = aggregated_df[:train_size]
    test_df = aggregated_df[train_size:]

    # Fit an Auto ARIMA model to the 'Price' series of the training data
    model = pm.auto_arima(train_df['Price'], seasonal=True, m=12, suppress_warnings=True, stepwise=True, error_action='ignore')

    # Forecast the training series using the model (in-sample prediction) to calculate training residuals
    train_forecast = model.predict_in_sample()
    train_residuals = train_df['Price'] - train_forecast

    # For the test set, use the model to forecast test-sample and calculate residuals
    test_forecast = model.predict(n_periods=len(test_df))
    test_residuals = test_df['Price'] - test_forecast

    # Append residuals to the respective DataFrame as a new feature for anomaly detection
    train_df = train_df.copy()
    train_df['ARIMA_Residuals'] = train_residuals

    test_df = test_df.copy()
    test_df['ARIMA_Residuals'] = test_residuals

    aggregated_df = pd.concat([train_df, test_df])

    # Add Isolation Forest spikes column
    # aggregated_df['spikes_if'] = utils.detect_spikes_if(aggregated_df, ['Price'], contamination=0.05)

    # Add rolling mean spikes column
    # aggregated_df["spikes"] = utils.detect_spikes(aggregated_df, "Price", window_size=window_size, center=center)
    # aggregated_df["spikes"] = utils.detect_spikes(aggregated_df, "Price")

    # aggregated_df["spikes"] = utils.detect_spikes_iqr(aggregated_df, "Price")

    
    return aggregated_df