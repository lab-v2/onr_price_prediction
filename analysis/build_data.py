import utils
import pandas as pd
from datetime import datetime
from constants import (
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

def get_data(volza_file_path, price_file_path):
    # Formatting the date and price for Volza data
    volza_pd = pd.read_csv(volza_file_path)
    volza_pd = volza_pd[
        (volza_pd["Country of Origin"].notnull())
        & (volza_pd["Country of Destination"].notnull())
    ]
    volza_pd = volza_pd.rename(columns={"Unnamed: 0": "ID"})
    volza_pd["Date"] = volza_pd["Date"].apply(lambda x: x.split(" ")[0])
    volza_pd["Date"] = pd.to_datetime(volza_pd["Date"], errors="raise", format="%Y-%m-%d")
    volza_pd = utils.convert_to_kg(volza_pd)

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
    date_wise_volza = date_wise_volza.join(ais_popular_pd, how="left").fillna(
        method=FILL_METHOD
    )
    aggregated_df = date_wise_volza.join(prices_pd, how="left").fillna(method=FILL_METHOD)
    aggregated_df = aggregated_df.merge(
        brent_df[[DATE_COLUMN, BRENT_OIL_COLUMN]], on="Date", how="left"
    ).fillna(method=FILL_METHOD)
    aggregated_df = aggregated_df.merge(
        wti_df[[DATE_COLUMN, WTI_OIL_COLUMN]], on="Date", how="left"
    ).fillna(method=FILL_METHOD)

    # Add spikes column
    aggregated_df["spikes"] = utils.detect_spikes(aggregated_df, "Price")
    
    return aggregated_df