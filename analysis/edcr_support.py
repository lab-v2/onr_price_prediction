#!/usr/bin/env python
# coding: utf-8

# In[1]:



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
    VOLZA_COLUMNS,
    AIS_COLUMNS,
    OIL_COLUMNS,
    ARIMA_RESIDUAL_COLUMN,
    RANDOM_STATE
)
import pandas as pd
import constants
import sys
from datetime import datetime
from build_data import get_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

import matplotlib.pyplot as plt
import numpy as np


# In[2]:


COMMODITYS = ['cobalt', 'copper', 'germanium', 'magnesium']
#target_COMMODITY = "copper"
target_COMMODITY = "germanium"
WINDOW_SIZE = 20

pre_features = []
pre_labels = []
tar_features = []
tar_labels = []

for COMMODITY in COMMODITYS:
    VOLZA_FILE_PATH = f"../volza/{COMMODITY}/{COMMODITY}.csv"
    PRICE_FILE_PATH = f"../volza/{COMMODITY}/{COMMODITY}_prices.csv"
    
    # Get the data
    data = get_data(VOLZA_FILE_PATH, PRICE_FILE_PATH, window_size=WINDOW_SIZE, center=False)
    
    # Add Isolation Forest spikes column
    data['spikes_if'] = utils.detect_spikes_if(data, TARGET_COLUMN, contamination=0.1)
    
    # Add Bowen's spike detection
    _, features, labels = utils.detect_spikes_new(data, TARGET_COLUMN, window_size=WINDOW_SIZE, center=False)
    if COMMODITY == target_COMMODITY:
        tar_features.extend(features)
        tar_labels.extend(labels)
        continue
    pre_features.extend(features)
    pre_labels.extend(labels)




# In[3]:


#import matplotlib.pyplot as plt
#
#aggregated_df = data.copy()
#data.head(2)
#
#
## In[4]:
#
#
## Add Lagged Features
#aggregated_df = utils.add_lagged_features(aggregated_df, 'Price', n_lags=5)
#
## Add Moving Averages
#aggregated_df = utils.add_moving_averages(aggregated_df, 'Price', windows=[5, 10])
#
## Add Returns
#aggregated_df = utils.add_returns(aggregated_df, 'Price')
#
## Add Volatility
#aggregated_df = utils.add_volatility(aggregated_df, 'returns', windows=[7, 14])
#
#
## In[5]:
#
#
#import matplotlib.pyplot as plt
## Plotting the price data
#utils.plot_prices(aggregated_df,column='spikes_new')
#
#
## In[6]:
#
#
#PRICE_FEATURE_COLUMNS = ['ma_5', 'ma_10', 'returns']
#PRICE_LAG_FEATURE_COLUMNS = ['lag_1', 'lag_2', 'lag_3']
#FEATURE_COLUMNS = VOLZA_COLUMNS + OIL_COLUMNS + [ARIMA_RESIDUAL_COLUMN]  # Mix and match features here
#SPIKE_COLUMN = ['spikes_new']


# In[7]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

sampler = RandomOverSampler

# Prepare mixed data
#X_mix, y_mix = data_processing.prepare_features_and_target(aggregated_df, FEATURE_COLUMNS, SPIKE_COLUMN)
X_mix = np.array(pre_features)
y_mix = np.array(pre_labels)
print("mix labels:", pd.value_counts(y_mix))

# Split mixed data
X_train_mix, X_test_mix, y_train_mix, y_test_mix = train_test_split(X_mix, y_mix, test_size=0.3, random_state=RANDOM_STATE, shuffle=False)
X_train_mix, X_val_mix, y_train_mix, y_val_mix = train_test_split(X_train_mix, y_train_mix, test_size=(10/70), random_state=RANDOM_STATE, shuffle=False)  # Adjusting test_size to get ~10% of the original

# Balancing
X_train_mix, y_train_mix = sampler(random_state=RANDOM_STATE).fit_resample(X_train_mix, y_train_mix)

# Scaling
X_train_mix, X_test_mix, X_val_mix = data_processing.scale_features(X_train_mix, X_test_mix, X_val_mix)

## Sequence making
#X_train_mix, y_train_mix = data_processing.create_sequences(X_train_mix, y_train_mix, WINDOW_SIZE)
#X_test_mix, y_test_mix = data_processing.create_sequences(X_test_mix, y_test_mix, WINDOW_SIZE)
#X_val_mix, y_val_mix = data_processing.create_sequences(X_val_mix, y_val_mix, WINDOW_SIZE)

# Evaluate and create pre-trained model
output_file_path = f'{target_COMMODITY}_{WINDOW_SIZE}/test/results_test.csv'
pred_file_path = f'{target_COMMODITY}_{WINDOW_SIZE}/test/predictions/test'
model_path = f'{target_COMMODITY}_{WINDOW_SIZE}/best_model'
X_train_mix = np.expand_dims(X_train_mix, axis = 2)
X_test_mix = np.expand_dims(X_test_mix, axis = 2)
X_val_mix = np.expand_dims(X_val_mix, axis = 2)
print(pred_file_path)
print(f"train_data: {X_train_mix.shape}, {type(X_train_mix)}, {type(X_train_mix[0])}")
print(f"test_data: {X_test_mix.shape}")
print(f"valid_data: {X_val_mix.shape}")
print(f"train_label: {y_train_mix.shape}, {type(y_train_mix)}")
print(f"test_label: {y_test_mix.shape}")
print(f"valid_label: {y_val_mix.shape}")
#results_df  = models.evaluate_all(X_train_mix, y_train_mix, X_val_mix, y_val_mix, X_test_mix, y_test_mix, output_file_path, pred_file_path, model_path, False)

# In[8]:


# In[9]:


# Prepare price data
#X_price, y_price = data_processing.prepare_features_and_target(aggregated_df, TARGET_COLUMN, SPIKE_COLUMN)
X_price = np.array(tar_features)
y_price = np.array(tar_labels)

# Split price data
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X_price, y_price, test_size=0.3, random_state=RANDOM_STATE, shuffle=False)
X_train_price, X_val_price, y_train_price, y_val_price = train_test_split(X_train_price, y_train_price, test_size=(10/70), random_state=RANDOM_STATE, shuffle=False)

# Balancing
X_train_price, y_train_price = sampler(random_state=RANDOM_STATE).fit_resample(X_train_price, y_train_price)

# Scaling
X_train_price, X_test_price, X_val_price = data_processing.scale_features(X_train_price, X_test_price, X_val_price)

# Sequence making
#X_train_price, y_train_price = data_processing.create_sequences(X_train_price, y_train_price, WINDOW_SIZE)
#X_test_price, y_test_price = data_processing.create_sequences(X_test_price, y_test_price, WINDOW_SIZE)
#X_val_price, y_val_price = data_processing.create_sequences(X_val_price, y_val_price, WINDOW_SIZE)


X_train_price = np.expand_dims(X_train_price, axis = 2)
X_test_price = np.expand_dims(X_test_price, axis = 2)
X_val_price = np.expand_dims(X_val_price, axis = 2)
# In[10]:
# In[10]:
results_df  = models.evaluate_all(X_train_price, y_train_price, X_val_price, y_val_price, X_test_price, y_test_price, output_file_path, pred_file_path, model_path, False)


#from tensorflow.keras.models import load_model
#
#saved_model_path = f'{model_path}/{best_model_descriptor}.h5'
#model = load_model(saved_model_path)
## model.summary()
#
#
## In[11]:
#
#
#print(X_train_price.shape, X_val_price.shape, X_test_price.shape)
#print(X_train_mix.shape, X_val_mix.shape, X_test_mix.shape)
#
#
## In[12]:
#
#
#results_df = models.retrain_best_model(saved_model_path, X_train_price, y_train_price, X_val_price, y_val_price, X_test_price, y_test_price)
#
#
## In[13]:
#
#
#results_df


# In[ ]:




