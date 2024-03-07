#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np


# In[2]:


a = [x for x in range(100)]


# In[20]:


def detect_spikes_new(df, column, window_size, center=False):
    spikes = np.zeros(len(df), dtype=float)

    if isinstance(column, str):
        column = [column]
    
    features = []
    labels = []
    for col in column:
        series = df[col].tolist()
        # Calculate rolling mean and std deviation for the current column
        rolling_mean = df[col].rolling(window=window_size, center=center).mean()
        rolling_std = df[col].rolling(window=window_size, center=center).std()
        
        # Calculate if a spike occurred based on threshold
        for i in range(window_size, len(df)):  # Start from window_size - 1 to have a full window
            print(f"i:{df[col].iloc[i]},mean:{rolling_mean.iloc[i]},std:{rolling_std.iloc[i]}")
            if abs(df[col].iloc[i] - rolling_mean.iloc[i-1]) > 2 * rolling_std.iloc[i-1]:
                spikes[i] = 1
            features.append(df[col][i - window_size : i ])
            print(series[i - window_size : i ])
            labels.append(spikes[i])

    spikes = spikes.astype(int)
    
    return spikes


# In[4]:


data = {'price':a}
df = pd.DataFrame(data)


# In[21]:


detect_spikes_new(df,'price',10,False)


# In[7]:


rolling_mean = df['price'].rolling(window=10, center=True).mean()


# In[9]:


rolling_mean[:20]


# In[ ]:




