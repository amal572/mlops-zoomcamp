#!/usr/bin/env python
# coding: utf-8


# In[24]:


import pickle
import pandas as pd
import math
import os


# In[5]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[6]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[7]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[8]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[9]:


mean_duration = sum(y_pred) / len(y_pred)


# In[10]:


variance = sum((X_val - mean_duration) ** 2 for X_val in y_pred) / len(y_pred)


# In[12]:


standard_deviation = math.sqrt(variance)

print("Standard Deviation of the predicted durations:", standard_deviation)


# In[16]:


year = 2023
month = 3


# In[17]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[18]:


df['ride_id']


# In[19]:


df_result = pd.DataFrame()


# In[20]:


df_result['ride_id'] =df['ride_id']
df_result['predictions'] =y_pred


# In[21]:


df_result.head()


# In[22]:


output_file = 'result.bin'


# In[23]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[25]:


file_size = os.path.getsize(output_file)


# In[26]:


print(file_size)


# In[28]:


# Convert bytes to megabytes
file_size_mb = file_size / (1024 * 1024)
file_size_mb


# In[ ]:




