#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[13]:


data_train = pd.read_csv("/Users/wbm/Documents/Dicoding ML/Time-Series/data_temp.csv")
data_train.head()


# In[14]:


data_train = data_train[["time", "Temperature"]]
data_train.head()


# In[15]:


# check whether there are null value
data_train.isnull().sum()


# In[16]:


# Drop the missing value
data_train = data_train.dropna()
data_train.count()


# In[17]:


# Plot the dataset
time = data_train['time'].values
temperature = data_train['Temperature'].values

plt.figure(figsize=(15,5))
plt.plot(time, temperature)
plt.title('Furnace Temperature', fontsize = 20)


# In[18]:


# Function for converting our input data to the format that can be ingest by the model.
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)  # Corrected 'drop_remainder'
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).prefetch(1)



# In[19]:


# Calculate 10% of the temperature scale for MAE target
temperature_range = temperature.max() - temperature.min()
mae_target = 0.1 * temperature_range
print(f"MAE target: {mae_target}")


# In[20]:


# Split the data into training and validation dataset
time_train, time_valid, temp_train, temp_valid = train_test_split(time, temperature, test_size=0.2, random_state=42)


# In[21]:


# Create windowed dataset for training
train_set = windowed_dataset(temp_train, window_size= 60, batch_size=100, shuffle_buffer=1000)

# Create windowed dataset for validation
valid_set = windowed_dataset(temp_valid, window_size= 60, batch_size=100, shuffle_buffer=1000)


# In[22]:


# Model Training

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(60, return_sequences= True),
    tf.keras.layers.LSTM(60),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
])


# In[23]:


# Setting Optimizer

optimizer = tf.keras.optimizers.SGD(learning_rate=1.0000e-04, momentum=0.9)

# Compiling the model
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# Fit the model on the training set and validate on the validation set
history = model.fit(train_set,epochs=100, validation_data=valid_set)


# In[ ]:




