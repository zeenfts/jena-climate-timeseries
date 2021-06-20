# -*- coding: utf-8 -*-
"""
# <center> Climate Time Series - Weather

---

<center> [dataset](https://www.kaggle.com/mnassrib/jena-climate)

<small> *note: the output was run on GPU mode*
"""

import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from pip._internal import main as pipmain
pipmain(['install', 'tensorflow-addons'])

from tensorflow_addons.optimizers import AdamW
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
np.random.seed(233)
tf.random.set_seed(233)

# only use the last three years (2013-2016)
train = pd.read_csv('jena_climate_2009_2016.csv', parse_dates=['Date Time'])
train[train['Date Time'] == '2013-01-01 00:00:00']

train_copy = train.copy()
train = train.iloc[210525:].reset_index(drop=True)

# only choose column 'T (degC)' because the pattern almost the same for all columns

"""# Data Preparation"""
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).prefetch(1)

# data_sample = windowed_dataset(train['T (degC)'], 60, 64, 100)

train.shape[0] * .8, train.shape[0] * .2 #42006

# Validation (+-20%) - just take it like that because timestamp every 10 minutes thus skip the daily/monthly cut
### train_data, valid_data = train_test_split(train, stratify=train['T (degC)'], test_size=.2)
val_index = int(len(train) - 42006)

train_temp = train[['T (degC)']].iloc[:val_index]
valid_temp = train[['T (degC)']].iloc[val_index:]

scaler = MinMaxScaler()

scaled_train = scaler.fit_transform(train_temp[['T (degC)']])
scaled_valid = scaler.transform(valid_temp[['T (degC)']])

LNGTH, LEN2, BATCH = 168019, 42005, 32

train_gen = TimeseriesGenerator(scaled_train, scaled_train, length=LNGTH, batch_size=BATCH)
valid_gen = TimeseriesGenerator(scaled_valid, scaled_valid, length=LEN2, batch_size=BATCH)

# Test for three days in the future (334 datapoint)
future_dt = np.array([[i] for i in range(334)])     
future_gen = TimeseriesGenerator(future_dt, future_dt, length=333, batch_size=BATCH)

"""# Modelling"""
SCHEDULE = tf.optimizers.schedules.PiecewiseConstantDecay([1407*20, 1407*30], [1e-3, 1e-4, 1e-5])
step = tf.Variable(0, trainable=False)
schedule = tf.optimizers.schedules.PiecewiseConstantDecay([10000, 15000], [1e-0, 1e-1, 1e-2])
LR = 1e-1 * schedule(step)
WD = lambda: 1e-4 * SCHEDULE(step)
OPTIMIZER = AdamW(learning_rate=SCHEDULE, weight_decay=WD)

def build_model():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True), input_shape=(168019,1)))
  model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)))
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dropout(.3))
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dropout(.2))
  model.add(tf.keras.layers.Dense(32, activation='relu'))
  model.add(tf.keras.layers.Dense(1))

  model.compile(loss=tf.losses.MeanSquaredError(), 
                optimizer=OPTIMIZER, 
                metrics=[tf.metrics.MeanAbsoluteError()])
  return model

def format_time(elapsed):
  '''
  Takes a time in seconds and returns a string hh:mm:ss
  '''
  # Round to the nearest second.
  elapsed_rounded = int(round((elapsed)))

  # Format as hh:mm:ss
  return str(datetime.timedelta(seconds=elapsed_rounded))

cb = tf.keras.callbacks

model = build_model()
stopper = cb.EarlyStopping(patience=3, min_delta=0.05, baseline=0.8,
              mode='min', monitor='val_mean_absolute_error', restore_best_weights=True,
              verbose=1)

tf.keras.utils.plot_model(model, show_shapes=True, rankdir='TP')

total_t0 = time.time()

hist = model.fit(train_gen, epochs=10, validation_data=valid_gen, callbacks=[stopper], verbose=2)

print('')
print('Training complete!')

print('Total training took {:} (h:mm:ss)'.format(format_time(time.time()-total_t0)))

"""# Evaluation"""

eval_df = pd.DataFrame(hist.history)
length = len(eval_df)

"""# Forecasting"""

# fit model & predict the future (only 3 days because this method will give a long time running)
def pred_future_deep(model):
    list_pred = []
    
    first_eval_batch = future_dt
    current_batch = first_eval_batch.reshape((1, 334, 1))
    
    for i in range(len(future_dt)):

        # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
        current_pred = [i for i in model.predict(current_batch)[0]]

        # store prediction
        list_pred.append(current_pred) 

#         if i == 0:
            # update batch to now include prediction and drop first value
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
            
#         current_batch = np.append(current_batch[:],[[current_pred]],axis=1)
        
    return list_pred

first_jan_2017 = pred_future_deep(model)
first_jan_2017 = scaler.inverse_transform(first_jan_2017)
first_jan_2017.shape

res_sample = pd.DataFrame(first_jan_2017)
res_sample.columns = ['degC res']
res_sample['idx'] = [i for i in range(210026,210360)]
res_sample = res_sample.set_index(['idx'])

plt.figure(figsize=(15,7))
plt.plot(train['T (degC)'])
plt.plot(res_sample)
# plt.xticks(pd.date_range('2017-01-01 00:00:00', periods=334, freq='10min'))