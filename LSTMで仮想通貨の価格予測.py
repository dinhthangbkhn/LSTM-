"""This code is based on https://qiita.com/licht110/items/f89c699cbdff05ec90de"""

import poloniex
import time
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf



print("----------DOWNLOAD DATA----------")
polo = poloniex.Poloniex()
polo.timeout = 2
rawdata = polo.returnChartData('USDT_BTC',
                               period=300,
                               start=time.time()-3600*24*180,
                               end=time.time())
price_data = pd.DataFrame([float(i.get('open')) for i in rawdata])
mss = MinMaxScaler()
input_dataframe = pd.DataFrame(mss.fit_transform(price_data))


print("----------PREPROCESS DATA----------")
def _load_data(data,n_prev=50):
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].values)
        docY.append(data.iloc[i+n_prev].values)
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def train_test_split(df, test_size=0.1, n_prev=50):
    ntrn = int(round(len(df)*(1-test_size)))
    X_train, Y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, Y_test = _load_data(df.iloc[ntrn:], n_prev)
    return (X_train, Y_train), (X_test, Y_test)

input_dataframe = input_dataframe[0:2000]
(X_train, Y_train), (X_test, Y_test) = train_test_split(input_dataframe)

print(X_train.shape)
print(Y_train.shape)

print("-----------BUILD MODEL----------")
hidden_neurons = 300
in_out_neurons = 1
length_of_sequences = 50

model = keras.Sequential()
model.add(keras.layers.LSTM(units=hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons)))
model.add(keras.layers.Dense(in_out_neurons))
model.add(keras.layers.Activation("linear"))
model.compile(loss="mean_squared_error", optimizer=tf.train.AdamOptimizer(0.01), metrics = ["accuracy"])
model.fit(X_train,Y_train, batch_size= 600, epochs = 2, validation_split=0.2)
# keras.models.save_model(model, filepath="./checkpoint/LSTMcheckpoint.ckpt")
# saver.save(model, save_path=)
# model.save_weights("./checkpoint/LSTMcheckpoint.ckpt")


print("----------EVALUATE MODEL----------")
model.evaluate(X_test, Y_test)

pred_data = model.predict(X_train)
plt.plot(Y_test, label= "train")
plt.plot(model.predict(X_test), label = "pred")
plt.legend(loc="upper left")
# plt.show()
plt.savefig('figure.png')