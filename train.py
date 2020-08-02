import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pandas as pd 
import numpy as np  
import datetime

LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 30

data = pd.read_csv('data/data.csv')

y = data.pop('co') # To predict humidity or co (which ever performs better lol)

data = data.drop(['ts', 'device'], axis=1) # Drop useless columns

# Normalize humidity and temp

data['humidity'] = (data['humidity']-data['humidity'].min())/(data['humidity'].max()-data['humidity'].min())
data['temp'] = (data['temp']-data['temp'].min())/(data['temp'].max()-data['temp'].min())

# One hot encode light and motion columns

one_hots = np.concatenate((np.expand_dims(pd.get_dummies(data.pop('light'), 
							dtype=float)[1].values, axis=1), 
							np.expand_dims(pd.get_dummies(data.pop('motion'), 
							dtype=float)[1].values, axis=1)), axis=1)

X = np.concatenate((data.values, one_hots), axis=1)

# Set for predictions

X, y = X[:-1], y[:-1]
X_pred, y_pred = X[-1:], y[-1:]

model = Sequential()
model.add(Dense(6, input_dim=6, activation='relu'))
model.add(Dense(12, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(18, activation='relu'))
#model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

Optimizer = SGD(lr=LR, momentum=0.0)

model.compile(optimizer=Optimizer, loss='MSE')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.3, callbacks=[tensorboard_callback])

print(y_pred)
print(model.predict(X_pred))