import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the signal power data
data = pd.read_csv('save.csv', header=None, names=['count', 'power'], index_col='count')

# Plot the data
plt.plot(data['power'])
plt.show()

# Define the threshold for anomaly detection
threshold = 3.0

# Create a new column to mark the anomalies
data['anomaly'] = np.where(np.abs(data['power'].diff()) > threshold, 1, 0)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Normalize the data using standard scaling
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data[['power']])
test_data_scaled = scaler.transform(test_data[['power']])

# Reshape the data for LSTM input
train_data_reshaped = train_data_scaled.reshape(train_data_scaled.shape[0], 1, train_data_scaled.shape[1])
test_data_reshaped = test_data_scaled.reshape(test_data_scaled.shape[0], 1, test_data_scaled.shape[1])

# Define the LSTM Autoencoder model
model = Sequential()
model.add(LSTM(128, input_shape=(1, 1), return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(RepeatVector(1))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_regularizer=l2(0.01)))
model.compile(loss='mse', optimizer=Adam(lr=0.0001)) # Add L2 regularization with lambda=0.01

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# Train the model
history = model.fit(train_data_reshaped, train_data_reshaped, epochs=150, batch_size=32, validation_data=(test_data_reshaped, test_data_reshaped), callbacks=[early_stopping], verbose=1)

# Calculate mean and standard deviation on training set
train_mse = model.predict(train_data_reshaped)
train_mse = np.mean(np.power(train_data_reshaped - train_mse, 2), axis=1)
mean = np.mean(train_mse)
std = np.std(train_mse)

# Mark the data points as anomalies if the MSE is above a threshold
test_mse = model.predict(test_data_reshaped)
test_mse = np.mean(np.power(test_data_reshaped - test_mse, 2), axis=1)
test_data['mse'] = test_mse

# Create a new column to mark the anomalies that are predicted by the model and the values that are greater than the threshold of 3 db power change
test_data['anomaly_predicted'] = np.where(test_mse > mean + 3*std, 1, 0)
test_data['anomaly'] = np.where((np.abs(test_data['power'].diff()) > threshold) & (test_data['anomaly_predicted'] == 1), 1, 0)


#Print the index of the anomalies
print(test_data[test_data['anomaly'] == 1].index)
print(test_data[test_data['anomaly_predicted'] == 1].index)

#Sort the data by the time
test_data = test_data.sort_index()

#Plot the data
plt.plot(test_data.index, test_data['power'])
plt.scatter(test_data[test_data['anomaly'] == 1].index, test_data[test_data['anomaly'] == 1]['power'], color='r')
plt.show()
