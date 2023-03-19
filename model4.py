import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import matplotlib.pyplot as plt

# Load the data for signal power data
data = pd.read_csv('save.csv', header=None, names=['count', 'power'], index_col='count')
data['anomaly'] = np.where(np.abs(data['power'].diff()) > 3.0, 1, 0)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define the LSTM autoencoder model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(1, 1)))
model.add(RepeatVector(1))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_data['power'].values.reshape(-1, 1, 1), train_data['power'].values.reshape(-1, 1, 1), epochs=50, batch_size=32)

# Use the model to predict on the test data
test_predictions = model.predict(test_data['power'].values.reshape(-1, 1, 1))

# Calculate the mean squared error between the test data and the predictions
mse = np.mean(np.power(test_data['power'].values.reshape(-1, 1, 1) - test_predictions, 2), axis=1)

print(mse)

#plot mse values
plt.plot(mse)
plt.show()

# Determine the threshold for anomaly detection
threshold = np.mean(mse) + np.std(mse)
print(threshold)

# Label the anomalies in the test data
test_data['anomaly_predicted'] = np.where(mse > threshold, 1, 0)



# Print the results
print(test_data[['power', 'anomaly', 'anomaly_predicted']])