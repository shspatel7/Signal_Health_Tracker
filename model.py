import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data for signal power data
test_data = pd.read_csv('new.csv', header=None, names=['count', 'power'], index_col='count')

# Create a new column to mark the anomalies
test_data['anomaly'] = np.where(np.abs(test_data['power'].diff()) > 3, 1, 0)

# Plot the data
plt.plot(test_data['power'])
plt.show()

# Normalize the data using standard scaling
scaler = StandardScaler()
test_data_scaled = scaler.fit_transform(test_data[['power']])

# Reshape the data for LSTM input
test_data_reshaped = test_data_scaled.reshape(test_data_scaled.shape[0], 1, test_data_scaled.shape[1])

# Load the saved model
model = load_model('model.h5')

# Print the diagram of the model
model.summary()

# Predict the test data
predictions = model.predict(test_data_reshaped)

# Calculate the mean squared error between the predictions and the actual data
mse = np.mean(np.power(test_data_reshaped - predictions, 2), axis=1)

# Mark the data points as anomalies if the MSE is above a threshold
test_data['mse'] = mse
test_data['anomaly_predicted'] = np.where(test_data['mse'] > 0.1, 1, 0)

#Align test data with increasing count
test_data = test_data.sort_index()


# Plot the results
plt.plot(test_data['power'])
plt.plot(test_data[test_data['anomaly'] == 1].index, test_data[test_data['anomaly'] == 1]['power'], 'ro')
plt.plot(test_data[test_data['anomaly_predicted'] == 1].index, test_data[test_data['anomaly_predicted'] == 1]['power'], 'go')
plt.show()

# Print the accuracy of the model
accuracy = len(test_data[test_data['anomaly'] == test_data['anomaly_predicted']]) / len(test_data)
print('Accuracy: ', accuracy)

# Print the precision of the model
precision = len(test_data[(test_data['anomaly'] == 1) & (test_data['anomaly_predicted'] == 1)]) / len(test_data[test_data['anomaly_predicted'] == 1])
print('Precision: ', precision)

#Print the index of the anomalies
print(test_data)
print(test_data[test_data['anomaly'] == 1].index)
print(test_data[test_data['anomaly_predicted'] == 1].index)