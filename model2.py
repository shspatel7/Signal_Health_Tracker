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


#Sort the index
# test_data = test_data.sort_index()
# test_data_scaled = test_data_scaled.sort_index()

# #Plot the test data over the scaled test data
# plt.plot(test_data['power'], color='blue', label='Original')
# plt.plot(test_data.index, test_data_scaled, color='red', label='Scaled')
# plt.legend(loc='best')
# plt.show()


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
model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001)) # Add L2 regularization with lambda=0.01

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# Train the model
history = model.fit(train_data_reshaped, train_data_reshaped, epochs=150, batch_size=32, validation_data=(test_data_reshaped, test_data_reshaped), callbacks=[early_stopping], verbose=1)

# Make predictions on the test data
test_data_predictions = model.predict(test_data_reshaped)

#Plot the loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()



# #Plot the predictions
# test_data = test_data.sort_index()
# plt.plot(test_data['power'], color='blue', label='Original')
# plt.plot(test_data.index, test_data_predictions[:,0,0], color='red', label='Predicted')
# plt.legend(loc='best')
# plt.show()


# Calculate the mean squared error between the predictions and the actual data
mse = np.mean(np.power(test_data_reshaped - test_data_predictions, 2), axis=1)


#print the mean squared error
plt.plot(mse)
plt.show()

print(np.mean(mse))
print(np.std(mse))
print(np.mean(mse) + 3 * np.std(mse))

# Mark the data points as anomalies if the MSE is above a threshold
test_data['mse'] = mse
test_data['anomaly_predicted'] = np.where(test_data['mse'] > np.mean(test_data['mse']) + 3 * np.std(test_data['mse']), 1, 0)

#Align test data with increasing count
test_data = test_data.sort_index()


# Plot the results
plt.plot(test_data['power'])
plt.plot(test_data[test_data['anomaly'] == 1].index, test_data[test_data['anomaly'] == 1]['power'], 'ro')
plt.plot(test_data[test_data['anomaly_predicted'] == 1].index, test_data[test_data['anomaly_predicted'] == 1]['power'], 'go')
plt.show()

# Print the accuracy of the model
print(len(test_data[test_data['anomaly'] == test_data['anomaly_predicted']]))
print(len(test_data))
accuracy = len(test_data[test_data['anomaly'] == test_data['anomaly_predicted']]) / len(test_data)
print('Accuracy: ', accuracy)

# Print the precision of the model
try:
    precision = len(test_data[(test_data['anomaly'] == 1) & (test_data['anomaly_predicted'] == 1)]) / len(test_data[test_data['anomaly_predicted'] == 1])
    print('Precision: ', precision)
except ZeroDivisionError:
    print('Precision: 0')

#Print the index of the anomalies
print(test_data[test_data['anomaly'] == 1].index)
print(test_data[test_data['anomaly_predicted'] == 1].index)