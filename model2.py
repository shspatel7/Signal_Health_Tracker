import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed, Flatten
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

train_data = train_data.sort_index()
test_data = test_data.sort_index()

# Create the input and output data for the model
X = np.abs(np.diff(train_data['power']))
y = np.array(train_data['anomaly'][1:])


#Reshape the data so it 3 dimensional for the LSTM model
X = X.reshape((X.shape[0], 1, 1))
y = y.reshape((y.shape[0], 1, 1))

# Define the LSTM model with encoder-decoder architecture
model = Sequential()
model.add(LSTM(128, input_shape=(1, 1), activation='relu', return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(16, activation='relu', return_sequences=True))
model.add(LSTM(8, activation='relu', return_sequences=True))
model.add(LSTM(4, activation='relu', return_sequences=True))
model.add(LSTM(2, activation='relu', return_sequences=True))
model.add(LSTM(1, activation='relu', return_sequences=True))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mae', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


print(model.summary())

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

# Train the model
model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

#Change the index of the test data
test_data.index = range(0,len(test_data))

# Make predictions on the test data
new_X = np.abs(np.diff(test_data['power']))
#Reshape the data so it 3 dimensional for the LSTM model
new_X = new_X.reshape((new_X.shape[0], 1, 1))

test_data_predictions = model.predict(new_X)

#Plot the loss


plt.plot(test_data_predictions)
plt.plot(new_X[:,0,0])
plt.show()

print(test_data_predictions.shape)
print(new_X.shape)

#Statistics of the predictions
print("Mean: ", np.mean(test_data_predictions))
print("Standard Deviation: ", np.std(test_data_predictions))
print("Min: ", np.min(test_data_predictions))
print("Max: ", np.max(test_data_predictions))
print("Median: ", np.median(test_data_predictions))
print("Variance: ", np.var(test_data_predictions))
print("Percentile: ", np.percentile(test_data_predictions, 90))


# Calculate the mean average error between the predictions and the actual data
test_mae = np.mean(np.abs(new_X - test_data_predictions), axis=1)

# train_mae = np.mean(np.abs(train_data_reshaped - model.predict(train_data_reshaped)), axis=1)

# #print the mean squared error
# plt.hist(train_mae, bins=50)
# plt.xlabel("Train MAE loss")
# plt.ylabel("No of samples")
# plt.show()


plt.plot(test_mae)
plt.show()



# Mark the data points as anomalies if the MSE is above a threshold
test_data['mae'] = test_mae
prediction_threshold = np.mean(test_mae) + 4 * np.var(test_mae)
test_data['anomaly_predicted'] = np.where(test_data['mae'] > prediction_threshold, 1, 0)


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

#Print the actual power values of the anomalies
print(test_data[test_data['anomaly'] == 1]['power'])
print(test_data[test_data['anomaly_predicted'] == 1]['power'])