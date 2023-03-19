import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping

# define the model architecture
model = Sequential()
model.add(Dense(128, input_dim=1, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Get data from csv file
data = pd.read_csv('save.csv', header=None, names=['count', 'power'], index_col='count')

# Plot the data
plt.plot(data['power'])
plt.show()

#Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


train_data = train_data.sort_index()
test_data = test_data.sort_index()
# create the input/output data for the model
X = np.abs(np.diff(train_data['power']))
train_data['anomaly'] = np.where(np.abs(train_data['power'].diff()) > 3.0, 1, 0)
y = np.array(train_data['anomaly'][1:])



early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')


# train the model
model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

#Change the index of the test data so it has index in the increasing order starting from 0
test_data.index = range(0, len(test_data))

# use the model to predict changes in new data
new_X = np.abs(np.diff(test_data['power']))
test_data['anomaly'] = np.where(np.abs(test_data['power'].diff()) > 3.0, 1, 0)
predictions = model.predict(new_X)

#Get the statistics of the predicted data
print("Mean of the predicted data is: ", np.mean(predictions))
print("Standard deviation of the predicted data is: ", np.std(predictions))
print("Minimum value of the predicted data is: ", np.min(predictions))
print("Maximum value of the predicted data is: ", np.max(predictions))
print("Median of the predicted data is: ", np.median(predictions))
print("Variance of the predicted data is: ", np.var(predictions))
print("Percentile of the predicted data is: ", np.percentile(predictions, 90))


#Plot the loss values over the new_X values
plt.plot(predictions)
plt.plot(new_X)
plt.show()

print(predictions.shape)
print(new_X.shape)

#Determine the threshold value for anomaly detection in predicted data where the change is greater than 3
# between two consecutive values in original data set

predicted_data_threshold = np.mean(predictions) + 3 * np.var(predictions)
print(predicted_data_threshold)
#Add 1 more row to the prediction array in front to match the test data
predictions = np.insert(predictions, 0, 0)



#Create a new column to mark the predicted anomalies
test_data['anomaly_predicted'] = np.where(predictions > predicted_data_threshold, 1, 0)

#Print the Index of the anomalies
print(test_data[test_data['anomaly_predicted'] == 1].index)
print(test_data[test_data['anomaly'] == 1].index)

#Find the accuracy of the model and precision of the model based on the predicted anomalies
TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(test_data)):
    if test_data['anomaly'][i] == 1 and test_data['anomaly_predicted'][i] == 1:
        TP += 1
    if test_data['anomaly'][i] == 1 and test_data['anomaly_predicted'][i] == 0:
        FN += 1
    if test_data['anomaly'][i] == 0 and test_data['anomaly_predicted'][i] == 0:
        TN += 1
    if test_data['anomaly'][i] == 0 and test_data['anomaly_predicted'][i] == 1:
        FP += 1

#Plot the TP, FP, TN, FN on the confusion matrix
plt.plot(TP, 'ro')
plt.plot(FP, 'bo')
plt.plot(TN, 'go')
plt.plot(FN, 'yo')
plt.legend(['TP', 'FP', 'TN', 'FN'])
plt.show()

print("True Positive: ", TP)
print("False Positive: ", FP)
print("True Negative: ", TN)
print("False Negative: ", FN)

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)

print("Accuracy: ", accuracy)
print("Precision: ", precision)




#Plot the anomaly and predicted anomaly
plt.plot(test_data['power'])
plt.plot(test_data[test_data['anomaly'] == 1].index, test_data[test_data['anomaly'] == 1]['power'], 'ro')
plt.plot(test_data[test_data['anomaly_predicted'] == 1].index, test_data[test_data['anomaly_predicted'] == 1]['power'], 'go')
plt.show()



