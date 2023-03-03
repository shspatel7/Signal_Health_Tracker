import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the signal power data
data = pd.read_csv('new.csv', header=None, names=['count', 'power'], index_col='count')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(train_data)
print(test_data)

# Normalize the data using standard scaling
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define the Keras model
model = Sequential()
model.add(Dense(256, input_dim=train_data.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])

# Set up early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(train_data, train_data, epochs=200, batch_size=64, validation_data=(test_data, test_data), callbacks=[early_stop])

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_data, test_data)

# Make predictions on the test data
predictions = model.predict(test_data)

# Identify the anomalies in the predictions
anomalies = np.where(np.abs(predictions - test_data) > 3)[0]

# Print the anomalies
print('Anomalies:', anomalies)
