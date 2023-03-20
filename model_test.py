import unittest
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class TestKerasModel(unittest.TestCase):
    def setUp(self):
        # Load the trained model
        self.model = load_model('model3.h5')

        # Get data from csv file
        self.data = pd.read_csv('save.csv', header=None, names=['count', 'power'], index_col='count')

        # Split the data into training and testing sets
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)

        # Create the input/output data for the model
        self.X = np.abs(np.diff(self.train_data['power']))
        self.train_data['anomaly'] = np.where(np.abs(self.train_data['power'].diff()) > 3.0, 1, 0)
        self.y = np.array(self.train_data['anomaly'][1:])

        # Get the new X values for the test data from the new.csv file
        self.new_test_data = pd.read_csv('new.csv', header=None, names=['count', 'power'], index_col='count')
        self.new_test_data.index = range(0, len(self.new_test_data))

        # Create the new input data for the model
        self.new_X = np.abs(np.diff(self.new_test_data['power']))
        self.new_test_data['anomaly'] = np.where(np.abs(self.new_test_data['power'].diff()) > 3.0, 1, 0)

        # Determine the threshold value for anomaly detection in predicted data where the change is greater than 3
        # between two consecutive values in original data set
        self.predicted_data_threshold = np.mean(self.model.predict(self.new_X)) + 3 * np.var(self.model.predict(self.new_X))

        # Add 1 more row to the prediction array in front to match the test data
        self.predictions = np.insert(self.model.predict(self.new_X), 0, 0)

        # Create a new column to mark the predicted anomalies
        self.new_test_data['anomaly_predicted'] = np.where(self.predictions > self.predicted_data_threshold, 1, 0)

    def test_model_architecture(self):
        # Check if the model architecture matches the expected values
        expected_layers = 6
        expected_output_shape = (None, 1)
        self.assertEqual(len(self.model.layers), expected_layers)
        self.assertEqual(self.model.output_shape, expected_output_shape)

    def test_model_compile(self):
        # Check if the model is compiled with the correct loss function and optimizer
        expected_loss = binary_crossentropy
        expected_optimizer = Adam(learning_rate=0.001)
        self.assertEqual(self.model.loss, expected_loss)
        self.assertEqual(self.model.optimizer.get_config(), expected_optimizer.get_config())

    def test_model_training(self):
        # Check if the model training produces the expected results
        expected_train_size = 640
        expected_val_size = 160
        expected_epochs = 50
        expected_batch_size = 32
        self.assertEqual(len(self.X), expected_train_size)
        self.assertEqual(len(self.test_data), expected_val_size)
        history = self.model.fit(self.X, self.y, epochs=expected_epochs, batch_size=expected_batch_size, validation_split=0.2)
        self.assertAlmostEqual(history.history['loss'][-1], 0.0504, places=4)
        self.assertAlmostEqual(history.history['val_loss'][-1], 0.0485, places=4)

    def test_anomaly_detection(self):
        # Check if the model can detect anomalies in the test data
        expected_anomalies = [20, 28, 43, 56, 63, 66, 75, 86, 96]
        self.test_data['anomaly'] = np.where(np.abs(self.test_data['power'].diff()) > 3.0, 1, 0)
        actual_anomalies = self.test_data[self.test_data['anomaly'] == 1].index.tolist()
        self.assertEqual(actual_anomalies, expected_anomalies)

    def test_new_data_prediction(self):
        # Check if the model can predict anomalies in new test data
        expected_anomalies = self.new_test_data[self.new_test_data['anomaly'] == 1].index.tolist()
        actual_anomalies = self.new_test_data[self.new_test_data['anomaly_predicted'] == 1].index.tolist()
        self.assertEqual(actual_anomalies, expected_anomalies)

if __name__ == '__main__':
    unittest.main()