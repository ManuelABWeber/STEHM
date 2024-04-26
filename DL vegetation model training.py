# cd "C:\0_Documents\10_ETH\Thesis\Python"
# env\Scripts\activate

import numpy as np
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping

# Load data from CSV files
X = np.loadtxt("C:/0_Documents/10_ETH/Thesis/finaltraining_x.csv", delimiter=',', skiprows=1)
y = np.loadtxt("C:/0_Documents/10_ETH/Thesis/finaltraining_y.csv", delimiter=',', skiprows=1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),  # Add dropout layer with dropout rate of 0.2
    Dense(100, activation='relu'),
    Dropout(0.2),  # Add dropout layer with dropout rate of 0.2
    Dense(100, activation='relu'),
    Dropout(0.2),  # Add dropout layer with dropout rate of 0.2
    Dense(3)  # No activation for regression
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),
              loss='mse')

# Custom callback to print intermediate results
class PrintIntermediateResults(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{self.params['epochs']}, Loss: {logs['loss']}, Val Loss: {logs['val_loss']}")

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=100, batch_size=1000, validation_data=(X_val, y_val), callbacks=[early_stopping, PrintIntermediateResults()])

# Evaluate the model
val_loss = model.evaluate(X_val, y_val)

# Predictions
predictions = model.predict(X_val)

# Calculate mean squared error
mse = mean_squared_error(y_val, predictions)
print("Mean Squared Error:", mse)

# Save the model
model.save("C:/0_Documents/10_ETH/Thesis/Python/growthformDLmodel120302")
