import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
dataset = pd.read_csv(r"C:\Users\Lenovo\PycharmProjects\lstm_weather_forecast\static\combined_file.csv")
dataset['PRECTOT'].fillna(dataset['PRECTOT'].mean(), inplace=True)

# Select the features and target variables
features = dataset[['LAT', 'LON', 'PRECTOT', 'PS', 'QV2M', 'T2MWET', 'TS', 'WS50M','WS10M','WS50M_RANGE','T2M_MAX','T2M_MIN','T2M_RANGE','WS10M_MAX','WS10M_MIN','WS50M_MAX','WS50M_MIN','WS50M_RANGE']]
targets = dataset[['T2M', 'RH2M', 'WS10M_RANGE']]  # Features to predict: 'T2M', 'RH2M', 'WS10M_RANGE'

# Normalize both the features and the target variables using Min-Max scaling
scaler_features = MinMaxScaler()
scaler_targets = MinMaxScaler()

features_scaled = scaler_features.fit_transform(features)
targets_scaled = scaler_targets.fit_transform(targets)

# Reshape the data for LSTM input (samples, time steps, features)
features_reshaped = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_reshaped, targets_scaled, test_size=0.2, random_state=42)

# Define the LSTM model using TensorFlow's built-in layers
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(1, features_reshaped.shape[2])),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(targets_scaled.shape[1])  # Number of units = number of target features
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
epochs = 1
batch_size = 32
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Final Test Loss: {test_loss}')

# Make predictions
test_predictions = model.predict(X_test)

# Inverse transform the scaled predictions and observed values
y_pred = scaler_targets.inverse_transform(test_predictions)
y_observed = scaler_targets.inverse_transform(y_test)

# Print actual vs predicted values for all target features
for i in range(targets_scaled.shape[1]):
    target_name = targets.columns[i]
    target_rmse = np.sqrt(mean_squared_error(y_observed[:, i], y_pred[:, i]))
    target_mse = mean_squared_error(y_observed[:, i], y_pred[:, i])
    target_mae = mean_absolute_error(y_observed[:, i], y_pred[:, i])
    print(f'Target Feature: {target_name}')
    print(f'  Root Mean Squared Error (RMSE): {target_rmse}')
    print(f'  Mean Squared Error (MSE): {target_mse}')
    print(f'  Mean Absolute Error (MAE): {target_mae}')

    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Actual': y_observed[:, i],
        'Predicted': y_pred[:, i]
    })
    print(comparison_df)

# Save the model
model.save("multi_feature_lstm_model.h5")