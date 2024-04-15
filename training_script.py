import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from simple_lstm import SimpleLSTM
import joblib

# Load the dataset
dataset = pd.read_csv(r"C:\Users\Lenovo\Contacts\finale.csv")

# Select the features and target variable
features = dataset[['LAT', 'LON', 'PRECTOT', 'PS', 'QV2M', 'RH2M', 'T2MWET', 'TS', 'WS10M', 'WS50M']]
target = dataset['T2M']

# Normalize both the features and the target variable using Min-Max scaling
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

features_scaled = scaler_features.fit_transform(features)
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Reshape the data for LSTM input (samples, time steps, features)
features_reshaped = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_reshaped, target_scaled, test_size=0.2, random_state=42)

# Save the scaler
joblib.dump(scaler_features, "scaler.joblib")

# Instantiate the SimpleLSTM model
input_size = features_reshaped.shape[2]
hidden_size = 50
simple_lstm_model = SimpleLSTM(input_size, hidden_size)

# Train the model
final_weights = simple_lstm_model.train(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
joblib.dump(simple_lstm_model, "simple_lstm_model.joblib")
