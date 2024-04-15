import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

# Basic LSTM model implementation using only NumPy
class SimpleLSTM:
    def __init__(self, input_size, hidden_size):
        scale = 0.01  # Adjust this scale as needed
        self.Wf = np.random.randn(input_size, hidden_size) * scale
        self.Wi = np.random.randn(input_size, hidden_size) * scale
        self.Wc = np.random.randn(input_size, hidden_size) * scale
        self.Wo = np.random.randn(input_size, hidden_size) * scale

        self.Uf = np.random.randn(hidden_size, hidden_size) * scale
        self.Ui = np.random.randn(hidden_size, hidden_size) * scale
        self.Uc = np.random.randn(hidden_size, hidden_size) * scale
        self.Uo = np.random.randn(hidden_size, hidden_size) * scale

        self.bias_f = np.zeros((1, hidden_size))
        self.bias_i = np.zeros((1, hidden_size))
        self.bias_c = np.zeros((1, hidden_size))
        self.bias_o = np.zeros((1, hidden_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, weights):
        h_t = np.zeros((x.shape[0], x.shape[1], self.Wf.shape[1]))

        # Initialize cache
        cache = []

        for t in range(x.shape[1]):
            x_t = x[:, t, :]

            f_t = self.sigmoid(np.dot(x_t, weights[0]) + np.dot(h_t[:, t - 1, :], weights[4]) + weights[8])
            i_t = self.sigmoid(np.dot(x_t, weights[1]) + np.dot(h_t[:, t - 1, :], weights[5]) + weights[9])
            c_t_hat = self.tanh(np.dot(x_t, weights[2]) + np.dot(h_t[:, t - 1, :], weights[6]) + weights[10])
            o_t = self.sigmoid(np.dot(x_t, weights[3]) + np.dot(h_t[:, t - 1, :], weights[7]) + weights[11])

            h_t[:, t, :] = f_t * h_t[:, t - 1, :] + i_t * c_t_hat
            h_t[:, t, :] = o_t * self.tanh(h_t[:, t, :])

            # Store values in cache
            cache.append((h_t[:, t, :], c_t_hat, h_t[:, t - 1, :], c_t_hat, f_t, i_t, o_t, c_t_hat, x_t))

        return h_t, cache

    def backward(self, da_next, dc_next, cache):
        """
        Implement the backward pass for a single time step of a LSTM.

        Arguments:
        da_next -- Gradient of loss with respect to next hidden state
        dc_next -- Gradient of loss with respect to next cell state
        cache -- cache storing information from the forward pass

        Returns:
        gradients -- dictionary containing gradients with respect to various parameters
        """
        # Unpack the cache
        (a_next, c_next, a_prev, c_prev, f_t, i_t, o_t, c_t_hat, x_t) = cache

        # Dimensions
        n_x, m = x_t.shape
        n_a, m = a_next.shape

        # Compute the gradients
        dot = da_next * np.tanh(c_next) * o_t * (1 - o_t)
        dcct = (dc_next * i_t + o_t * (1 - np.tanh(c_next) ** 2) * i_t * da_next) * (1 - c_t_hat ** 2)
        dit = (dc_next * c_t_hat + o_t * (1 - np.tanh(c_next) ** 2) * c_t_hat * da_next) * i_t * (1 - i_t)
        dft = (dc_next * c_prev + o_t * (1 - np.tanh(c_next) ** 2) * c_prev * da_next) * f_t * (1 - f_t)
        dgt = (dc_next * f_t + o_t * (1 - np.tanh(c_next) ** 2) * f_t * da_next) * (1 - c_t_hat ** 2)

        # Gradients with respect to parameters
        dWf = np.dot(dft, np.concatenate((a_prev, x_t), axis=0).T)
        dWi = np.dot(dit, np.concatenate((a_prev, x_t), axis=0).T)
        dWc = np.dot(dcct, np.concatenate((a_prev, x_t), axis=0).T)
        dWo = np.dot(dot, np.concatenate((a_prev, x_t), axis=0).T)
        dbf = np.sum(dft, axis=1, keepdims=True)
        dbi = np.sum(dit, axis=1, keepdims=True)
        dbc = np.sum(dcct, axis=1, keepdims=True)
        dbo = np.sum(dot, axis=1, keepdims=True)

        # Gradients with respect to previous timestep
        da_prev = np.dot(weights[0][:, :n_a].T, dft) + np.dot(weights[1][:, :n_a].T, dit) + np.dot(
            weights[2][:, :n_a].T, dcct) + np.dot(weights[3][:, :n_a].T, dot)
        dc_prev = dc_next * f_t + o_t * (1 - np.tanh(c_next) ** 2) * f_t * da_next

        gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dWi": dWi, "dWc": dWc, "dWo": dWo,
                     "dbf": dbf, "dbi": dbi, "dbc": dbc, "dbo": dbo}

        return gradients

    def train(self, X_train, y_train, epochs, batch_size, learning_rate=0.01, validation_data=None):
        input_size = X_train.shape[2]
        hidden_size = self.Wf.shape[1]

        # Initialize weights
        weights = [self.Wf, self.Wi, self.Wc, self.Wo, self.Uf, self.Ui, self.Uc, self.Uo,
                   self.bias_f, self.bias_i, self.bias_c, self.bias_o]

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                # Get mini-batch
                mini_batch_X = X_train_shuffled[i:i + batch_size]
                mini_batch_y = y_train_shuffled[i:i + batch_size]

                # Forward pass
                h_t = self.forward(mini_batch_X, weights)

                # Backward pass (Gradient descent)
                gradients = self.backward(h_t, mini_batch_y, weights)

                # Update weights using mini-batch gradient descent
                for w, grad in zip(weights, gradients):
                    w -= learning_rate * grad

            # Print loss for each epoch
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss}')

            # Validation loss
            if validation_data is not None:
                X_val, y_val = validation_data
                val_h_t = self.forward(X_val, weights)
                val_loss = np.mean(0.5 * (val_h_t - y_val) ** 2)
                print(f'Validation Loss: {val_loss}')

        # Update weights in the model
        self.Wf, self.Wi, self.Wc, self.Wo, self.Uf, self.Ui, self.Uc, self.Uo, \
            self.bias_f, self.bias_i, self.bias_c, self.bias_o = weights

        return weights

    # Instantiate the SimpleLSTM model
input_size = features_reshaped.shape[2]
hidden_size = 50
simple_lstm_model = SimpleLSTM(input_size, hidden_size)

# Train the model and get the final weights
final_weights = simple_lstm_model.train(X_train, y_train, epochs=10, batch_size=32,
                                        validation_data=(X_test, y_test))

# Make predictions on the test set
y_pred_scaled = simple_lstm_model.forward(X_test, final_weights)

# Reshape y_pred_scaled to 2D array
y_pred_scaled_2d = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])

# Inverse transform the scaled predictions and observed values
y_pred = scaler_target.inverse_transform(y_pred_scaled_2d)
y_observed = scaler_target.inverse_transform(y_test.reshape(-1, 1))  # Reshape y_test to 2D array

# Ensure that 'Actual' and 'Predicted' arrays have the same length
min_length = min(len(y_observed), len(y_pred))
y_observed = y_observed[:min_length]
y_pred = y_pred[:min_length]

# Print the shapes of y_observed and y_pred for diagnosis
print("Shape of y_observed:", y_observed.shape)
print("Shape of y_pred:", y_pred.shape)

# Compare the predicted and observed values
min_length = min(len(y_observed.flatten()), len(y_pred.flatten()))
comparison_df = pd.DataFrame({
    'Actual': y_observed.flatten()[:min_length],
    'Predicted': y_pred.flatten()[:min_length]
})

# Print the comparison DataFrame
print(comparison_df)

