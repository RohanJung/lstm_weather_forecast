import joblib
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
        self.input_size = input_size
        self.hidden_size = hidden_size

    def set_weights(self, weights):
        # Set the weights externally
        self.Wf, self.Wi, self.Wc, self.Wo, self.Uf, self.Ui, self.Uc, self.Uo, \
        self.bias_f, self.bias_i, self.bias_c, self.bias_o = weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, weights):
        h_t = np.zeros((x.shape[0], x.shape[1], self.Wf.shape[1]))

        for t in range(x.shape[1]):
            x_t = x[:, t, :]

            f_t = self.sigmoid(np.dot(x_t, weights[0]) + np.dot(h_t[:, t-1, :], weights[4]) + weights[8])
            i_t = self.sigmoid(np.dot(x_t, weights[1]) + np.dot(h_t[:, t-1, :], weights[5]) + weights[9])
            c_t_hat = self.tanh(np.dot(x_t, weights[2]) + np.dot(h_t[:, t-1, :], weights[6]) + weights[10])
            o_t = self.sigmoid(np.dot(x_t, weights[3]) + np.dot(h_t[:, t-1, :], weights[7]) + weights[11])

            h_t[:, t, :] = f_t * h_t[:, t-1, :] + i_t * c_t_hat
            h_t[:, t, :] = o_t * self.tanh(h_t[:, t, :])

        return h_t

    def backward(self, x, h_t, y_true, weights):
        # Initialize gradients
        dWf, dWi, dWc, dWo, dUf, dUi, dUc, dUo = [np.zeros_like(w) for w in weights[:8]]
        dbias_f, dbias_i, dbias_c, dbias_o = [np.zeros_like(b) for b in weights[8:]]

        # Initialize the gradient of the hidden state and the cell state
        dh_t = np.zeros_like(h_t)
        dc_t = np.zeros_like(h_t)

        # Calculate the loss and initialize the regularization term
        loss = np.mean(0.5 * (h_t - y_true) ** 2)
        reg_loss = 0.5 * np.sum([np.sum(w ** 2) for w in weights[:8]])

        # Backward pass through time steps

        # Combine gradients and add regularization term
        gradients = [dWf, dWi, dWc, dWo, dUf, dUi, dUc, dUo, dbias_f, dbias_i, dbias_c, dbias_o]
        gradients = [grad + reg_loss * weights[i] for i, grad in enumerate(gradients)]

        return gradients, loss

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
                gradients, loss = self.backward(mini_batch_X, h_t, mini_batch_y, weights)

                # Update weights using mini-batch gradient descent
                weights = [w - learning_rate * grad for w, grad in zip(weights, gradients)]

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

    def export_model(self, filename):
        # Export the model using joblib
        joblib.dump(self, filename)

# Instantiate the SimpleLSTM model
input_size = features_reshaped.shape[2]
hidden_size = 50
simple_lstm_model = SimpleLSTM(input_size, hidden_size)

# Train the model
# Train the model and get the final weights
final_weights = simple_lstm_model.train(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

# Export the trained model
simple_lstm_model.export_model("simple_lstm_model.joblib")
