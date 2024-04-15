# from flask import Flask, request, render_template, jsonify
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
#
# # Load the saved LSTM model
# model = tf.keras.models.load_model('multi_feature_lstm_model.h5')
#
# # Load the feature scaler
# scaler_features = MinMaxScaler()
# scaler_features = scaler_features.fit(pd.read_csv('./static/combined_file.csv')[['LAT', 'LON', 'PRECTOT', 'PS', 'QV2M', 'T2MWET', 'TS', 'WS50M', 'WS10M', 'WS50M_RANGE', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS10M_MAX', 'WS10M_MIN', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']])
#
# app = Flask(__name__)
#
# # @app.route('/')
# # def index():
# #     return render_template('test.html')
# @app.route('/index')
# def index():
#     return render_template('index.html')
#
# @app.route('/dashboard')
# def dash():
#     return render_template('dashboard.html')
#
# @app.route('/graph')
# def graph():
#     return render_template('graph.html')
#
# @app.route('/login')
# def login():
#     return render_template('login.html')
#
# @app.route('/dash')
# def dash():
#     return render_template('dash.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the input feature values from the request
#     input_data = request.form.to_dict()
#
#     print("Input data received:", input_data)  # Debugging statement
#
#     # Preprocess the input data
#     input_features = [float(input_data[feature]) for feature in ['LAT', 'LON', 'PRECTOT', 'PS', 'QV2M', 'T2MWET', 'TS', 'WS50M', 'WS10M', 'WS50M_RANGE', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS10M_MAX', 'WS10M_MIN', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']]
#     input_features = scaler_features.transform([input_features])
#     input_features = np.reshape(input_features, (1, 1, input_features.shape[1]))
#
#     print("Input features after preprocessing:", input_features)  # Debugging statement
#
#     # Make the prediction using the loaded model
#     prediction = model.predict(input_features)
#
#     print("Raw prediction from the model:", prediction)  # Debugging statement
#
#     # Inverse transform the prediction
#     scaler_target = MinMaxScaler()
#     scaler_target = scaler_target.fit(pd.read_csv('./static/combined_file.csv')[['T2M', 'RH2M', 'WS10M_RANGE']])
#     prediction = scaler_target.inverse_transform(prediction)
#     prediction = prediction.tolist()
#
#     print("Final prediction:", prediction)  # Debugging statement
#
#     # Return the prediction as a JSON response
#     return jsonify({
#         'T2M': prediction[0][0],
#         'RH2M': prediction[0][1],
#         'WS10M_RANGE': prediction[0][2]
#     })
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the saved LSTM model
model = tf.keras.models.load_model('multi_feature_lstm_model.h5')

# Load the feature scaler
scaler_features = MinMaxScaler()
scaler_features = scaler_features.fit(pd.read_csv('./static/combined_file.csv')[['LAT', 'LON', 'PRECTOT', 'PS', 'QV2M', 'T2MWET', 'TS', 'WS50M', 'WS10M', 'WS50M_RANGE', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS10M_MAX', 'WS10M_MIN', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/index')
def index_page():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/dash')
def dash_page():
    return render_template('dash.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input feature values from the request
    input_data = request.form.to_dict()

    print("Input data received:", input_data)  # Debugging statement

    # Preprocess the input data
    input_features = [float(input_data[feature]) for feature in ['LAT', 'LON', 'PRECTOT', 'PS', 'QV2M', 'T2MWET', 'TS', 'WS50M', 'WS10M', 'WS50M_RANGE', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS10M_MAX', 'WS10M_MIN', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']]
    input_features = scaler_features.transform([input_features])
    input_features = np.reshape(input_features, (1, 1, input_features.shape[1]))

    print("Input features after preprocessing:", input_features)  # Debugging statement

    # Make the prediction using the loaded model
    prediction = model.predict(input_features)

    print("Raw prediction from the model:", prediction)  # Debugging statement

    # Inverse transform the prediction
    scaler_target = MinMaxScaler()
    scaler_target = scaler_target.fit(pd.read_csv('./static/combined_file.csv')[['T2M', 'RH2M', 'WS10M_RANGE']])
    prediction = scaler_target.inverse_transform(prediction)
    prediction = prediction.tolist()

    print("Final prediction:", prediction)  # Debugging statement

    # Return the prediction as a JSON response
    return jsonify({
        'T2M': prediction[0][0],
        'RH2M': prediction[0][1],
        'WS10M_RANGE': prediction[0][2]
    })

@app.route('/new_predict', methods=['POST'])
def new_location_predict():
    # Get the input feature values from the request
    input_data = request.form.to_dict()

    # Ensure keys are in lowercase
    input_data = {key.lower(): value for key, value in input_data.items()}

    # Access latitude and longitude values
    lat = float(input_data.get('lat'))
    lon = float(input_data.get('lon'))

    # Generate other input features with a value of 15
    other_features = [15] * 16  # 8 other features with value 15

    # Combine latitude, longitude, and other features
    input_features = [lat, lon] + other_features

    # Preprocess the input data
    input_features = scaler_features.transform([input_features])
    input_features = np.reshape(input_features, (1, 1, input_features.shape[1]))

    # Make the prediction using the loaded model
    prediction = model.predict(input_features)

    # Inverse transform the prediction
    scaler_target = MinMaxScaler()
    scaler_target = scaler_target.fit(pd.read_csv(r"C:\Users\Lenovo\Contacts\finale.csv")['T2M'].values.reshape(-1, 1))
    prediction = scaler_target.inverse_transform(prediction)

    # Return the prediction as a JSON response
    return str(prediction[0][0])



if __name__ == '__main__':
    app.run(debug=True)
