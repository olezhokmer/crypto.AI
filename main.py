import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify

app = Flask(__name__)

def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train, scaled_data, scaler

def build_model(x_train):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(tf.keras.layers.LSTM(50, return_sequences=False))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    historical_data = data.get('prices', data)
    if not historical_data:
        return jsonify({"error": "No price data provided"}), 400

    df = pd.DataFrame(historical_data)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df[['time', 'price']]
    df.rename(columns={'price': 'close'}, inplace=True)

    x_train, y_train, scaled_data, scaler = prepare_data(df)

    if len(x_train) < 1:
        return jsonify({"error": "Not enough data to create training samples."}), 400

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = build_model(x_train)
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    predictions = []
    last_60_days = scaled_data[-60:]
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))

    for _ in range(100):
        predicted_price = model.predict(last_60_days)
        predictions.append(predicted_price[0][0])
        last_60_days = np.append(last_60_days[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    response = [
        {
            'time': (df['time'].iloc[-1] + pd.Timedelta(hours=i + 1)).timestamp() * 1000,
            'price': float(price[0]) 
        }
        for i, price in enumerate(predicted_prices)
    ]

    return jsonify({"predictedPrices": response})


if __name__ == '__main__':
    app.run(debug=True)
