import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os
import requests

plt.style.use("fivethirtyeight")

app = Flask(__name__)
model = load_model('stock_dl_model.h5')

API_KEY = '13a2210b41694e43817ab3a67dc4264b' # Optional for financial news

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        stock = request.form.get('stock', 'POWERGRID.NS')
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now()

        # Fetch stock data
        df = yf.download(stock, start=start, end=end)
        if df.empty:
            return render_template('index.html', error="Invalid stock symbol or no data available.")
        
        data_desc = df.describe()

        # Calculate EMAs
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Plot EMA 20 & 50
        plt.figure(figsize=(14,7))
        plt.plot(df.Close, label='Close Price', color='blue')
        plt.plot(ema20, label='EMA 20', color='orange')
        plt.plot(ema50, label='EMA 50', color='green')
        plt.legend()
        plot_path_ema_20_50 = 'static/ema_20_50.png'
        plt.savefig(plot_path_ema_20_50)
        plt.close()

        # Plot EMA 100 & 200
        plt.figure(figsize=(14,7))
        plt.plot(df.Close, label='Close Price', color='blue')
        plt.plot(ema100, label='EMA 100', color='purple')
        plt.plot(ema200, label='EMA 200', color='red')
        plt.legend()
        plot_path_ema_100_200 = 'static/ema_100_200.png'
        plt.savefig(plot_path_ema_100_200)
        plt.close()

        # Data Preprocessing
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)

        # Inverse Scaling
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plot Prediction vs Actual
        plt.figure(figsize=(14,7))
        plt.plot(y_test, label='Actual Price', color='blue')
        plt.plot(y_predicted.flatten(), label='Predicted Price', color='red')
        plt.legend()
        plot_path_prediction = 'static/prediction_vs_actual.png'
        plt.savefig(plot_path_prediction)
        plt.close()

        # Plotly Live Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Original Price'))
        fig.add_trace(go.Scatter(y=y_predicted.flatten(), mode='lines', name='Predicted Price'))
        graph_json = fig.to_json()

        # Fetch Financial News
        news_data = []
        if API_KEY:
            try:
                response = requests.get(f'https://newsapi.org/v2/everything?q={stock}&apiKey={API_KEY}')
                news_data = response.json().get('articles', [])[:5]
            except Exception as e:
                print('News fetch error:', e)

        return render_template('index.html', 
                                plot_path_ema_20_50=plot_path_ema_20_50,
                                plot_path_ema_100_200=plot_path_ema_100_200,
                                plot_path_prediction=plot_path_prediction,
                                graph_json=graph_json,
                                data_desc=data_desc.to_html(classes='table table-bordered'),
                                news_data=news_data)
    except Exception as e:
        return render_template('index.html', error=str(e))


@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join('static', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return 'File not found', 404

if __name__ == '__main__':
    app.run(debug=True)
