The quick development in the area of artificial intelligence and machine learning have greatly impacted the financial markets and made it possible to predict stock prices more accurately. Stock forecasting using conventional methods depends on historical patterns, fundamental analysis, and professionals’ recommendations, but all these techniques fail to respond to market volatility and intricate non-linear interdependencies between variables. Machine learning models provide us with a data-driven solution by utilizing big datasets, detecting hidden patterns, and making live adjustments with inputs in real-time.
This research introduces a stock price prediction model that combines machine learning models with a Flask-based web application for an easy accessibility to the users. The forecast model incorporates historical data of the stock market retrieved from the Yahoo Finance API to learn predictive models using methods like the Long Short-Term Memory (LSTM) networks and Random Forest regression to enhance the accuracy of the forecasting. The Flask framework is used to develop an interactive web interface where users can enter the stock ticker symbols and get predictions in real-time.

I.	PROPOSED METHODOLOGY
a.	System Overview
The proposed stock price prediction is a data-driven approach that leverages the Long  Short-Term Memory (LSTM) networks for analyzing and forecasting stock prices. The architecture is designed to capture temporal dependencies in financial data, ensuring reliable predictions. Five stages compose the system in detail : Data Collection, Data Preprocessing, Feature Engineering, Model Design and Training, and Prediction and Evaluation.
![image](https://github.com/user-attachments/assets/e40f3844-fdde-4837-a24e-fe9c6762ee7e)
b.	Data Collection
The stock price data is sourced using the Yahoo Finance API via the yfinance Python library. This data includes essential financial metrics such as Open, High, Low, Close and Volume prices over a selected period of time. The API provides historical data, which is critical for training the LSTM model. Additionally, the parameters like the stock ticker symbol, date range, and data frequency are specified to ensure accurate and relevant data collection.
![image](https://github.com/user-attachments/assets/f7176543-4741-4257-9335-f9f755da2906)
c.  Data Preprocessing
Preprocessing is an essential step to clean and normalize data for improved model training. The following procedures are applied:
	Handling Misiing Values:
Any missing or corrupted data points are identified and handled using forward or backward filling methods to ensure continuity in the time series.
	Normalization:
The data is normalized using Min-Max Scaling to restrict values between 0 and 1. This normalization accelerates model convergence and prevents gradient explosion.
X_Norm=(X+X_min)/(X_max-X_min )
  Feature Extraction:
The primary feature used is the Closing Price. To capture time-based dependencies, additional time seiries features like Moving Averages and Relative Strength Index (RSI) can be added.
d.	Model Architecture
The center of the system is a deep learning LSTM Network. LSTMs are suited to be used for time series prediction since they have the capacity to identify long term relationships utilizig gated memory cells. The model contains the folloeing features:
	Input Layer: This accepts squential data with the shape (60,1) representing 60 days of stock prices.
	LSTM Layers: We have used four LSTM layers with 50 units each.
	Dropout Layers: These are applied with a dropout rate of 20% to prevent overfitting. 
	Dense Layers: In this a single neuron is used in the output layer with a linear activation function to forecast the next stock price.
	Loss Function: Mean Squared Error (MSE) is utilized to measure the performance of the model.
	Optimizer: Adam Optimizer is due to its adaptive learning rate feature.
e.	Model Training
The LSTM model is trained using the preprocessed data. Training is conducted for 50 epochs with a batch size of 64. The learning process is monitored using the loss function, and adjustments are made using the early stopping techniques if necessary.
	Evaluation Metrics
The model is evaluated using the following metrics:
	Mean Squared Error (MSE): 
It measures the average squared difference between the actual and the predicted stock prices.
MSE=  1/n ∑_(i=1)^n▒〖(yi-y^i)^2 〗
	Root Mean Squared Error (RMSE):
RSME= √MSE

      Model	                        MSE	     MAE    R^2 Score.
Linear  Regression	              0.00045 	0.0158	 0.92
Support  Vector Machine (SVM)   	0.00051 	0.0172	 0.89
Random  Forest	                  0.00039	  0.0143	 0.94
LSTM    Network	                  0.00036 	0.0137	 0.96
