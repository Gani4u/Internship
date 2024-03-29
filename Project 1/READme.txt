Stock Price Prediction with LSTM

This project utilizes Long Short-Term Memory (LSTM) neural networks to predict stock prices based on historical data. The code preprocesses the data, creates a time series dataset, defines and trains an LSTM model using Keras, and then makes predictions. It also includes functions for data visualization and evaluation.

Key Features:

Data preprocessing and normalization
LSTM model architecture design and training
Visualization of actual vs predicted stock prices
Evaluation of model performance using mean squared error

Dependencies:

NumPy
Pandas
Matplotlib
scikit-learn
Keras
Instructions:

Run the script stock_price_prediction.py
Ensure the CSV file EOD-AAPL_in.csv is in the same directory.
Adjust parameters such as epochs, batch size, and model architecture as needed.
Explore the plotted results to understand the model's predictions.