
# Stock Price Prediction Using LSTM (Amazon)

## Project Overview
This project focuses on predicting the future stock prices of Amazon using a Long Short-Term Memory (LSTM) neural network. Stock market data is sequential in nature and influenced by historical trends. LSTM, a type of Recurrent Neural Network (RNN), is well suited for time-series forecasting problems such as stock price prediction.

The model is trained using historical Amazon stock data and predicts future closing prices based on past price movements.

---

## Objectives
- To analyze historical Amazon stock price data
- To preprocess and normalize time-series data
- To build and train an LSTM-based deep learning model
- To predict future stock closing prices
- To evaluate model performance using error metrics

---

## Dataset Description
Dataset Name: Amazon Stock Price Dataset  
Source: Kaggle  
File Name: Amazon.csv  

### Features
- Date: Trading date
- Open: Opening price of the stock
- High: Highest price of the day
- Low: Lowest price of the day
- Close: Closing price of the stock
- Volume: Number of shares traded

Only the Close price is used for prediction as it represents the final trading value for the day.

---

## Technologies and Libraries Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras

---

## Methodology
1. Load and inspect the dataset
2. Convert the Date column to datetime format
3. Sort the data in chronological order
4. Extract the Close price column
5. Normalize the data using MinMaxScaler
6. Create time-series sequences using the previous 60 days
7. Split the data into training and testing sets
8. Build an LSTM neural network model
9. Train the model using historical data
10. Predict stock prices on test data
11. Evaluate the model using RMSE
12. Visualize actual vs predicted prices

---

## Model Architecture
- LSTM Layer with 50 units (return sequences enabled)
- LSTM Layer with 50 units
- Dense output layer with 1 unit

Optimizer: Adam  
Loss Function: Mean Squared Error

---

## Evaluation Metric
Root Mean Squared Error (RMSE)

RMSE measures the average prediction error. Lower RMSE values indicate better model accuracy.

---

## Results
The LSTM model successfully captures temporal dependencies in Amazon stock prices and produces predictions that closely follow actual closing price trends.

---

## Conclusion
This project demonstrates that LSTM neural networks are effective for stock price prediction. By learning from historical closing prices, the model identifies patterns and trends in financial time-series data.

---

## Future Enhancements
- Predict stock prices for the next 30 days
- Include multiple features such as Open, High, Low, and Volume
- Compare performance with Linear Regression and ARIMA models
- Deploy the model using Streamlit or Flask
- Integrate real-time stock market data

---

## How to Run the Project
1. Upload Amazon.csv to Google Colab
2. Install required Python libraries
3. Run all notebook cells sequentially
4. Train the LSTM model
5. View predictions and evaluation results
