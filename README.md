# Bitcoin Price Prediction Using LSTM Neural Networks

## Project Overview
This project aims to predict Bitcoin prices using Long Short-Term Memory (LSTM) neural networks based on historical price data. It visualizes the predictions using Streamlit, providing an interactive and user-friendly interface for comparing predicted and actual prices, as well as forecasting future trends.

## Features
- Data preprocessing including normalization and sequencing
- Training an LSTM model on historical Bitcoin price data
- Predicting Bitcoin prices on test data
- Visualizing actual vs. predicted prices using Streamlit
- Forecasting future Bitcoin prices and visualizing the predictions

## Installation

### Prerequisites
- Python 3.6+
- Necessary Python libraries (see `requirements.txt`)

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/bitcoin-price-prediction.git
   ```

2. Navigate to the project directory:
   ```sh
   cd bitcoin-price-prediction
   ```

3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Ensure you have the historical Bitcoin price data CSV file named `bitcoin.csv` in the project directory.

2. Load the pre-trained LSTM model (ensure the model file `Bitcoin.keras` is in the correct path):
   ```python
   model = load_model(r"C:\Users\My PC\Downloads\Sanjay_final_year_project\Bitcoin.keras")
   ```

3. Run the Streamlit application:
   ```sh
   streamlit run app.py
   ```

4. Open your web browser and go to the local Streamlit URL provided in the terminal to interact with the application.

## Project Structure
```
bitcoin-price-prediction/
│
├── Bitcoin.keras              # Pre-trained LSTM model file
├── bitcoin.csv                # Historical Bitcoin price data
├── app.py                     # Main application file for Streamlit
├── requirements.txt           # Python dependencies
└── README.md                  # Project README file
```

## Methodology
1. **Data Collection:** Load historical Bitcoin price data from a CSV file.
2. **Data Preprocessing:** Reverse the data to maintain chronological order, drop non-essential columns, and normalize the data.
3. **Data Splitting:** Split the data into training and testing sets.
4. **Model Preparation:** Create sequences of 100 days for the LSTM model.
5. **Model Training:** Train the LSTM model using the training set.
6. **Prediction and Evaluation:** Evaluate the model on the test set and predict future Bitcoin prices.
7. **Visualization:** Use Streamlit to visualize the actual vs. predicted prices and future price predictions.

## Future Enhancements
- Incorporate additional features like trading volume and market sentiment for improved accuracy.
- Experiment with other machine learning models and compare performance.
- Implement real-time data fetching and prediction updates for dynamic forecasts.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
