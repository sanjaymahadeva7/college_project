import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

model_file = "Bitcoin.keras"
# Load Model 
model = load_model(model_file)

st.set_page_config(layout="wide")  # Use wide layout for better visualization

# CSS for background image
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://t3.ftcdn.net/jpg/04/92/73/28/360_F_492732884_lIfYjpLvVxJI9aww7URzBBYdGmv54ynQ.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Title and Description
st.title('Bitcoin Price Prediction')
st.write("""
This application predicts future Bitcoin prices using a machine learning model. 
Upload your dataset or use the default one, visualize the data, and see the predictions.
""")

# Sidebar for user input
st.sidebar.subheader('Upload your CSV file')
uploaded_file = st.sidebar.file_uploader("Choose a file")

# Load data
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.sidebar.write("Using default dataset")
    data = pd.read_csv("bitcoin.csv", encoding='utf-8')

# Display data
st.subheader('Bitcoin Price Data')
st.write(data)

# Reverse data for time series
data_reversed = data.iloc[::-1].reset_index(drop=True)
data = data_reversed

# Plotting the data
st.subheader('Bitcoin Line Chart')
data_chart = data.drop(columns=['Open', 'High', 'Low', 'Volume', 'Market Cap', 'End', 'Start'])
fig = px.line(data_chart, x=data_chart.index, y='Close', title='Bitcoin Price Over Time')
st.plotly_chart(fig)

# Split data
train_data = data[:-500]
test_data = data[-500:]

train_values = train_data['Close'].values.reshape(-1, 1)
test_values = test_data['Close'].values.reshape(-1, 1)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

train_values_scaled = scaler.fit_transform(train_values)
test_values_scaled = scaler.transform(test_values)

train_data['Close_scaled'] = train_values_scaled
test_data['Close_scaled'] = test_values_scaled

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq_end = i + seq_length
        X.append(data[i:seq_end])
        y.append(data[seq_end])
    return np.array(X), np.array(y)

seq_length = 40

# Create sequences for training and testing sets
X_train, y_train = create_sequences(train_values_scaled, seq_length)
X_test, y_test = create_sequences(test_values_scaled, seq_length)

# Reshape X_train and X_test to be 3-dimensional (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], seq_length, 1)
X_test = X_test.reshape(X_test.shape[0], seq_length, 1)

# Predictions and errors
predictions_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)

y_test_inverse = scaler.inverse_transform(y_test)


st.subheader('Actual vs. Predicted Closing Prices')
fig_actual_vs_pred = go.Figure()
fig_actual_vs_pred.add_trace(go.Scatter(x=np.arange(len(y_test_inverse)), y=y_test_inverse.flatten(), mode='lines', name='Actual'))
fig_actual_vs_pred.add_trace(go.Scatter(x=np.arange(len(predictions)), y=predictions.flatten(), mode='lines', name='Predicted'))
fig_actual_vs_pred.update_layout(title='Actual vs. Predicted Closing Prices', xaxis_title='Time Steps', yaxis_title='Closing Price')
st.plotly_chart(fig_actual_vs_pred)

# Future prediction
st.sidebar.subheader('Future Prediction Settings')
future_days = st.sidebar.slider("Select number of future days to predict:", min_value=1, max_value=50, value=5)

future_predictions = []

initial_input = X_test[-1]

current_input = initial_input

for _ in range(future_days):
    current_input = current_input.reshape(1, seq_length, 1)
    next_pred = model.predict(current_input)
    future_predictions.append(next_pred[0, 0])
    current_input = np.roll(current_input, -1, axis=1)
    current_input[0, -1, 0] = next_pred

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions_inverse = scaler.inverse_transform(future_predictions).flatten()

future_indices = np.arange(len(y_test_inverse), len(y_test_inverse) + future_days)

# Plotting future predictions
st.subheader('Future Bitcoin Price Predictions')
fig_future = go.Figure()

# Adding shaded area for future predictions
fig_future.add_trace(go.Scatter(
    x=np.arange(len(y_test_inverse)), y=y_test_inverse.flatten(),
    mode='lines', name='Actual', line=dict(color='royalblue')
))
fig_future.add_trace(go.Scatter(
    x=future_indices, y=future_predictions_inverse,
    mode='lines+markers', name='Future Predictions',
    line=dict(color='firebrick', dash='dash'),
    marker=dict(size=6, color='firebrick', symbol='circle')
))

# Add shaded area
fig_future.add_trace(go.Scatter(
    x=np.concatenate([future_indices, future_indices[::-1]]),
    y=np.concatenate([future_predictions_inverse, np.zeros_like(future_predictions_inverse)]),
    fill='toself',
    fillcolor='rgba(255, 182, 193, 0.2)',
    line=dict(color='rgba(255, 182, 193, 0)')
))

# Annotations
fig_future.add_annotation(
    x=future_indices[-1], y=future_predictions_inverse[-1],
    text=f'Prediction: {future_predictions_inverse[-1]:.2f}',
    showarrow=True, arrowhead=1, ax=-40, ay=-40
)

fig_future.update_layout(
    title='Actual vs. Future Bitcoin Prices',
    xaxis_title='Time Steps',
    yaxis_title='Closing Price',
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white')
)
fig_future.update_xaxes(showgrid=False, color='white')
fig_future.update_yaxes(showgrid=False, color='white')

st.plotly_chart(fig_future)

# Footer
st.sidebar.markdown("Developed by [Sanjay](https://www.linkedin.com/in/sanjaymahadeva7/)")
