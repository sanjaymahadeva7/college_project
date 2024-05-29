import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from keras.models import load_model

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
train_data = data_chart[:-100]
test_data = data_chart[-200:]

# Scaling data
scaler = MinMaxScaler(feature_range=(0,1))
train_data_scale = scaler.fit_transform(train_data)
test_data_scale = scaler.transform(test_data)

# Preparing data for prediction
base_days = 100
x = []
y = []
for i in range(base_days, len(test_data_scale)):
    x.append(test_data_scale[i-base_days:i])
    y.append(test_data_scale[i, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

st.write(f"x shape: {x.shape}")
st.write(f"y shape: {y.shape}")

# Predicting
st.subheader("Predicted vs Original Prices")
pred = model.predict(x)
pred = scaler.inverse_transform(pred)
preds = pred.reshape(-1, 1)
ys = scaler.inverse_transform(y.reshape(-1, 1))

# Creating DataFrames for plotting
preds_df = pd.DataFrame(preds, columns=["Predicted Price"])
ys_df = pd.DataFrame(ys, columns=["Original Price"])

# Plotting predictions vs original prices
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=preds_df.index, y=preds_df["Predicted Price"], mode='lines', name='Predicted'))
fig_pred.add_trace(go.Scatter(x=ys_df.index, y=ys_df["Original Price"], mode='lines', name='Original'))
fig_pred.update_layout(title='Predicted vs Original Bitcoin Prices', xaxis_title='Days', yaxis_title='Price')
st.plotly_chart(fig_pred)

# Future prediction
st.sidebar.subheader('Future Prediction Settings')
future_days = st.sidebar.slider("Select number of future days to predict:", min_value=1, max_value=30, value=5)

m = y
z = []
for i in range(base_days, len(m) + future_days):
    m = m.reshape(-1, 1)
    inter = [m[-base_days:, 0]]
    inter = np.array(inter)
    inter = np.reshape(inter, (inter.shape[0], inter.shape[1], 1))
    pred = model.predict(inter)
    z = np.append(z, pred)

# Display future predictions
st.subheader("Future Days Prediction")
z = np.array(z)
z = scaler.inverse_transform(z.reshape(-1, 1))

future_df = pd.DataFrame(z, columns=["Future Predicted Price"])
fig_future = px.line(future_df, y='Future Predicted Price', title='Future Bitcoin Prices Prediction')
st.plotly_chart(fig_future)

# Footer
st.sidebar.markdown("Developed by [sanjay](https://www.linkedin.com/in/sanjaymahadeva7/)")
