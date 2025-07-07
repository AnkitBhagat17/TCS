import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Page configuration
st.set_page_config(
    page_title="TCS Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('TCS_Data.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df[df['volume'] > 0]
    df = df.sort_values('date').reset_index(drop=True)

    # Metrics
    df['daily_return'] = df['close'].pct_change() * 100
    df['hl_spread'] = df['high'] - df['low']
    df['MA_20'] = df['close'].rolling(window=20).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    df['MA_200'] = df['close'].rolling(window=200).mean()
    df['volatility'] = df['daily_return'].rolling(window=30).std() * 100
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.day_name()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    return df

# Load data
df = load_data()
filtered_df = df.copy()

# Sidebar
st.sidebar.title("📊 TCS Stock Dashboard")
st.sidebar.markdown("---")

min_date = df['date'].min()
max_date = df['date'].max()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

filtered_df = df[(df['date'] >= pd.to_datetime(start_date)) &
                 (df['date'] <= pd.to_datetime(end_date))]

# Check for duplicate dates and handle them
if filtered_df['date'].duplicated().any():
    # st.warning("Duplicate dates found in the data. They will be aggregated.")
    filtered_df = filtered_df.groupby('date').agg({
        'open': 'first',  # or 'mean', 'sum', etc.
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'daily_return': 'mean',
        'hl_spread': 'mean',
        'MA_20': 'mean',
        'MA_50': 'mean',
        'MA_200': 'mean',
        'volatility': 'mean',
        'year': 'first',
        'month': 'first',
        'weekday': 'first',
        'vwap': 'mean'
    }).reset_index()

chart_options = st.sidebar.multiselect(
    "Select Charts to Display:",
    ["Price Trend", "Volume Analysis", "Moving Averages", "Volatility",
     "Price Distribution", "Returns Distribution", "Monthly Heatmap",
     "Price vs Volume", "Recent Candlestick",
     "Cumulative Returns", "Correlation Heatmap", "Weekday Analysis", "VWAP"],
    default=["Price Trend", "Volume Analysis", "Moving Averages"]
)

# Title
st.title("🏢 TCS Stock Analysis Dashboard")
st.markdown("---")

# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Current Price", f"₹{filtered_df['close'].iloc[-1]:.2f}")
with col2:
    st.metric("Daily Change", f"{filtered_df['daily_return'].iloc[-1]:.2f}%")
with col3:
    st.metric("Volume", f"{filtered_df['volume'].iloc[-1]:,.0f}")
with col4:
    st.metric("52W High", f"₹{filtered_df['high'].max():.2f}")
with col5:
    st.metric("52W Low", f"₹{filtered_df['low'].min():.2f}")

st.markdown("---")

# 📊 Charts
if "Price Trend" in chart_options:
    st.subheader("📈 Stock Price Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['close'],
                             mode='lines', name='Close Price'))
    fig.update_layout(title="TCS Stock Price Over Time", xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

if "Volume Analysis" in chart_options:
    st.subheader("📊 Trading Volume")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['volume'],
                             mode='lines', name='Volume', line=dict(color='orange')))
    fig.update_layout(title="Volume Over Time", xaxis_title="Date", yaxis_title="Volume")
    st.plotly_chart(fig, use_container_width=True)

if "Moving Averages" in chart_options:
    st.subheader("📉 Price with Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['MA_20'], name='MA 20'))
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['MA_50'], name='MA 50'))
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['MA_200'], name='MA 200'))
    fig.update_layout(title="Moving Averages", xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

if "Volatility" in chart_options:
    st.subheader("📊 Volatility")
    fig = px.line(filtered_df, x='date', y='volatility', title="30-Day Rolling Volatility (%)")
    st.plotly_chart(fig, use_container_width=True)

if "Price Distribution" in chart_options:
    st.subheader("📊 Price Distribution")
    fig = px.histogram(filtered_df, x='close', nbins=50)
    fig.update_layout(xaxis_title="Close Price", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

if "Returns Distribution" in chart_options:
    st.subheader("📊 Daily Returns Distribution")
    fig = px.histogram(filtered_df.dropna(subset=['daily_return']), x='daily_return', nbins=50)
    fig.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

if "Monthly Heatmap" in chart_options:
    st.subheader("🔥 Monthly Returns Heatmap")
    monthly_returns = filtered_df.groupby(['year', 'month'])['daily_return'].sum().unstack()
    fig = px.imshow(monthly_returns, color_continuous_scale='RdYlGn', aspect='auto')
    fig.update_layout(title="Monthly Returns Heatmap", xaxis_title="Month", yaxis_title="Year")
    st.plotly_chart(fig, use_container_width=True)

if "Price vs Volume" in chart_options:
    st.subheader("📊 Price vs Volume Correlation")
    fig = px.scatter(filtered_df, x='close', y='volume', opacity=0.5)
    fig.update_layout(xaxis_title="Close Price", yaxis_title="Volume")
    st.plotly_chart(fig, use_container_width=True)

if "Recent Candlestick" in chart_options:
    st.subheader("🕯️ Candlestick Chart")
    recent = filtered_df.tail(100)
    fig = go.Figure(data=[go.Candlestick(
        x=recent['date'], open=recent['open'], high=recent['high'],
        low=recent['low'], close=recent['close'])])
    fig.update_layout(title="Recent Candlestick", height=400)
    st.plotly_chart(fig, use_container_width=True)

if "Cumulative Returns" in chart_options:
    st.subheader("📈 Cumulative Returns")
    cum_return = (1 + filtered_df['daily_return'] / 100).cumprod()
    fig = px.line(x=filtered_df['date'], y=cum_return, title="Cumulative Return Over Time")
    fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return")
    st.plotly_chart(fig, use_container_width=True)

if "Correlation Heatmap" in chart_options:
    st.subheader("🌀 Correlation Heatmap")
    corr = filtered_df[['open', 'high', 'low', 'close', 'volume', 'daily_return']].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

if "Weekday Analysis" in chart_options:
    st.subheader("📅 Weekday Average Returns")
    weekday_avg = filtered_df.groupby('weekday')['daily_return'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    fig = px.bar(x=weekday_avg.index, y=weekday_avg.values,
                 labels={'x': 'Weekday', 'y': 'Average Return (%)'},
                 title="Average Return by Weekday")
    st.plotly_chart(fig, use_container_width=True)

if "VWAP" in chart_options:
    st.subheader("💹 VWAP vs Close Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['close'], name="Close"))
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['vwap'], name="VWAP"))
    fig.update_layout(title="VWAP vs Close Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# Summary
st.markdown("---")
st.subheader("📋 Data Summary")
col1, col2 = st.columns(2)

with col1:
    st.write(f"Total Records: {len(filtered_df):,}")
    st.write(f"Date Range: {filtered_df['date'].min().strftime('%Y-%m-%d')} → {filtered_df['date'].max().strftime('%Y-%m-%d')}")
    st.write(f"Average Close: ₹{filtered_df['close'].mean():.2f}")
    st.write(f"Average Volume: {filtered_df['volume'].mean():,.0f}")

with col2:
    st.write(f"Last Close: ₹{filtered_df['close'].iloc[-1]:.2f}")
    st.write(f"1-Day Return: {filtered_df['daily_return'].iloc[-1]:.2f}%")
    st.write(f"Volatility (30D): {filtered_df['volatility'].iloc[-1]:.2f}%")
    st.write(f"Max Price: ₹{filtered_df['high'].max():.2f}")

# Forecasting Models
st.markdown("---")
st.subheader("📈 Forecasting Models")

model_type = st.selectbox("Select Forecasting Model",
                          ["Prophet", "ARIMA", "SARIMA"])
periods_input = st.number_input("Days to Forecast", min_value=7, max_value=365, value=30)

# Prepare data
ts = filtered_df[['date', 'close']].set_index('date')

# Check for duplicates and handle them
if ts.index.duplicated().any():
    st.warning("Duplicate dates found in the data. They will be aggregated.")
    ts = ts.groupby(ts.index).agg({
        'close': 'last'  # Keep the last close price for duplicate dates
    })

# Resample to daily frequency and fill missing values
ts = ts.asfreq('D').fillna(method='ffill')

if model_type == "Prophet":
    st.markdown("### 🔮 Prophet Forecast")
    dfp = ts.reset_index().rename(columns={'date':'ds','close':'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods_input)
    forecast = m.predict(future)
    st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
    st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(periods_input))

elif model_type == "ARIMA":
    st.markdown("### 📊 ARIMA Forecast")
    model = ARIMA(ts['close'], order=(1,1,1))  # Basic ARIMA parameters
    try:
        model_fit = model.fit()
        future = model_fit.forecast(steps=periods_input)
        idx = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=periods_input, freq='D')
        fig = go.Figure([go.Scatter(x=ts.index, y=ts['close'], name='Historical'),
                         go.Scatter(x=idx, y=future, name='ARIMA Forecast')])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error fitting ARIMA model: {str(e)}")
        st.info("Try adjusting the ARIMA parameters (p,d,q) for better results")

elif model_type == "SARIMA":
    st.markdown("### 📉 SARIMA Forecast")
    try:
        mod = sm.tsa.statespace.SARIMAX(ts['close'], order=(1,1,1), seasonal_order=(1,1,1,12))
        res = mod.fit(disp=False)
        pred = res.get_forecast(steps=periods_input)
        fc = pred.predicted_mean
        idx = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=periods_input, freq='D')
        fig = go.Figure([go.Scatter(x=ts.index, y=ts['close'], name='Historical'),
                         go.Scatter(x=idx, y=fc, name='SARIMA Forecast')])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error fitting SARIMA model: {str(e)}")
        st.info("Try adjusting the seasonal parameters for better results")

# Footer
st.markdown("---")
st.markdown("*📊 Dashboard created using Streamlit for TCS Stock Analysis*")
