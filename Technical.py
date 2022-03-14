# Imports
import streamlit as st
import yfinance as yf
import datetime as date
import pandas as pd
import numpy as np
import ta
import time
import hvplot.pandas
import quandl
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import init_notebook_mode
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import bt
import talib
from prophet import Prophet
import altair as alt
import tensorflow as tf
import math
import json
import requests
import os
from wordcloud import WordCloud
from collections import Counter
from nltk import ngrams
import itertools
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import matplotlib as mpl


# Call class module and import workbooks
from CreatePages import CreateMultiplePages
import Technical
import Sentiment

# Define app function
def app():

    # Use Seaborn plotting style
    plt.style.use('seaborn')

    # Sidebar text input for stock ticker
    stock_selection = st.sidebar.text_input("Enter a valid stock ticker:", "TSLA")

    # Search bar
    search_app = st.sidebar.button("Search")

    # Date input side bar
    date_input = st.sidebar.date_input("Select Date Range (max three years)", 
                                            value = date.datetime.now() - date.timedelta(days = 365 * 3))
    # Create columns
    col1, col2 = st.columns(2)

    # Initiate first column
    with col1:

        # Title
        st.subheader("Automated Stock Analyser: Technical Analysis")

        # Introduction
        st.markdown("""Self-service stock analyser using a combination of machine learning and technical indicators to assist in the research
        process for selecting stocks.""")

        # Retrieve data
        asset_data = yf.download(stock_selection, 
                                    start = date_input, 
                                    end = date.datetime.now())

        # Apply moving averages to the dataframe
        asset_data["SMA50"] = asset_data["Close"].rolling(window=50).mean()
        asset_data["SMA100"] = asset_data["Close"].rolling(window=100).mean()

        # Insert subheader
        st.subheader("Technical Analysis")

        # Technical analysis multi-select
        indicator_options = st.selectbox(
                                        'Select Technical Indicators you want to overlay over your chart:',
                                        ('Fibonacci Retracement', 'Exponential Moving Average (EMA)'))

        # Exponential moving average indicator
        if indicator_options == 'Exponential Moving Average (EMA)':

            # Include features in reporting tool    
            with st.spinner("Data loading..."):
                

                # Include digital features
                time.sleep(1)
                # Initiate try

                # If stock selection is valid
                try:

                    # Calculate the exponential moving averages of the closing data
                    EMA_short = talib.EMA(asset_data['Close'], timeperiod=12).to_frame()

                    # Rename column
                    EMA_short = EMA_short.rename(columns={0: 'Close'})

                    # Calculate the exponential moving averages of the closing data
                    EMA_long = talib.EMA(asset_data['Close'], timeperiod=50).to_frame()

                    # Rename column
                    EMA_long = EMA_long.rename(columns={0: 'Close'})

                    # Copy EMA dataframe
                    signal = EMA_long.copy()

                    # Hold positions where data is null
                    signal[EMA_long.isnull()] = 0

                    # Where short EMA is greater than long EMA
                    signal[EMA_short > EMA_long] = 1

                    # Where long EMA is greater than short EMA
                    signal[EMA_short < EMA_long] = -1

                    # Extract only data where a buy and sell decision are required
                    transition = signal[signal['Close'].diff()!=0]

                    # Capture buy signals
                    buy_signal = transition[transition['Close'] == 1]

                    # Capture sell signals
                    sell_signal = transition[transition['Close'] == -1]

                    # Store index in variable
                    long_index = buy_signal.index

                    # Capture index of selling positions using buy index created above
                    buy_position = asset_data[asset_data.index.isin(long_index)]

                    # Store index in variable
                    short_index = sell_signal.index

                    # Capture index of selling positions using short index created above
                    sell_position = asset_data[asset_data.index.isin(short_index)]

                    # Initiate figure
                    ema_fig = go.Figure()

                    # Add candlestick charting
                    ema_fig.add_trace(
                            go.Candlestick(x=asset_data.index,
                                    open=asset_data['Open'],
                                    high=asset_data['High'],
                                    low=asset_data['Low'],
                                    close=asset_data['Close'],
                                    name="Stock Prices"
                                        )            
                    )

                    # Include EMA 50 to plot
                    ema_fig.add_trace(
                            go.Scatter(
                                x=asset_data.index,
                                y=EMA_long['Close'],
                                name="EMA 50"
                            )
                    )

                    # Include EMA short to plot
                    ema_fig.add_trace(
                            go.Scatter(
                                x=asset_data.index,
                                y=EMA_short['Close'],
                                name = "EMA 12"
                            )
                    )

                    # Add in buy signals to chart
                    ema_fig.add_trace(
                            go.Scatter(
                                x=buy_position.index,
                                y=buy_position['Close'], 
                                name="Buy Signal",
                                marker=dict(color="#511CFB", size=15),
                                mode="markers",
                                marker_symbol="triangle-up"
                            )
                    )

                    # Add in sell signals to chart
                    ema_fig.add_trace(
                            go.Scatter(
                                x=sell_position.index,
                                y=sell_position['Close'], 
                                name="Sell Signal",
                                marker=dict(color="#750086", size=15),
                                mode="markers",
                                marker_symbol="triangle-down"
                            )
                    )

                    # Update the layout with charting, axis and title
                    ema_fig.update_layout(
                        xaxis_rangeslider_visible=False,
                        title="Daily Close (" + stock_selection.upper() + ") Prices",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)"
                    )
                    
                    # Show chart
                    st.plotly_chart(ema_fig)

                # Run exception if error occurs to prompt the user to enter a valid ticker  
                except:
                    st.error("""Please try entering a valid ticker.""")

        # Exponential moving average indicator
        elif indicator_options == 'Fibonacci Retracement':

            # Include features in reporting tool    
            with st.spinner("Data loading..."):

                # Include digital features
                time.sleep(1)
                # Initiate try

                # If stock selection is valid
                try:

                    maximum_price = asset_data['Close'].max()
                    minimum_price = asset_data['Close'].min()

                    # Calculate the Fibonacci Levels

                    difference = maximum_price - minimum_price
                    first_level = maximum_price - difference * 0.236
                    second_level = maximum_price - difference * 0.382
                    third_level = maximum_price - difference * 0.5
                    fourth_level = maximum_price - difference * 0.618

                    # Calculate the Short Term Exponential Moving Average
                    ShortEMA = asset_data.Close.ewm(span=12, adjust=False).mean()

                    #Calculate the Long Term Exponential Moving Average
                    LongEMA = asset_data.Close.ewm(span=26, adjust=False).mean()

                    #Calculate the Moving Average Convergence / Divergence (MACD)
                    MACD = ShortEMA - LongEMA

                    #Calculate the Signal Line: This is a 9 period average of the MACD line itself
                    signal = MACD.ewm(span=9, adjust = False).mean()
                    asset_data['Signal Line'] = signal
                    asset_data['MACD'] = MACD
                    def getLevels(price):
                        if price >= first_level:
                            return (maximum_price, first_level)
                        elif price >= second_level:
                            return (first_level, second_level)
                        elif price >= third_level:
                            return (second_level, third_level)
                        elif price >= fourth_level:
                            return (third_level, fourth_level)
                        else:
                            return (fourth_level, minimum_price)

                    
                    def strategy(df):
                        buy_list = []
                        sell_list = []
                        flag = 0
                        last_buy_price = 0
                        
                        # Loop through the data set
                        for i in range(0, df.shape[0]):
                            price = df['Close'][i]
                            #if this is the first data point within the data set, then get the level above and below it.
                            if i == 0:
                                upper_lvl, lower_lvl = getLevels(price)
                                buy_list.append(np.nan)
                                sell_list.append(np.nan)
                                
                            # Else if the current price is greater than or equal to the upper_lvl, or less than or equal to the lower_lvl, then we know the price has 'hit' or crossed a new Fibonacci Level
                            elif (price >= upper_lvl) | (price <= lower_lvl):
                                
                                # Check to see if the MACD line crossed above or below the signal line
                                if df['Signal Line'][i] > df['MACD'][i] and flag == 0:
                                    last_buy_price = price
                                    buy_list.append(price)
                                    sell_list.append(np.nan)
                                    # Set the flag to 1 to signal that the share was bought 
                                    flag = 1
                                elif df['Signal Line'][i] < df['MACD'][i] and flag == 1 and price >= last_buy_price:
                                    buy_list.append(np.nan)
                                    sell_list.append(price)
                                    #Set the flag to 0 to signal that the share was sold
                                    flag = 0
                                else:
                                    buy_list.append(np.nan)
                                    sell_list.append(np.nan)
                                    
                            else:
                                buy_list.append(np.nan)
                                sell_list.append(np.nan)
                            
                            #Update the new level
                            upper_lvl, lower_lvl = getLevels(price)
                            
                        return buy_list, sell_list

                        # Create buy and sell columns
                    buy, sell = strategy(asset_data)
                    asset_data['Buy_Signal_Price'] = buy
                    asset_data['Sell_Signal_Price'] = sell

                    fib = go.Figure()
                    fib.add_trace(go.Scatter(x=asset_data.index,y= asset_data["Close"], name = 'Price ($USD)'))

                    fib.add_trace(go.Scatter(x=asset_data.index, y=asset_data['Buy_Signal_Price'], name = "Buy Signal", marker=dict(color="green", size=15), mode = 'markers', marker_symbol = 'triangle-up'))
                    fib.add_trace(go.Scatter(x=asset_data.index, y=asset_data['Sell_Signal_Price'], name="Sell Signal", marker=dict(color="red", size=15), mode = 'markers', marker_symbol = 'triangle-down'))

                    fib.add_hline(maximum_price, line_color='red')
                    fib.add_hline(first_level, line_color='orange')
                    fib.add_hline(second_level, line_color='yellow')
                    fib.add_hline(third_level, line_color='green')
                    fib.add_hline(fourth_level, line_color='blue')
                    fib.add_hline(minimum_price, line_color='purple')

                    fib.update_layout(
                        xaxis_rangeslider_visible=False,
                        title=f"Fibonacci Retracement for {stock_selection.upper()}",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)"
                    )

                    st.plotly_chart(fib)

                   

                # Run exception if error occurs to prompt the user to enter a valid ticker  
                except:
                    st.error("""Please try entering a valid ticker.""")

        # New section for strategy backtesting
        st.subheader("Backtesting Your Trading Strategies")

        # Markdown outlining the purpose of this section
        st.markdown("""The objective of this section is to allow the user to backtest a trading strategy
        based off a number of different trading strategies. Simply select a strategy from the options below and 
        see the results displayed in an interactive chart.""")

        # User input: backtesting trading strategies
        trade_strategy = st.selectbox("Select a trading strategy:",
                                            ["Fibonacci Retracement", "Exponential Moving Average"])
        
        # If condition for EMA
        if trade_strategy == 'Exponential Moving Average':
            
            # Include features in reporting tool    
            with st.spinner("Data loading..."):
                    
                # Include digital features
                time.sleep(1)

                # Calculate the exponential moving averages of the closing data
                EMA_short = talib.EMA(asset_data['Close'], timeperiod=12).to_frame()

                # Rename column
                EMA_short = EMA_short.rename(columns={0: 'Close'})

                # Calculate the exponential moving averages of the closing data
                EMA_long = talib.EMA(asset_data['Close'], timeperiod=50).to_frame()

                # Rename column
                EMA_long = EMA_long.rename(columns={0: 'Close'})

                # Copy EMA dataframe
                signal = EMA_long.copy()

                # Hold positions where data is null
                signal[EMA_long.isnull()] = 0

                # Where short EMA is greater than long EMA
                signal[EMA_short > EMA_long] = 1

                # Where long EMA is greater than short EMA
                signal[EMA_short < EMA_long] = -1

                # Extract only data where a buy and sell decision are required
                transition = signal[signal['Close'].diff()!=0]

                # Capture buy signals
                buy_signal = transition[transition['Close'] == 1]

                # Capture sell signals
                sell_signal = transition[transition['Close'] == -1]

                # Store index in variable
                long_index = buy_signal.index

                # Capture index of selling positions using buy index created above
                buy_position = asset_data[asset_data.index.isin(long_index)]

                # Store index in variable
                short_index = sell_signal.index

                # Capture index of selling positions using short index created above
                sell_position = asset_data[asset_data.index.isin(short_index)]

                # Produce backtesting strategy
                bt_strategy = bt.Strategy('EMA_crossover',
                                [   bt.algos.RunWeekly(),
                                    bt.algos.WeighTarget(signal),
                                    bt.algos.Rebalance()
                                ]
                            )
                
                # Turns off errors for when plotting chart is required
                st.set_option('deprecation.showPyplotGlobalUse', False)

                # Send backtest result to a dataframe
                bt_backtest = bt.Backtest(bt_strategy, asset_data['Close'].to_frame())

                # Run the backtest
                bt_result = bt.run(bt_backtest)

                # Create the plot
                bt_result.plot(title = f'EMA Backtesting strategy: {stock_selection.upper()}')

                # Show the plot (Seaborn library charting utilised)
                st.pyplot(plt.show())

        # Else if trading strategy is a Fibonacci Retracement
        elif trade_strategy == 'Fibonacci Retracement':
            
            # Include features in reporting tool    
            with st.spinner("Data loading..."):

                # Include digital features
                time.sleep(1)
                # Initiate try

                # If stock selection is valid
                try:

                    maximum_price = asset_data['Close'].max()
                    minimum_price = asset_data['Close'].min()

                    # Calculate the Fibonacci Levels

                    difference = maximum_price - minimum_price
                    first_level = maximum_price - difference * 0.236
                    second_level = maximum_price - difference * 0.382
                    third_level = maximum_price - difference * 0.5
                    fourth_level = maximum_price - difference * 0.618

                    # Calculate the Short Term Exponential Moving Average
                    ShortEMA = asset_data.Close.ewm(span=12, adjust=False).mean()

                    #Calculate the Long Term Exponential Moving Average
                    LongEMA = asset_data.Close.ewm(span=26, adjust=False).mean()

                    #Calculate the Moving Average Convergence / Divergence (MACD)
                    MACD = ShortEMA - LongEMA

                    #Calculate the Signal Line: This is a 9 period average of the MACD line itself
                    signal = MACD.ewm(span=9, adjust = False).mean()
                    asset_data['Signal Line'] = signal
                    asset_data['MACD'] = MACD
                    def getLevels(price):
                        if price >= first_level:
                            return (maximum_price, first_level)
                        elif price >= second_level:
                            return (first_level, second_level)
                        elif price >= third_level:
                            return (second_level, third_level)
                        elif price >= fourth_level:
                            return (third_level, fourth_level)
                        else:
                            return (fourth_level, minimum_price)

                    
                    def strategy(df):
                        buy_list = []
                        sell_list = []
                        flag = 0
                        last_buy_price = 0
                        
                        # Loop through the data set
                        for i in range(0, df.shape[0]):
                            price = df['Close'][i]
                            #if this is the first data point within the data set, then get the level above and below it.
                            if i == 0:
                                upper_lvl, lower_lvl = getLevels(price)
                                buy_list.append(np.nan)
                                sell_list.append(np.nan)
                                
                            # Else if the current price is greater than or equal to the upper_lvl, or less than or equal to the lower_lvl, then we know the price has 'hit' or crossed a new Fibonacci Level
                            elif (price >= upper_lvl) | (price <= lower_lvl):
                                
                                # Check to see if the MACD line crossed above or below the signal line
                                if df['Signal Line'][i] > df['MACD'][i] and flag == 0:
                                    last_buy_price = price
                                    buy_list.append(price)
                                    sell_list.append(np.nan)
                                    # Set the flag to 1 to signal that the share was bought 
                                    flag = 1
                                elif df['Signal Line'][i] < df['MACD'][i] and flag == 1 and price >= last_buy_price:
                                    buy_list.append(np.nan)
                                    sell_list.append(price)
                                    #Set the flag to 0 to signal that the share was sold
                                    flag = 0
                                else:
                                    buy_list.append(np.nan)
                                    sell_list.append(np.nan)
                                    
                            else:
                                buy_list.append(np.nan)
                                sell_list.append(np.nan)
                            
                            #Update the new level
                            upper_lvl, lower_lvl = getLevels(price)
                            
                        return buy_list, sell_list

                        # Create buy and sell columns
                    buy, sell = strategy(asset_data)
                    asset_data['Buy_Signal_Price'] = buy
                    asset_data['Sell_Signal_Price'] = sell

                    fib_data = asset_data.copy()

                    fib_data = fib_data[['Close','Buy_Signal_Price','Sell_Signal_Price']]

                    def active_strat(fib_data):

                        if fib_data["Buy_Signal_Price"] >= 0:

                            return "Active"

                        elif fib_data["Sell_Signal_Price"] >= 0:

                            return "Inactive"

                        else:

                            return np.NaN


                    fib_data["Active_strat"] = fib_data.apply(active_strat, axis=1)

                    fib_data["Active_strat"] = fib_data["Active_strat"].ffill()

                    fib_data["Active_strat"] = fib_data["Active_strat"].replace(
                        np.NaN, "Inactive"
                    )
                    
                    fib_data_active = fib_data[fib_data["Active_strat"] == "Active"]
                    fib_data_inactive = fib_data[fib_data["Active_strat"] == "Inactive"]

                    fib_data_active["Portfolio_Returns"] = fib_data_active["Close"].pct_change().astype(float)

                    fib_data_active.fillna(0, inplace=True)

                    fib_data_active["Portfolio_Returns"] = np.where(
                    fib_data_active["Buy_Signal_Price"] > 0,
                    np.NaN,
                    fib_data_active["Portfolio_Returns"],)

                    fib_data_active["Cumulative_returns"] = (
                        1 + fib_data_active["Portfolio_Returns"]
                    ).cumprod() - 1

                    fib_backtest = pd.concat([fib_data_active, fib_data_inactive])

                    fib_backtest = fib_backtest.sort_index()

                    fib_backtest["Cumulative_returns"] = fib_backtest["Cumulative_returns"].ffill()

                    fib_backtest["Cumulative_returns"] = fib_backtest["Cumulative_returns"] * 100

                    fib_backtest["Cumulative_returns"].plot(title = f"Fibonacci Retracement backtesting Strategy for {stock_selection.upper()}",
                                                            x = 'Date', y = '% Returns')

                    st.set_option('deprecation.showPyplotGlobalUse', False)

                    # Show the plot (Seaborn library charting utilised)
                    st.pyplot(plt.show())
                
                except:
                    st.warning("Invalid. Please try again.")

    # Initiate column two
    with col2:

        # Create title for second column
        st.subheader("Risk Profile: Anamoly Detection Using Stock Forecasting")

        # Introduction to new section
        st.markdown("""The following section collects the risk profile of the stocks selected by the user.
         Once assets are entered, financial metrics are derived to assess the health and risk of the selections. The chart below uses Meta's Prophet time-series forecasting machine learning capabilities. This is then overlayed with an anamoly detector (similar to 
        an isolation forest) that can be used to detect buy or sell triggers that technical analysis may not provide.""")

        # Time series forecasting and anamoly detection
        def fit_predict_model(dataframe, interval_width = 0.99, changepoint_range = 0.8):
            
            # Prepare dataset


            # Initiate Prophet object
            m = Prophet(daily_seasonality = False, yearly_seasonality = False, weekly_seasonality = False,
                        seasonality_mode = 'additive',
                        interval_width = interval_width,
                        changepoint_range = changepoint_range)
            
            # Fit the data
            m = m.fit(dataframe)

            # Forecast
            forecast = m.predict(dataframe)

            # Create a new column
            forecast['fact'] = dataframe['y'].reset_index(drop = True)

            # Return forecast
            return forecast
        
        # Call function
        pred = fit_predict_model(asset_data.reset_index().rename(columns = {'Date' : 'ds', 'Close' : 'y'})[['ds','y']])

        # Define a function for anamoly detection
        def detect_anomalies(forecast):
            
            # Capture necessary columns
            forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()

            # Set new column to 0
            forecasted['anomaly'] = 0

            # If variables above yhat upper, flag for anamoly
            forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1

            # If variables below yhat lower, flag for anamoly
            forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

            # Create a new column for column importance
            forecasted['importance'] = 0

            # Determine anamoly importance to flag
            forecasted.loc[forecasted['anomaly'] ==1, 'importance'] = (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
            forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] = (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']
            
            # Return forecasted
            return forecasted
        
        # Call the above function
        pred = detect_anomalies(pred)

        def plot_anomalies(forecasted):
            interval = alt.Chart(forecasted).mark_area(interpolate="basis", color = '#7FC97F').encode(
            x=alt.X('ds:T',  title ='date'),
            y='yhat_upper',
            y2='yhat_lower',
            tooltip=['ds', 'fact', 'yhat_lower', 'yhat_upper']
            ).interactive().properties(
                title=f'Anomaly Detection for {stock_selection.upper()}'
            )

            fact = alt.Chart(forecasted[forecasted.anomaly==0]).mark_circle(size=15, opacity=0.7, color = 'Black').encode(
                x='ds:T',
                y=alt.Y('fact', title='Price ($USD)'),    
                tooltip=['ds', 'fact', 'yhat_lower', 'yhat_upper']
            ).interactive()

            anomalies = alt.Chart(forecasted[forecasted.anomaly!=0]).mark_circle(size=30, color = 'Red').encode(
                x='ds:T',
                y=alt.Y('fact', title='Price ($USD)'),    
                tooltip=['ds', 'fact', 'yhat_lower', 'yhat_upper'],
                size = alt.Size( 'importance', legend=None)
            ).interactive()

            return alt.layer(interval, fact, anomalies)\
                    .properties(width=870, height=450)\
                    .configure_title(fontSize=20)
            
        st.altair_chart(plot_anomalies(pred), use_container_width=True)

        # Create title for second column
        st.subheader("Risk Profile: Stock Price Prediction Using Deep Learning")

        # Add markdown to explain section of the dashboard
        st.markdown("""The following chart uses historic time-series stock data to make weekly predictions on a given stock of your choice. 
        The neural network reads in stock data from the Yahoo Finance API. It is then trained on the training data shown in the chart below and then 
        makes a prediction of the movements of the stock price. The results demonstrate that stock prices can be predicted (to an extent) through the use 
        of artificial intelligence.""")

        # Include features in reporting tool    
        with st.spinner("Your Prediction is loading and requires processing power... Please be patient."):

            asset_data.reset_index(inplace=True)
        
            def Dataset(Data, Date):

                Train_Data = Data["Adj Close"][Data["Date"] < Date].to_numpy()
                Data_Train = []
                Data_Train_X = []
                Data_Train_Y = []
                for i in range(0, len(Train_Data), 5):
                    try:
                        Data_Train.append(Train_Data[i : i + 5])
                    except:
                        pass

                if len(Data_Train[-1]) < 5:
                    Data_Train.pop(-1)

                Data_Train_X = Data_Train[0:-1]
                Data_Train_X = np.array(Data_Train_X)
                Data_Train_X = Data_Train_X.reshape((-1, 5, 1))
                Data_Train_Y = Data_Train[1 : len(Data_Train)]
                Data_Train_Y = np.array(Data_Train_Y)
                Data_Train_Y = Data_Train_Y.reshape((-1, 5, 1))

                Test_Data = Data["Adj Close"][Data["Date"] >= Date].to_numpy()
                Data_Test = []
                Data_Test_X = []
                Data_Test_Y = []
                for i in range(0, len(Test_Data), 5):
                    try:
                        Data_Test.append(Test_Data[i : i + 5])
                    except:
                        pass

                if len(Data_Test[-1]) < 5:
                    Data_Test.pop(-1)

                Data_Test_X = Data_Test[0:-1]
                Data_Test_X = np.array(Data_Test_X)
                Data_Test_X = Data_Test_X.reshape((-1, 5, 1))
                Data_Test_Y = Data_Test[1 : len(Data_Test)]
                Data_Test_Y = np.array(Data_Test_Y)
                Data_Test_Y = Data_Test_Y.reshape((-1, 5, 1))

                return Data_Train_X, Data_Train_Y, Data_Test_X, Data_Test_Y


            def Model():
                model = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.LSTM(
                            200,
                            input_shape=(5, 1),
                            activation=tf.nn.leaky_relu,
                            return_sequences=True,
                        ),
                        tf.keras.layers.LSTM(200, activation=tf.nn.leaky_relu),
                        tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu),
                        tf.keras.layers.Dense(100, activation=tf.nn.leaky_relu),
                        tf.keras.layers.Dense(50, activation=tf.nn.leaky_relu),
                        tf.keras.layers.Dense(5, activation=tf.nn.leaky_relu),
                    ]
                )
                return model


            def scheduler(epoch):

                if epoch <= 150:
                    lrate = (10 ** -5) * (epoch / 150)
                elif epoch <= 400:
                    initial_lrate = 10 ** -5
                    k = 0.01
                    lrate = initial_lrate * math.exp(-k * (epoch - 150))
                else:
                    lrate = 10 ** -6

                return lrate


            epochs = [i for i in range(1, 1001, 1)]
            lrate = [scheduler(i) for i in range(1, 1001, 1)]
            callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

            Train_X, Train_Y, Test_X, Test_Y = Dataset(asset_data, "2021-10-01")


            model = Model()

            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss="mse",
                metrics=tf.keras.metrics.RootMeanSquaredError(),
            )

            hist = model.fit(
                Train_X,
                Train_Y,
                epochs=1001,
                validation_data=(Test_X, Test_Y),
                callbacks=[callback],
            )
            
            prediction = model.predict(Test_X)
            
            plt.figure(figsize=(20, 5))
            plt.plot(
                asset_data["Date"][asset_data["Date"] < "2021-10-01"],
                asset_data["Adj Close"][asset_data["Date"] < "2021-10-01"],
                label="Training",
            )
            plt.plot(
                asset_data["Date"][asset_data["Date"] >= "2021-10-01"],
                asset_data["Adj Close"][asset_data["Date"] >= "2021-10-01"],
                label="Testing",
            )
            plt.plot(
                asset_data["Date"][asset_data["Date"] >= "2021-10-12"],
                prediction.reshape(-1),
                label="Predictions",
            )
            plt.xlabel("Time")
            plt.ylabel("Closing Price")
            plt.legend(loc="best")
            st.pyplot(plt.show())
        

        # Create title for second column
        st.subheader("Risk Profile: Stock Sentiment Analysis")

        # Add markdown to explain section of the dashboard
        st.markdown("""The following section retrieves news about your ticker of choice (assuming data is available). Overall sentiment is then derived to determine whether 
        the stock is potentially worth buying/selling/holding based off the sentiment derived.""")

        # Include features in reporting tool    
        with st.spinner("Fetching all your news..."):

            ticker = stock_selection.upper()
            
        
            try:
                # Registered for Free Trail to get the below API
                news_api= os.getenv("STOCK_NEWS_API")
                date_range = ['03012022-03012022',
                            '03022022-0302022',
                        ]
                
                df = pd.DataFrame()
                print(f'---> Getting {ticker} news...')
                for quarter in date_range:
                    page_num = 1
                    keep_going = True
                    while keep_going:
                        print(f'     -> Quarter {quarter} - Page {page_num}')
                        try:
                            #print('A')
                            link = requests.get(f'https://stocknewsapi.com/api/v1?tickers={ticker}&items=50&date={quarter}&page={page_num}&token=' + news_api)
                            link.content
                            news = link.json()
                            #print('B')
                            df = df.append(pd.json_normalize(news['data']))
                            page_num = page_num + 1
                                
                        except:
                            print(f'Error on {ticker} - {quarter} - {page_num}')
                            keep_going = False
            except:
                st.warning("We encountered an error. This could be that the ticker is invalid or the number of API calls has exceeded the limit. Please try again.")
            
            try:
                sentiment_df = df['sentiment'].value_counts().to_frame().rename(columns = {'sentiment' : 'Sentiment'})
            except KeyError:
                st.warning("Cannot derive sentiment at this time.")

            st.subheader("The word on the street: Sentiment Score")
            st.markdown("Based on the information pulled, here is the following results of sentiment based on analysis conducted.")
            try:
                st.table(sentiment_df)
            except:
                None
            
            st.markdown("In addition to deriving sentiment on your stock of choice, here are the key words that are most commonly referred to about your stock: ")

            
            df_text = df['text'].to_frame()

            # Instantiate the lemmatizer
            lemmatizer = WordNetLemmatizer()

            # Create a list of stopwords
            stop_words = list(stopwords.words("english"))

            # Additional stopwords
            additional_stopwords = ["said", "also"]

            # Expand the default stopwords list if necessary
            for stop in additional_stopwords:

                # Append list
                stop_words.append(stop)
                
            # Complete the tokenizer function
            def tokenizer(text):
                """Tokenizes text."""

                # Remove the punctuation from text
                regex = re.compile("[^a-zA-Z ]")
                re_clean = regex.sub("", text)

                # Create a tokenized list of the words
                words = word_tokenize(re_clean)

                # Lemmatize words into root words
                lem = [lemmatizer.lemmatize(word) for word in words]

                # Remove the stop words
                tokens = [word.lower() for word in words if word.lower() not in stop_words]

                # Return output
                return tokens

            # Create new column overlaying the tokenizer function
            df_text["tokenize"] = df_text["text"].apply(lambda x: tokenizer(x))

            # Generate the words within the dataframe
            stock_words = tokenizer("".join(str(df_text["tokenize"].tolist())))

            # Create list of words in stock_words
            stock_words_list  = " ".join([str(x) for x in stock_words])

            mpl.rcParams["figure.figsize"] = [20.0, 10.0]

            # Generate the word cloud
            wc = WordCloud().generate(stock_words_list)

            # Turns off errors for when plotting chart is required
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.imshow(wc)

            # Show the image
            
            st.pyplot()
