# Imports
import streamlit as st
import yfinance as yf
import datetime as dt
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
from CreatePages import CreateMultiplePages # Imported class module from Python file


def app():
    # Use Seaborn plotting style
    plt.style.use('seaborn')

    # Title
    st.title("Automated Stock Analyser: Technical Analysis")


    # Introduction
    st.markdown("""Self-service stock analyser using a combination of fundamental and technical indicators to assist in the research
    process for selecting stocks. In no way does this analyser depict financial advice. Users are expected to continue to complete their
    own research related to the stock(s) they analyse.""")

    # Sidebar: date
    st.subheader("Select Date Range (maximum three years history)")
    date_input = st.date_input("Select Date Range (max three years)", 
                                        value = dt.datetime.now() - dt.timedelta(days = 365 * 3))

    # Sidebar: Stock & search
    stock_selection = st.text_input("Enter a valid stock ticker:", "^GSPC")
    search_app = st.button("Search")

    # Retrieve data
    asset_data = yf.download(stock_selection, 
                                start = date_input, 
                                end = dt.datetime.now())

    # Apply moving averages to the dataframe
    asset_data["SMA50"] = asset_data["Close"].rolling(window=50).mean()
    asset_data["SMA100"] = asset_data["Close"].rolling(window=100).mean()

    # Insert subheader
    st.title("Technical Analysis")

    # Drop down box for chart type
    chart_type_select = st.selectbox("Select chart type", ('Line', 'Candlestick'))

    # Technical analysis multi-select
    indicator_options = st.selectbox(
                                    'Select Technical Indicators you want to overlay over your chart:',
                                    ('Simple Moving Average', 'None'))

    # Conditionals based on chart input
    if chart_type_select == 'Line':

        # Include features in reporting tool    
        with st.spinner("Data loading..."):
                
            # Include digital features
            time.sleep(1)
            # Initiate try
            try:

                # Initiate figure
                hist_figure = go.Figure()

                # Return basic line chart
                hist_figure.add_trace(go.Scatter(x = asset_data.index, y = asset_data['Close']))

                # Overlay indicators if selected
                if indicator_options is not None:
                    
                    # If indicator option is Simple Moving Average
                    if indicator_options == 'Simple Moving Average':
                        
                        # Update figure
                        hist_figure.add_trace(go.Scatter(x = asset_data.index, 
                                                         y = asset_data['SMA50'],
                                                    )
                                                )
                        # Update figure
                        hist_figure.add_trace(go.Scatter(x = asset_data.index, 
                                                         y = asset_data['SMA100'],
                                                    )
                                                )
                # Return figure
                st.plotly_chart(hist_figure, 
                                    use_container_width = True)

            # Run exception if error occurs  
            except:
                st.error("""Please try entering a valid ticker.""")

    elif chart_type_select == 'Candlestick':
            
        with st.spinner("Data loading..."):
                
            # Add digital feature
            time.sleep(1)

            # Initiate try
            try:
                # Create figure with secondary y-axis
                hist_fig = make_subplots(specs = [[{"secondary_y": True}]])

                # include candlestick with rangeselector
                hist_fig.add_trace(go.Candlestick(x = asset_data.index,
                                                    open = asset_data['Open'], high = asset_data['High'],
                                                    low = asset_data['Low'], close = asset_data['Close']),
                                                    secondary_y = True)

                # Add volumes
                hist_fig.add_trace(go.Bar(x = asset_data.index,
                                            y = asset_data['Volume']),
                                            secondary_y = False)

                # Show chart
                st.plotly_chart(hist_fig,
                                    use_container_width = True)

            # Run exception if error occurs                    
            except:
                st.error("""Please try entering a valid ticker.**""")

    st.title("Backtesting Your Trading Strategies")

    st.markdown("""The objective of this section is to allow the user to backtest a trading strategy
    based off a number of different trading strategies. Simply select a strategy from the options below and 
    see the results displayed in an interactive chart.""")

    st.selectbox("Select a trading strategy:",
    ["None", ""])
