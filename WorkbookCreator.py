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

# Call class module and import workbooks
from CreatePages import CreateMultiplePages
import Technical
import Sentiment
import StockPurchase
import AutomatedWeb

# Create an application instances
app = CreateMultiplePages()

# Page layout
st.set_page_config(layout = "wide")

# Title of the dashboard that the user is operating
st.title("Search Finance and Trading")
st.markdown("""Automate your stock trading using the Search Finance and Trading application.""")
st.markdown("""### How to use the tool: 
 * This tool will collect your Alpaca details from your local machine upon opening the dashboard. If you do not have Alpaca details, no problems, you can still research.
 * Look at the technical charts below and overlay the insights with the anamoly detector and deep learning predictor to make informed decisions about your trading.
 * Understand the sentiment behind your stock selection through our Natural Language Processing analysis. 
 * Once you are satisfied with a given stock, select the Stock Purchasing tool from the 'App Navigation' sidebar and enter your Alpaca details.""")

# Add pages based off custom imports above
app.add_page("Technical Analysis", Technical.app)
app.add_page("Sentiment Analysis", Sentiment.app)
app.add_page("Stock Purchase Tool", StockPurchase.app)

# The main application
app.run()