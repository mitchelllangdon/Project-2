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

# Create an application instances
app = CreateMultiplePages()

# Page layout
st.set_page_config(layout = "wide")

# Title of the dashboard that the user is operating
st.title("Trader1000 Algo Bot")
st.markdown("""Automate your stock trading using the Trader1000.""")

# Add pages based off custom imports above
app.add_page("Technical Analysis", Technical.app)
app.add_page("Sentiment Analysis", Sentiment.app)

# The main application
app.run()