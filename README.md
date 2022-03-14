# FinTech-Project-2
​
# Search Finance & Trading
​
​
![Search Finance & Trading](https://media.istockphoto.com/vectors/abstract-financial-chart-with-bulls-and-bear-in-stock-market-on-white-vector-id1160947231?k=20&m=1160947231&s=612x612&w=0&h=4EJe2IE2N8YBz8-Q3lU4YfqoC44CSeJH7NzJ80VOvHU=)
​
# Team Members
- Mitchell Langdon 
- Marcus Whitelock
- Christian Seeley 
- Mohamed Dallol 
- Ilisha Kaul
​
# Project Description/Outline
​
- ### Could we create a technical stock analysis, for profitbale trades?
​
- ### Could we create a automated trading bot for profitable passive income?

- ### Can a tool be built that is dynamic, that produces meaningful and actionable insights for potential investors?
​
​
# Analysis Report
## Data collection:

There were two primary data sources used to collect data for our analysis, including:
 * [Stock News API](https://stocknewsapi.com/); and 
 * [Alpaca API](https://app.alpaca.markets/brokerage/new-account/overview); and
 * Yahoo Finance API

 API calls were made to each of the data sources, to create a dynamic method of capturing and presenting data back to the individual via Streamlit (Python dashboard application).


​
​
 
​
## We have data ! Lets see what we can do with it.
​ ![API Call Using our application](Images_readme/Search_Finance.gif)
​
​
​
​
# We have the data now lets use it to predict and trade

Deep learning models and linear regression models were utilised to inform trading decisions and make predictions. The anamoly detector (using linear regression) shows prices on a time-series that warrant further investigation. See the graphic below.

![API Call Using our application](Images_readme/Anomaly.gif)


The deep learning model uses a Long Short Term Memory neural net to make predictions. The key concern with the deep learning model is overfitting (or have we simply trained the model efficiently?)... 

![API Call Using our application](Images_readme/Neural_Net.png)

We then also use this information to backtest specific strategies. Take for example, the exponential moving average strategy for TSLA:

![API Call Using our application](Images_readme/Backtesting.png)

We could then make a trading decision. Let's go ahead and buy Tesla given our predictions, strategies and sentiment have confirmed our purchase:
​













