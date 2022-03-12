import streamlit as st
# import AutomatedWeb
import alpaca_trade_api as tradeapi
import pandas as pd
import time
import os

### Future iterations
### Use streaming feature on Alpaca to load directly into streamlit
### https://medium.com/analytics-vidhya/financial-data-streaming-using-alpaca-and-streamlit-88aa21c75f27



# Initialize app
def app():

    # Title
    st.subheader("Purchasing Your Stocks")

    # Capture user details
    username = os.getenv("ALPACA_KEY")
    password = os.getenv("ALPACA_SECRET_KEY")

    # Introduction
    st.markdown("""Use this tool to automatically purchase or sell stocks using your Alpaca Trading account. 
    Ensure your Alpaca key and secret key are stored in your environment variables. Your key should be stored as 'ALPACA_KEY' and
    your secret key should be stored as 'ALPACA_SECRET_KEY'. For any issues, see this [link](https://kb.wisc.edu/cae/page.php?id=24500) to store environment variables 
    on your local machine.""")
    
    # Allow user to view current holdings with ALPACA
    st.subheader('Your Current Holdings/Positions')
    
    # Obtain current holdings on Alpaca
    api = tradeapi.REST(username, password, "https://paper-api.alpaca.markets", api_version="v2")

    # List all positions
    positions = api.list_positions()
    side = {"long": 1, "short": -1}
    
    # Create empty lists
    ticker = []
    current_price = []
    cost_basis = []
    shares = []
    today_change = []
    total_change = []

    # Append lists based on data from Alpaca
    for position in positions:
        ticker.append(position.symbol)
        current_price.append(position.current_price)
        cost_basis.append(position.cost_basis)
        shares.append(position.qty * side[position.side])
        today_change.append(position.unrealized_intraday_pl)
        total_change.append(position.unrealized_pl)

    # Create a dataframe
    portfolio = pd.DataFrame(
                            {
                                "Ticker": ticker,
                                "Current Price": current_price,
                                "Cost Basis": cost_basis,
                                "Shares": shares,
                                "Change Today": today_change,
                                "Total Return": total_change,
                            }
                        )
    
    # Set the index for ticker
    portfolio.set_index('Ticker', inplace = True)

    # Display portfolio holdings in a table if data is available
    if portfolio.shape[0] == 0:
        st.warning("You currently have no holdings with Alpaca. Make a trade below.")
    else:
        st.table(portfolio)

    # View current portfolio holdings
    st.subheader("Place an Order:")

    # Alpaca Details
    with st.form(key='my_form'):

        # Select the stocks to purchase
        ticker = st.text_input('Select assets to Purchase')

        # Number of shares to purchase
        num_shares = st.text_input('How many shares would you like to purchase/sell?')

        # Buy sell option
        buy_sell_select = st.selectbox("Select action", ('Buy', 'Sell'))

        # Submit form to initiate code
        st.form_submit_button('Place Order')
        try:
            try:
                if buy_sell_select == 'Buy':
                    
                    # Create API connection
                    api = tradeapi.REST(username, password, "https://paper-api.alpaca.markets", api_version="v2")

                    # Ticker
                    ticker = ticker

                    # Number of shares
                    number_of_shares = num_shares

                    # Index error handling
                    try: 

                        # Get final closing price
                        prices = api.get_barset(ticker, "1Min").df
                        limit_amount = prices[ticker]["close"][-1]

                        # Adjust for API errors
                        try:

                            # Submit order
                            api.submit_order(
                                        symbol=ticker, 
                                        qty=number_of_shares, 
                                        side='buy', 
                                        time_in_force="gtc", 
                                        type="limit", 
                                        limit_price=limit_amount
                                        )
                            
                            # Progress bar for added features
                            my_bar = st.progress(0)

                            # Show progress bar
                            for percent_complete in range(100):
                                time.sleep(0.01)
                                my_bar.progress(percent_complete + 1)

                            # Print success message
                            st.success('Your order has been completed.')

                        except:
                            st.warning("You either do not own the stock or you do not have adequate borrowing power.")
                    
                    # Handle index error with a warning message
                    except IndexError:
                        st.warning("You must enter a valid ticker to buy or sell.")

            # Handle exception
            except ValueError:
                st.warning('Enter your details above to proceed with a purchase or sale.')

            try:
                if buy_sell_select == 'Sell':
                    
                    # Create API connection
                    api = tradeapi.REST(username, password, "https://paper-api.alpaca.markets", api_version="v2")

                    # Ticker
                    ticker = ticker

                    # Number of shares
                    number_of_shares = num_shares

                    # Index error handling
                    try: 

                        # Get final closing price
                        prices = api.get_barset(ticker, "1Min").df
                        limit_amount = prices[ticker]["close"][-1]

                        # Adjust for API errors
                        try:

                            # Submit order
                            api.submit_order(
                                        symbol=ticker, 
                                        qty=number_of_shares, 
                                        side='sell', 
                                        time_in_force="gtc", 
                                        type="market"
                                        )
                            
                            # Progress bar for added features
                            my_bar = st.progress(0)

                            # Show progress bar
                            for percent_complete in range(100):
                                time.sleep(0.01)
                                my_bar.progress(percent_complete + 1)

                            # Print success message
                            st.success('Your order has been completed.')

                        except:
                            st.warning("You either do not own the stock (or enough of the stock) or you do not have adequate borrowing power.")
                    
                    # Handle index error with a warning message
                    except IndexError:
                        st.warning("You must enter a valid ticker to buy or sell.")

            # Handle exception
            except ValueError:
                st.warning('Enter your details above to proceed with a purchase or sale.')

        except:
                st.warning('Enter the details above to proceed.')