import streamlit as st

# Initialize app
def app():

    # Title
    st.subheader("Purchasing Your Stocks")


    # Introduction
    st.markdown("""Use this tool to automatically purchase stocks using your Alpaca Trading account.""")

    # Alpaca Details
    with st.form(key='my_form'):
        api_key = st.text_input('Enter your Alpaca API Key:')
        password = st.text_input('Enter your Alpaca Secret Key', type = "password")
        st.form_submit_button('Login')


    # Select the stocks to purchase
    st.text_input('Select assets to Purchase')

    # Buy or sell stocks
    buy_button = st.button("Buy")
    sell_button = st.button("Sell")

    if buy_button == 'Buy':
        pass
        # buy_func():

    if buy_button == 'Sell':
        pass
        # sell_func():