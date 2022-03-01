import streamlit as st

# Initialize app
def app():

    # Title
    st.title("Automated Stock Analyser: Sentiment Analysis")


    # Introduction
    st.markdown("""Use this tool to upload your sentiment data and complete natural language processing on your text. Otherwise, use the collected data to understand sentiment about a stock 
    you are researching.""")

    # Data upload feature
    uploaded_file = st.file_uploader("Choose a file")