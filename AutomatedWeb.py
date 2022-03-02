# Imports
# from selenium import webdriver
# import time


# def auto_web(username, password):

#     # Capture username and password
#     username = username
#     password = password

#     # Login url
#     url = "https://app.alpaca.markets/login"

#     # Paper URL to scrape information from
#     paper_url = "https://app.alpaca.markets/paper/dashboard/portfolio"

#     # Store webdriver in variable
#     driver = webdriver.Chrome("chromedriver.exe")

#     # Open web driver and open Alpaca
#     driver.get(url)

#     # Identify username and password fields
#     driver.find_element_by_name("username").send_keys(username)
#     driver.find_element_by_name("password").send_keys(password)

#     # Login
#     driver.find_element_by_css_selector(".ant-btn-primary").click()

#     # Allow Python to process login by adding timer
#     time.sleep(10)

#     # Move to paper_url web page for information to scrape data
#     driver.get(paper_url)
