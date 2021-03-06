# **A Stock Market sentiment analyzer based on twitter data using nlp**
#### -- Project Status: [Complete]

## Project deployed at: https://twitter-sentiment-stocks.herokuapp.com/ via streamlit

#### Project phases:
- [x] [curate tweets]
- [x] [Perform data cleaning]
- [x] Analyze data
- [x] submit findings
- [x] EDA analysis
- [x] compare stock price movement to buy/sell positions

## Project Intro
In an attempt to understand the voice of investors, this project seeks to understand the contextual language used on specific stocks. (Tesla in this example) 

### Methods Used
* Natural Processing Language
* Data lemmatization/stemming
* Data Tokenization
* Data Visualization

### Technologies
* Python
* GetOldTweets3
* Pandas
* Numpy
* Matplotlib
* Nltk
* Wordcloud 
* Text Blob
* yfinance

## Project Objectives
As people tweet about stocks on a daily scale, some things we hoped to discover included:

- What is the overall sentiment of a particular stock?
- Is the overall sentiment correlated to the stock price in any way?
- What positions do people on average towards any stock?

    ### things to note:

* The tweets were curated using GetOldTweets3. Twitter's API wasn't used as we found it to be very limited in its capabilities as a free user. This project does open up the question to what the sentiment is like on a grand scale.
* This analysis was done one 8000 tweets

## Key findings
- People overall are very bullish about Tesla's stock. However, there is a level of skepticism about how far the stock can climb given the current valuation.
- Most common words included: Call, Split, nice, wow, crazy

## Business Implication
- As an investor, most of my analysis drives from fundamentals of a company as well as recent news. One thing that has always been a challenge take into consideration is the general market consesus as we all may have different interpreatations of what the future may look like for any one company. The sentiment analyzer aims to move a step into that direction to further understand the general opinion of stocks 

## Contributing  Members

**Team Leads (Contacts) : [Samuel Lawrence]: http://samuel-lawrence.co.uk/**
