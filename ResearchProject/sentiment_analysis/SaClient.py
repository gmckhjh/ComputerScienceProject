from datetime import datetime, date, timedelta

import pandas as pd

from preprocessing.ticker_preprocessing import get_ticker_data
from sentiment_analysis.SentAnVader import analyse_sentiment
from sentiment_analysis.sa_scraper import WebScraperTwitter
from sentiment_analysis.sa_scraper.TwitterScraper import scrape_hashtag
from sentiment_analysis.sa_scraper.fin_news_scraper import scrape_finviz

"""
Client to run sentiment analysis methods. 
"""

# Get data for ticker
ticker = 'GOOG'
date_start = datetime.strptime("2022-04-20", '%Y-%m-%d')
date_end = datetime.strptime("2022-05-08", '%Y-%m-%d')

train_percentage = 0.8


# Get sentiment from Twitter using the snscrape library
def twitter_sentiment(hashtag: str, date_begin: date = date.today() - timedelta(days=1),
                      date_ending: date = date.today()):

    # Check for hashtag and add if needed
    if not hashtag.startswith("#"):
        hashtag = "#" + hashtag

    tweets = scrape_hashtag(hashtag, date_begin, date_ending)
    compound_scores, score_per_date = analyse_sentiment(tweets)

    return compound_scores, score_per_date


# Get sentiment from finviz news (financial news website)
def financial_news_sentiment(method_ticker: str):
    finance_news = scrape_finviz(method_ticker)
    compound_scores, score_per_date = analyse_sentiment(finance_news)

    return compound_scores, score_per_date


def combined_sentiment(method_ticker: str, date_begin: date, date_ending: date):
    twitter_compound_scores, twitter_score_per_date = twitter_sentiment(method_ticker, date_begin, date_ending)
    finance_compound_scores, finance_score_per_date = financial_news_sentiment(method_ticker)

    return twitter_compound_scores, twitter_score_per_date, finance_compound_scores, finance_score_per_date


print("Should be starting")
twitter_compound, twitter_per_date, finance_compounds, finance_per_date = combined_sentiment(ticker, date_start,
                                                                                             date_end)

price_data = get_ticker_data(ticker, date_start, date_end)
price_close = price_data['Close']

print("Twitter Compound scores: ", twitter_compound)
print("Twitter scores per date: ", twitter_per_date, ", closing price on that day", price_close)
print("Finance Compound scores: ", finance_compounds)
print("Finance scores per date: ", finance_per_date, ", closing price on that day", price_close)
