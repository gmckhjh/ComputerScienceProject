import time
from datetime import date
import preprocessor as p

import pandas as pd
import tweepy

"""
Script utilising the tweepy library to scrape tweets about a hashtag for sentiment analysis. Uses the twitter API which
is limited to a set amount of tweets. 
"""

# Keys and access
# Currently limited after Twitter API changes
api_key = 'FSEfX1E891umuqEog6YtQnpUs'
api_secret_key = '3IRHbu9IHPzc6Lq01KbP3z1rdNsmACZuCEklt1obc1AWg8eKPe'
access_token = '1428558191940538380-SP0SJxekmmYPKPKk1N4Bwk8WMPWSOy'
access_token_secret = '7oNgdazxsE7LRkuHovXx0igHbYyQ8PnudiTsJxEb2DHDf'

# Connection to twitter
auth_handler = tweepy.OAuthHandler(consumer_key=api_key, consumer_secret=api_secret_key)
auth_handler.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth_handler)

# search info
hashtag = 'UK'
tweet_amount = 200


# scrape data using twitters API
def scrape_data(tag):
    if '#' not in tag:
        tag = '#' + tag

    # Start timing
    t_start = time.time()

    # Gather tweets and call method to create dataframe
    tweet_data = tweepy.Cursor(api.search, q=hashtag, lang='en').items(tweet_amount)
    df = __create_df(tweet_data)

    # Finish timing
    t_stop = time.time()
    timer = t_stop - t_start
    print('Time taken: ', timer)

    return df


# Create pandas dataframe from scraped tweets
def __create_df(tweets):
    cols = ['timestamp', 'tweet']
    df = pd.DataFrame(columns=cols)

    for tweet in tweets:
        # Convert to timestamp
        d = tweet.created_at
        d = date(d.year, d.month, d.day)
        stamp = pd.Timestamp(d)

        # Clean and append
        tweet_text = p.clean(tweet.text)
        my_dict = [{'timestamp': stamp, 'tweet': tweet_text}]
        df = df.append(my_dict)

    # Set timestamp as index
    df = df.set_index('timestamp')

    return df
