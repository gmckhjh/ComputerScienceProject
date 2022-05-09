from Scweet.scweet import scrap
import pandas as pd
import datetime as dt
import time
import preprocessor as p

"""
Twitter scraper which uses the Scweet library, also limited by the twitter API is this is what it uses. 
"""

hashtag = 'Google'
tweet_data = pd.DataFrame()
num_days = 28
tweets_per_day = 3

__end_date = dt.date.today()
__start_date = __end_date - dt.timedelta(days=1)


# Scrape data from twitter
def scrape_data(tag):
    global __end_date
    global __start_date

    # Set up timer and create dataframe
    counter = num_days
    tweets = pd.DataFrame()
    t_start = time.time()

    while counter > 0:
        # Set start and end time and scrape tweets
        end = __end_date.strftime('%Y-%m-%d')
        start = __start_date.strftime('%Y-%m-%d')

        tweets = tweets.append(scrap(hashtag=tag, since=start, until=end, limit=tweets_per_day, interval=1,
                                     save_images=False))

        # Decrement days to search
        __end_date -= dt.timedelta(days=1)
        __start_date -= dt.timedelta(days=1)
        counter -= 1

    # Call method to create dataframe from tweets
    tweets = __create_df(tweets)

    # FInish timer
    t_end = time.time()
    print('time taken: ', t_end - t_start)

    return tweets


# Create pandas dataframe from scraped tweets
def __create_df(tweets):
    cols = ['timestamp', 'tweet']
    df = pd.DataFrame(columns=cols)

    for tweet in tweets.itertuples():
        # Convert to timestamp
        d = tweet.Timestamp
        stamp = pd.to_datetime(d, format='%Y-%m-%d', exact=True).date()

        # Clean and append
        tweet_text = p.clean(tweet.Text)
        my_dict = [{'timestamp': stamp, 'tweet': tweet_text}]
        df = df.append(my_dict)

    # Set timestamp as index
    df = df.set_index('timestamp')

    return df


# tweet_data = scrape_data(hashtag)
# print(tweet_data.info)
# print(tweet_data.values.tolist())
