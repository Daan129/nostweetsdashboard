# -*- coding: utf-8 -*-

# I M P O R T
import pandas as pd
import json
from datetime import datetime


# R E A D   J S O N   F I L E
with open("#NOS_search_tweets.json") as json_file:
    data = json.load(json_file)

df = pd.read_json("#NOS_search_tweets.json")

print(df.info())

# C R E A T E   N E W   D F
tweets = []


for tweet in data:
    tweet_text = tweet["text"]
    tweet_time = tweet["created_at"]
    tweet_time = datetime.strftime(datetime.strptime(tweet_time,'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
    hashtags_data = tweet["entities"]["hashtags"]
    for hashtag in hashtags_data:
        hashtags = hashtag["text"]
    user_id = tweet["user"]["id"]
    screen_name = tweet["user"]["name"]
    user_bio = tweet["user"]["description"]
    verified = tweet["user"]["verified"]
    screen_name = tweet["user"]["name"]
    if tweet["retweeted_status"] == "null":
        retweeted = 0
    else:
        retweeted = 1
    
    
    new_tweet = {
        "text": tweet_text,
        "retweeted": retweeted,
        "hashtags": hashtags,
        "tweet_time": tweet_time,
        "user_name": screen_name,
        "user_id": user_id,
        "user_bio": user_bio,
        "verified": verified,
        
        }
    
    tweets.append(new_tweet)


with open("clean_NOS_tweets.json", "w") as outfile:
  json.dump(tweets, outfile)