from twython import Twython # Python wrapper written to access Twitter API
import pandas as pd
"""
API Reference links
https://developer.twitter.com/en/docs
https://developer.twitter.com/en/docs/tweets/search/overview#
https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets
"""

tw_app_key = 'UYOasRwKsvFYfgRTdc5gzO7Eg'
tw_app_secret = 'X1O747GNNLsU5kGPJtghQknDyaOqJmb948PrpX6J14EAGF9jsQ'
tw_oauth_token = '1140507056127954944-Fl26PfbkMbxIPDxlXEOpTSbedFh18P'
tw_oauth_token_secret = 'K66KNuHtIMRP8HHX3y6b6uWBdDPjz6cCAM2YAablFFw2Y'


tc = Twython(app_key = tw_app_key,
             app_secret= tw_app_secret,
             oauth_token= tw_oauth_token,
             oauth_token_secret= tw_oauth_token_secret)
tc.get_account_settings() # to check whether connection is successful
##trending_topics =  tc.get_place_trends()

durgashatmi_search = tc.search(q = "#durgaashtami", lang = "en",
                               count = 100) # returns a dictionary
durgashtami_statuses = durgashatmi_search["statuses"]   
durgashtami_tweets = []     
names = []                    
for i in durgashtami_statuses:
    durgashtami_tweets.append(i["text"])
    names.append(i["user"]["name"])

durgashtami_df = pd.DataFrame({"User":names,
                               "Tweet":durgashtami_tweets})


ufc_tweets = tc.search(q = "UFC243", lang = "en", count = 100,
                       result_type = "popular")

sunday_tweets = tc.search(q = "Sunday", lang = "en", count = 100,
                       result_type = "popular")



    