# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:40:46 2019

@author: evkikum
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

icc_test_tables_df = pd.read_html("https://www.icc-cricket.com/rankings/mens/team-rankings/test")
# returns a list of data frame
icc_test_ranking = icc_test_tables_df[0]


icc_odi_tables_df = pd.read_html('https://www.icc-cricket.com/rankings/mens/team-rankings/odi')
icc_odi_ranking = icc_odi_tables_df[0]


icc_t20_tables_df = pd.read_html('https://www.icc-cricket.com/rankings/mens/team-rankings/t20i')
icc_t20_ranking = icc_t20_tables_df[0]

df = pd.merge(icc_test_ranking, icc_odi_ranking, on = 'Team')
df = pd.merge(df, icc_t20_ranking, on = 'Team')

df["Final_Points"] = df["Points_x"] + df["Points_y"] + df["Points"]
df = df.loc[:,["Team", 'Final_Points']]
df_final = df.sort_values('Final_Points', ascending=True)