import requests
from bs4 import BeautifulSoup
import pandas as pd
## Extract the content in page source
mntain_pg_src = requests.get("https://www.html.am/templates/downloads/bryantsmith/mountainouslysimple/")
mntain_pg_contents = mntain_pg_src.content

## Converting the web page contents to a Beatifulsoup object
mntain_pg_soup = BeautifulSoup(mntain_pg_contents)

## Extracting all hyperlinks
mntain_hyperlinks = mntain_pg_soup.find_all("a")
list_of_urls = []
for i in mntain_hyperlinks:
    list_of_urls.append(i['href'])

for i in mntain_hyperlinks:
    print(i.text)

    
## Extracting all the headings
mntain_h1 = mntain_pg_soup.find_all("h1")
for i in mntain_h1:
    print(i.text)
    
mntain_h2 = mntain_pg_soup.find_all("h2")
for i in mntain_h2:
    print(i.text)

colone = mntain_pg_soup.find_all("div", id = "columnOne")
colone[0].text

## Extracting div tag
div_all = mntain_pg_soup.find_all("div")
for i in div_all:
    print(i.attrs)

###################3 NASDAQ ##########################################
amzn_pg_src = requests.get("https://www.nasdaq.com/market-activity/stocks/amzn")
amzn_pg_content = amzn_pg_src.content
    
amzn_pg_soup = BeautifulSoup(amzn_pg_content)

amzn_span = amzn_pg_soup.find_all("span")
for i in amzn_span:
    print(i)

amzn_span0 = amzn_span[0]

amzn_header_name = amzn_pg_soup.find_all("span", {"class": "symbol-page-header__name"})
amzn_header_name[0].text

############# Extracting Tabular data using beautiful soup
icc_test_src = requests.get("https://www.icc-cricket.com/rankings/mens/team-rankings/test")
icc_test_content = icc_test_src.content
icc_test_soup = BeautifulSoup(icc_test_content)

icc_test_table = icc_test_soup.find_all("table")
icc_test_table[0]

############# Extracting Tabular data using pandas
## There is a pandas function which searches for tables in a web page
icc_test_tables_df = pd.read_html("https://www.icc-cricket.com/rankings/mens/team-rankings/test")
# returns a list of data frame
icc_test_ranking = icc_test_tables_df[0]

"""
Do a similar exercise for ODI and T20
merge (inner join) 3 tables
sum the points across all game formats for each team
Do a ranking on the sum of points
"""
    

############ IPL Points table - 2 tables in 1 page ########################
ipl_points_tables = pd.read_html("http://www.espncricinfo.com/wi/engine/series/1078425.html?view=pointstable")
    
    
    
    
    
    