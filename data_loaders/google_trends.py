from pytrends.request import TrendReq

pytrend = TrendReq(hl='en-US', tz=360)

kw_list = ["العربية"]
pytrend.build_payload(kw_list, cat=0, timeframe='now 7-d', geo='', gprop='')

# Interest Over Time
# interest_over_time_df = pytrend.interest_over_time()
# print(interest_over_time_df)

# # Interest by Region
interest_by_region_df = pytrend.interest_by_region()
print(interest_by_region_df)

# # Interest Over Time
# interest_over_time_df = pytrend.interest_over_time()
# print(interest_over_time_df.head())
#
# # Interest by Region
# interest_by_region_df = pytrend.interest_by_region()
# print(interest_by_region_df.head())
#
# # Related Queries, returns a dictionary of dataframes
# related_queries_dict = pytrend.related_queries()
# print(related_queries_dict)
#
# # Get Google Hot Trends data
# trending_searches_df = pytrend.trending_searches()
# print(trending_searches_df.head())
#
# # Get Google Top Charts
# top_charts_df = pytrend.top_charts(cid='actors', date=201611)
# print(top_charts_df.head())
#
# # Get Google Keyword Suggestions
# suggestions_dict = pytrend.suggestions(keyword='bitcoin sale')
# print(suggestions_dict)
