from pytrends.request import TrendReq
from data_loaders.market_cap import get_top_x_market_cap
from datetime import datetime, timedelta
import time
import pandas as pd
import requests
import calendar
import urllib.request
import json

pytrend = TrendReq(hl='en-US', tz=360)


class WeeklyData:
    top_x_market_cap_currencies = []
    crypto_currencies = []
    country_list = ['US', 'CA', 'SG', 'CN', 'JP', 'KR', 'IN', 'GB', 'DE', 'FR', 'ZA', 'GH', 'NG', 'AU', 'VE', 'BR', 'KE', 'RU']

    def __init__(self, hour='10'):
        self.hour = hour
        self.top_x_market_cap_currencies = get_top_x_market_cap(25)
        self.trends_time_frame = self.create_trends_time_frame(num_days=7, hour=self.hour)
        self.crypto_compare_time_stamp = self.create_crypto_compare_time_stamp(hour=self.hour)

    def create_data_set(self):
        print(60*'-')
        print('Creating data set')
        # Must start with interest over time columns
        self.create_interest_over_time_columns()
        self.create_hourly_price_historical()
        self.create_daily_twitter_data()

        df = pd.DataFrame(self.crypto_currencies)
        #df = df.fillna(0)
        #print(self.crypto_currencies[0])
        df.to_csv('crypto_data.csv', index=False)
        self.write_index_explanations(df)

    def create_interest_over_time_columns(self):
        print('Interest Over Time:')

        for currency in self.top_x_market_cap_currencies:
            currency_name = currency['name']

            currency_dict = {
                'name': currency_name
            }

            for country in self.country_list:
                print('Currency: {}, country: {}, time frame: {}'.format(currency_name, country,
                                                                                                self.trends_time_frame))
                time.sleep(1)

                pytrend.build_payload([currency_name], timeframe=self.trends_time_frame, geo=country, gprop='')

                # Interest Over Time
                interest_over_time_df = pytrend.interest_over_time()
                counter = 0

                if interest_over_time_df.shape[0] != 168:
                    print(
                        'Currency {} and country {} did not have correct amount of data points. Shape was {}'.format(
                            currency_name,
                            country,
                            interest_over_time_df.shape))
                    continue

                for _, row in interest_over_time_df.iterrows():
                    currency_dict['i_o_t_' + country + '_' + str(counter).zfill(4)] = row[currency_name]
                    counter += 1

            self.crypto_currencies.append(currency_dict)

    def create_hourly_price_historical(self, comparison_symbol='USD', limit=167, bin_width=1):

        #  params = ['close', "high", "low", "open", "time", "volumefrom", "volumeto", "timestamp"]

        for currency in self.top_x_market_cap_currencies:
            currency_name = currency['name']
            currency_symbol = currency['symbol']

            print('Hourly price historical: \n Currency: {}, {} '.format(currency_name, currency_symbol))

            currency_idx = self.get_index_of_crypto_currency_in_list(currency_name)

            if currency_idx == -1:
                print('Could not find currency-dictionary with name {} in list'.format(currency_name))
                continue

            url = 'https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym={}&limit={}&aggregate={}&toTs={}'.format(currency_symbol.upper(), comparison_symbol.upper(), limit, bin_width, self.crypto_compare_time_stamp)

            page = requests.get(url)
            historical_data_points = page.json()['Data']

            if len(historical_data_points) != 168:
                print(
                    'Currency {} did not have correct amount of data points. Length was {}'.format(
                        currency_name,
                        len(historical_data_points)))
                continue

            for idx, historical_data in enumerate(historical_data_points):
                hourly_closing_price = historical_data['close']
                hourly_opening_price = historical_data['open']
                hourly_volume_to = historical_data['volumeto']
                hourly_volume_from = historical_data['volumefrom']
                hourly_high = historical_data['high']
                hourly_low = historical_data['low']

                if not hourly_closing_price:
                    hourly_closing_price = 0

                if not hourly_opening_price:
                    hourly_opening_price = 0.01

                if not hourly_volume_to:
                    hourly_volume_to = 0.01

                self.crypto_currencies[currency_idx]['close_'+str(idx).zfill(4)] = hourly_closing_price
                self.crypto_currencies[currency_idx]['open_'+str(idx).zfill(4)] = hourly_opening_price
                self.crypto_currencies[currency_idx]['high_'+str(idx).zfill(4)] = hourly_high
                self.crypto_currencies[currency_idx]['low_'+str(idx).zfill(4)] = hourly_low
                self.crypto_currencies[currency_idx]['volume_to_'+str(idx).zfill(4)] = hourly_volume_to
                self.crypto_currencies[currency_idx]['volume_from_'+str(idx).zfill(4)] = hourly_volume_from

    def create_daily_twitter_data(self):
        rite_kit_api_client_key = '775323db6d69156ba811ce88d0e8b3a073b979148e5c'
        api_endpoint_url = 'https://api.ritekit.com/v1/stats/history/{}?client_id={}'

        for currency in self.top_x_market_cap_currencies:
            currency_name = currency['name']

            print('Twitter : \n Currency: {} '.format(currency_name))

            response = urllib.request.urlopen(api_endpoint_url.format(currency_name, rite_kit_api_client_key))
            json_response = json.load(response)

            #status_code = json_response['code']
            response_data = json_response['data']

            if not response_data:
                continue

            now = datetime.now()

            for hashtag_daily_data in response_data:
                day_string = hashtag_daily_data['date']
                datetime_object = datetime.strptime(day_string, '%Y-%m-%d')
                delta = now - datetime_object
                num_days_ago = delta.days

                if num_days_ago <= 7:
                    currency_idx = self.get_index_of_crypto_currency_in_list(currency_name)

                    num_tweets = hashtag_daily_data['tweets']
                    num_retweets = hashtag_daily_data['retweets']
                    exposure = hashtag_daily_data['exposure']

                    if not num_tweets:
                        num_tweets = 0

                    if not num_retweets:
                        num_retweets = 0

                    if not exposure:
                        exposure = 0

                    reversed_number = abs(num_days_ago - 7)
                    tweet_name = 'tweets_{}'.format(reversed_number)
                    retweets_name = 'retweets_{}'.format(reversed_number)
                    exposure_name = 'exposure_{}'.format(reversed_number)

                    self.crypto_currencies[currency_idx][tweet_name] = num_tweets
                    self.crypto_currencies[currency_idx][retweets_name] = num_retweets
                    self.crypto_currencies[currency_idx][exposure_name] = exposure

    @staticmethod
    def create_crypto_compare_time_stamp(hour):
        curr_year = time.strftime('%Y').split(',')[0]
        curr_month = time.strftime('%m').split(',')[0]
        curr_day = time.strftime('%d').split(',')[0]

        d = datetime(year=int(curr_year), month=int(curr_month), day=int(curr_day), hour=int(hour))

        return calendar.timegm(d.utctimetuple())

    @staticmethod
    def create_trends_time_frame(num_days, hour=None):
        today = datetime.today().date()
        if hour is None:
            hour = time.strftime('%H').split(',')[0]
        new_date = today - timedelta(days=num_days)
        time_frame = new_date.strftime('%Y-%m-%dT' + hour) + ' ' + today.strftime('%Y-%m-%dT' + str(int(hour) - 1))
        return time_frame

    def get_index_of_crypto_currency_in_list(self, currency_name):
        for idx, currency in enumerate(self.crypto_currencies):
            if currency['name'] == currency_name:
                return idx

        return -1

    @staticmethod
    def write_index_explanations(df):
        header_list = list(df)
        header_list_wo_numbers = [''.join([x for x in word if not x.isdigit()]) for word in header_list]
        header_list_wo_number_wo_duplicates = sorted(list(set(header_list_wo_numbers)))

        with open('column_explanation.csv', 'w') as file:
            file.write('sep=,')
            file.write('\n')
            file.write('header, start_column, end_column')

            csv_column_count = 0

            for unique_header in header_list_wo_number_wo_duplicates:
                file.write('\n')

                # Minus 1 because of 0-index
                how_many_times_repeated_in_header = header_list_wo_numbers.count(unique_header) - 1

                start_column = csv_column_count
                end_column = csv_column_count + how_many_times_repeated_in_header
                file.write('{}, {}, {}'.format(unique_header, start_column, end_column))

                csv_column_count += how_many_times_repeated_in_header + 1

if __name__ == '__main__':
    weekly_data = WeeklyData(hour='10')
    weekly_data.create_data_set()
