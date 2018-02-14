from pytrends.request import TrendReq
from data_loaders.market_cap import get_top_x_market_cap
from datetime import datetime, timedelta
import time
import pandas as pd

pytrend = TrendReq(hl='en-US', tz=360)

top_x_market_cap_currencies = get_top_x_market_cap(3)
country_list = ['US', 'CA', 'SG',
                'CN']  # [ 'JP', 'KR', 'IN', 'GB', 'DE', 'FR', 'ZA', 'GH', 'NG', 'AU', 'VE', 'BR', 'KE']


class WeeklyData:
    crypto_currencies = []

    def __init__(self, hour='10'):
        self.hour = hour
        self.trends_time_frame = self.create_trends_time_frame(num_days=7, hour=self.hour)

    def create_data_set(self):
        # Must start with interest over time columns
        self.create_interest_over_time_columns()
        df = pd.DataFrame(self.crypto_currencies)
        df.to_csv('interest_over_time.csv')

    def create_interest_over_time_columns(self):
        for currency in top_x_market_cap_currencies:
            currency_name = currency['name']

            interest_over_time_dict = {
                'name': currency_name
            }

            for country in country_list:
                print('Interest Over Time: \n currency: {}, country: {}, time frame: {}'.format(currency_name, country,
                                                                                                self.trends_time_frame))

                pytrend.build_payload([currency_name], timeframe=self.trends_time_frame, geo=country, gprop='')

                # Interest Over Time
                interest_over_time_df = pytrend.interest_over_time()
                counter = 0
                print(interest_over_time_df.shape)
                exit()
                if interest_over_time_df.shape[0] != 168:
                    print(
                        'Currency {} and country {} did not have correct amount of data points. Shape was {}'.format(
                            currency_name,
                            country,
                            interest_over_time_df.shape))
                    continue

                for _, row in interest_over_time_df.iterrows():
                    interest_over_time_dict['i_o_t_' + country + '_' + str(counter)] = row[currency_name]
                    counter += 1

                self.crypto_currencies.append(interest_over_time_dict)

    @staticmethod
    def create_trends_time_frame(num_days, hour=None):
        today = datetime.today().date()
        if hour is None:
            hour = time.strftime('%H').split(',')[0]
        new_date = today - timedelta(days=num_days)
        timeframe = new_date.strftime('%Y-%m-%dT' + hour) + ' ' + today.strftime('%Y-%m-%dT' + str(int(hour) - 1))
        return timeframe


if __name__ == '__main__':
    weekly_data = WeeklyData(hour='10')
    weekly_data.create_data_set()
