import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from datetime import datetime, timedelta
from sklearn.ensemble import *


# import seaborn as sns # Only used for heatmap
# sns.set()


class CurrencyPredictor:
    x = None
    y = None

    x_df = None  # Data frame of x - features (instead of numpy array)

    classifiers = {
        'LinearRegression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=101),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=500, learning_rate=0.1),
        'Bagging Regressor': BaggingRegressor(n_estimators=500),
        'AdaBoost Regressor': AdaBoostRegressor(n_estimators=500, learning_rate=0.1),
        'Extra Tree Regressor': ExtraTreesRegressor(n_estimators=500),

    }

    summary = list()

    def __init__(self, currency, create_single_currency_data_set=False, prediction_period=1):
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)

        self.currency = currency
        self.prediction_period = prediction_period

        if create_single_currency_data_set:
            self.df = self.create_data_frame_for_currency(self.currency)
            self.df.fillna(0.0, inplace=True)
            self.df.to_csv('{}.csv'.format(self.currency), index=False)
            self.df = pd.read_csv('./{}.csv'.format(self.currency), parse_dates=['date'], index_col='date')
        else:
            self.df = pd.read_csv('./{}.csv'.format(self.currency), parse_dates=['date'], index_col='date')

        self.normalize_currency_data_set()
        self.create_labels_based_on_prediction_period()
        self.df_copy = self.df

        train_x, test_x, train_y, test_y = train_test_split(self.x, self.y, test_size=0.2, random_state=101)

        self.do_training(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

    def do_training(self, train_x, train_y, test_x, test_y):
        for name, clf in self.classifiers.items():
            nada = clf.fit(train_x, train_y)
            score = nada.score(test_x, test_y)
            prediction = clf.predict(test_x)
            # Maybe use different set than the one that is trained on
            cost_values = self.x_df[-167 - 10:]  # We'll take the last period elements to make our predictions on them
            print(cost_values.shape)
            forecast = clf.predict(cost_values)

            accuracy = score * 100
            mae = mean_absolute_error(test_y, clf.predict(test_x))
            mse = mean_squared_error(test_y, clf.predict(test_x))
            r2 = r2_score(test_y, clf.predict(test_x))

            print(f'R2: {r2:.2f}')
            print(f'MAE: {mae:.2f}')
            print(f'MSE: {mse:.2f}')
            print('{0:.4f}%'.format(accuracy))
            print()

            self.summary.append({
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'accuracy': accuracy,
                'prediction': prediction,
                'name': name,
                'forecast': forecast,
                'close': cost_values
            })

    def plot_predictions(self):
        for sum_dict in self.summary:
            name = sum_dict['name'] + '_forecast'
            self.df_copy[name] = sum_dict['forecast']
            self.df_copy['close'].plot(figsize=(12, 6), label='Close (actual value)',
                                       title='Currency: {}, Prediction period: {} hour'.format(self.currency,
                                                                                               self.prediction_period))
            self.df_copy[name].plot(figsize=(12, 6), label=name)

            plt.legend()
            plt.show()

            # def plot_heat_map(self):
            # predictions = [(sum_dict['prediction'], sum_dict['name']) for sum_dict in summary]
            # names = [x[1] for x in predictions] + ['test']
            #
            # stack_predict = np.vstack([x[0] for x in predictions] + [test_Y]).T
            # corr_df = pd.DataFrame(stack_predict, columns=names)
            # plt.figure(figsize=(10,5))
            # sns.heatmap(corr_df.corr(), annot=True)
            # plt.show()

    def create_labels_based_on_prediction_period(self):
        """
        Creating x and y from the data set. For each set of features x we create an label y
         features x -> close, interest_over_time_RU, hourly_relative_change, etc per hour
         label y -> The close price in 'prediction_period' steps ahead of current feature x close price

         Example: A prediction period of 1 will lead to predicting prices 1 hour ahead
        """
        self.df['Price_After_period'] = self.df['close'].shift(-self.prediction_period)
        self.df['Price_After_period'].fillna(0, inplace=True)
        # print(self.df['Price_After_period'])
        # exit()
        print(self.df['Price_After_period'].values.shape)
        self.df.drop(self.df.tail(1).index, inplace=True)
        self.x_df = self.df.drop('Price_After_period', axis=1)
        self.x = self.df.loc[:, self.x_df.columns != 'date'].values.reshape(-1, len(list(self.x_df)))
        self.y = self.df['Price_After_period']

    def normalize_currency_data_set(self):
        scale = MinMaxScaler()
        self.df.fillna(0.0, inplace=True)
        used_columns = list(set(list(self.df)) - set('date'))
        self.df[used_columns] = scale.fit_transform(self.df[used_columns])

        # df['close'].plot(figsize=(12,6),label='Close')
        # df['close'].rolling(window=7).mean().plot(label='7 Day Avg')
        # plt.legend()
        # plt.show()

    @staticmethod
    def create_data_frame_for_currency(currency):
        df = pd.read_csv('./crypto_data.csv')  # , parse_dates=['date'], index_col='date'

        # df.drop(['name'], axis=1, inplace=True)
        headers = list(df.drop(['name'], axis=1, inplace=False))
        currencies = df.set_index('name').T.to_dict('list')

        ethereum = currencies[currency]

        currency_dict = dict(zip(headers, ethereum))

        date_dict = {}
        list_of_dicts = []
        now = datetime.now()

        for j in range(0, 168):
            now = now + timedelta(hours=1)
            date_dict[str(now)] = {}

        counter = 0

        country_list = ['US', 'CA', 'SG', 'CN', 'JP', 'KR', 'IN', 'GB', 'DE', 'FR', 'ZA', 'GH', 'NG', 'AU', 'VE', 'BR',
                        'KE',
                        'RU']

        for key, val in date_dict.items():
            new_dict = {}

            new_dict['close'] = currency_dict['close_{}'.format(str(counter).zfill(4))]

            new_dict['hourly_relative_change'] = currency_dict['h_r_c{}'.format(str(counter).zfill(4))]

            for country in country_list:
                if 'i_o_t_{}_{}'.format(country, str(counter).zfill(4)) in currency_dict:
                    new_dict['interest_over_time_{}'.format(country)] = currency_dict[
                        'i_o_t_{}_{}'.format(country, str(counter).zfill(4))]

            if counter < 24:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(0)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(0)]
                new_dict['tweet_exposure'] = currency_dict['retweets_{}'.format(0)]

            elif 24 < counter < 48:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(1)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(1)]
                new_dict['tweet_exposure'] = currency_dict['exposure_{}'.format(1)]

            elif 48 < counter < 72:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(2)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(2)]
                new_dict['tweet_exposure'] = currency_dict['exposure_{}'.format(2)]

            elif 72 < counter < 96:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(3)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(3)]
                new_dict['tweet_exposure'] = currency_dict['exposure_{}'.format(3)]

            elif 96 < counter < 120:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(4)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(4)]
                new_dict['tweet_exposure'] = currency_dict['exposure_{}'.format(4)]

            elif 128 < counter < 144:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(5)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(5)]
                new_dict['tweet_exposure'] = currency_dict['exposure_{}'.format(5)]

            else:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(6)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(6)]
                new_dict['tweet_exposure'] = currency_dict['exposure_{}'.format(6)]

            if counter < 167:
                new_dict['hourly_relative_volume_change'] = currency_dict['h_r_v_c{}'.format(str(counter).zfill(4))]

            new_dict['date'] = key
            counter += 1
            list_of_dicts.append(new_dict)

        return pd.DataFrame(list_of_dicts)


if __name__ == '__main__':
    currency_predictor = CurrencyPredictor('Bitcoin', create_single_currency_data_set=True, prediction_period=1)
    currency_predictor.plot_predictions()
