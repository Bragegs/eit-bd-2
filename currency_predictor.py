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

import seaborn as sns  # Only used for heatmap
sns.set()


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
    num_days_of_data = None

    image_dir = './images'
    start_time_date = None

    def __init__(self, currency,
                 data_files,
                 create_single_currency_data_set=False,
                 prediction_period=1,
                 start_time_date=None):
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)

        self.currency = currency
        self.prediction_period = prediction_period
        self.data_files = data_files
        self.num_days_of_data = len(data_files) * 7
        self.start_time_date = start_time_date

        if create_single_currency_data_set:
            #  We have to create the data set
            self.df = self.create_data_frame_for_currency(self.currency, self.data_files)
            self.df.fillna(0.0, inplace=True)
            self.df.to_csv('./csv_files/{}.csv'.format(self.currency), index=False)
            self.df = pd.read_csv('./csv_files/{}.csv'.format(self.currency), parse_dates=['date'], index_col='date')
        else:
            #  The data set is already created
            self.df = pd.read_csv('./csv_files/{}.csv'.format(self.currency), parse_dates=['date'], index_col='date')

        self.normalize_currency_data_set()
        self.create_labels_based_on_prediction_period()
        self.df_copy = self.df

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=0.2, random_state=101)

        self.do_training(train_x=self.train_x,
                         train_y=self.train_y,
                         test_x=self.test_x,
                         test_y=self.test_y)

    def do_training(self, train_x, train_y, test_x, test_y):
        for name, clf in self.classifiers.items():
            nada = clf.fit(train_x, train_y)
            # Sklearn/base.py line 357
            # Returns the coefficient of determination R^2 of the prediction
            score = nada.score(test_x, test_y)
            prediction_on_test = clf.predict(test_x)
            # We'll take the last period elements to make our predictions on them
            #cost_values = self.x_df.drop(self.x_df.tail(self.prediction_period).index, inplace=False)
            # cost_values = self.x_df
            # cost_values = self.x_df[-(167 * len(self.data_files)) - self.prediction_period:]
            cost_values = self.x_df
            forecast = clf.predict(cost_values)

            accuracy = score * 100
            mae = mean_absolute_error(test_y, prediction_on_test)
            mse = mean_squared_error(test_y, prediction_on_test)
            # The actual error value(how much we predict wrong) / all error values
            # R2 is a way of getting 'accuracy' has a probability distribution between 0 and 1
            # 1 - residual sum of square / total sum of squares
            r2 = r2_score(test_y, prediction_on_test)

            print(name)
            print(f'R2: {r2:.3f}')
            print(f'MAE: {mae:.3f}')
            print(f'MSE: {mse:.3f}')
            print('{0:.4f}%'.format(accuracy))
            print()

            self.summary.append({
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'accuracy': accuracy,
                'prediction_on_test_set': prediction_on_test,
                'name': name,
                'forecast': forecast,
                'close': cost_values
            })

    def plot_predictions(self):

        for sum_dict in self.summary:
            name = sum_dict['name'] + '_forecast'
            print('Plotting {}_{}.png'.format(name, self.currency))
            self.df_copy[name] = sum_dict['forecast']
            self.df_copy['close'].plot(figsize=(12, 6), label='Close (actual value)',
                                       title='Currency: {}, Prediction period: {} hour'.format(self.currency,
                                                                                               self.prediction_period))
            self.df_copy[name].plot(figsize=(12, 6), label=name)
            plt.legend()
            plt.savefig('./{}/{}_{}.png'.format(self.image_dir, name, self.currency), bbox_inches='tight')
            plt.gcf().clear()
            #exit()
            # plt.show()

    def plot_heat_map(self):
        print('Plotting heat map from prediction on test set')
        predictions = [(sum_dict['prediction_on_test_set'], sum_dict['name']) for sum_dict in self.summary]
        names = [x[1] for x in predictions] + ['test_y']

        stack_predict = np.vstack([x[0] for x in predictions] + [self.test_y]).T
        corr_df = pd.DataFrame(stack_predict, columns=names)
        plt.figure(figsize=(10,5))
        sns.heatmap(corr_df.corr(), annot=True)
        plt.savefig('./{}/prediction_heatmap_{}.png'.format(self.image_dir, self.currency), bbox_inches='tight')
        plt.gcf().clear()
        # plt.show()

    def get_best_regressor(self):
        highest_acc_value = -1
        highest_acc_idx = 0

        for idx, sum_dict in enumerate(self.summary):
            if sum_dict['accuracy'] > highest_acc_value:
                highest_acc_idx = idx

        return self.summary[highest_acc_idx]

    def create_labels_based_on_prediction_period(self):
        """
        Creating x and y from the data set. For each set of features x we create an label y
         features x -> close, interest_over_time_RU, hourly_relative_change, etc per hour
         label y -> The close price in 'prediction_period' steps ahead of current feature x close price

         Example: A prediction period of 1 will lead to predicting prices 1 hour ahead
        """
        self.df['Price_After_Period'] = self.df['close'].shift(-self.prediction_period)
        self.df['Price_After_Period'].fillna(0, inplace=True)
        # print(self.df['Price_After_Period'])
        # exit()
        # print(self.df['Price_After_Period'].values.shape)
        self.df.drop(self.df.tail(1).index, inplace=True)
        self.x_df = self.df.drop('Price_After_Period', axis=1)
        self.x = self.df.loc[:, self.x_df.columns != 'date'].values.reshape(-1, len(list(self.x_df)))
        self.y = self.df['Price_After_Period']

    def normalize_currency_data_set(self):
        scale = MinMaxScaler()
        self.df.fillna(0.0, inplace=True)
        # Normalize all columns except the date column
        used_columns = list(set(list(self.df)) - set('date'))
        self.df[used_columns] = scale.fit_transform(self.df[used_columns])

        self.df['close'].plot(figsize=(12,6),label='Close')
        self.df['close'].rolling(window=self.num_days_of_data).mean().plot(
            label='{} Day Avg'.format(self.num_days_of_data))
        plt.legend()
        plt.savefig('./{}/avg_price_{}_days.png'.format(self.image_dir, self.num_days_of_data), bbox_inches='tight')
        plt.gcf().clear()
        # plt.legend()
        # plt.show()

    def create_data_frame_for_currency(self, currency, data_files):
        data_file_count = 0
        list_of_currency_dicts = []

        # Find all hours that is going to be used in df
        if not self.start_time_date:
            start_time_date = datetime.now()
        else:
            start_time_date = self.start_time_date

        for data_file in data_files:
            date_list = []

            df = pd.read_csv('./csv_files/{}'.format(data_file))

            for j in range(168 * data_file_count, 168 * (data_file_count + 1)):
                start_time_date = start_time_date + timedelta(hours=1)
                date_list.append(str(start_time_date))

            print('Building data frame for {}st period'.format(data_file_count + 1))

            list_of_currency_dicts = self.build_data_frame_dict(df=df,
                                                                list_of_currency_dicts=list_of_currency_dicts,
                                                                currency=currency,
                                                                date_list=date_list)
            data_file_count += 1

        return pd.DataFrame(list_of_currency_dicts)

    @staticmethod
    def build_data_frame_dict(df, list_of_currency_dicts, currency, date_list):

        # df.drop(['name'], axis=1, inplace=True)
        headers = list(df.drop(['name'], axis=1, inplace=False))
        currencies = df.set_index('name').T.to_dict('list')

        selected_currency = currencies[currency]

        currency_dict = dict(zip(headers, selected_currency))

        hour_counter = 0

        country_list = ['US', 'CA', 'SG', 'CN', 'JP', 'KR', 'IN', 'GB', 'DE', 'FR', 'ZA', 'GH', 'NG', 'AU', 'VE', 'BR', 'KE', 'RU']

        for date_str in date_list:
            if hour_counter > len(list(df)):
                # The hour counter is now beyound the current data frame
                # The next data frame must be loaded
                break

            new_dict = {}

            new_dict['close'] = currency_dict['close_{}'.format(str(hour_counter).zfill(4))]
            new_dict['open'] = currency_dict['open_{}'.format(str(hour_counter).zfill(4))]
            new_dict['high'] = currency_dict['high_{}'.format(str(hour_counter).zfill(4))]
            new_dict['low'] = currency_dict['low_{}'.format(str(hour_counter).zfill(4))]
            new_dict['volume_to'] = currency_dict['volume_to_{}'.format(str(hour_counter).zfill(4))]
            new_dict['volume_from'] = currency_dict['volume_from_{}'.format(str(hour_counter).zfill(4))]

            for country in country_list:
                if 'i_o_t_{}_{}'.format(country, str(hour_counter).zfill(4)) in currency_dict:
                    new_dict['interest_over_time_{}'.format(country)] = currency_dict[
                        'i_o_t_{}_{}'.format(country, str(hour_counter).zfill(4))]

            if hour_counter < 24:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(0)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(0)]
                new_dict['tweet_exposure'] = currency_dict['retweets_{}'.format(0)]

            elif 24 < hour_counter < 48:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(1)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(1)]
                new_dict['tweet_exposure'] = currency_dict['exposure_{}'.format(1)]

            elif 48 < hour_counter < 72:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(2)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(2)]
                new_dict['tweet_exposure'] = currency_dict['exposure_{}'.format(2)]

            elif 72 < hour_counter < 96:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(3)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(3)]
                new_dict['tweet_exposure'] = currency_dict['exposure_{}'.format(3)]

            elif 96 < hour_counter < 120:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(4)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(4)]
                new_dict['tweet_exposure'] = currency_dict['exposure_{}'.format(4)]

            elif 128 < hour_counter < 144:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(5)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(5)]
                new_dict['tweet_exposure'] = currency_dict['exposure_{}'.format(5)]

            else:
                new_dict['num_tweets'] = currency_dict['tweets_{}'.format(6)]
                new_dict['num_retweets'] = currency_dict['retweets_{}'.format(6)]
                new_dict['tweet_exposure'] = currency_dict['exposure_{}'.format(6)]

            new_dict['date'] = date_str
            hour_counter += 1
            list_of_currency_dicts.append(new_dict)

        return list_of_currency_dicts


if __name__ == '__main__':
    data_files = ['crypto_data_week_9.csv', 'crypto_data_week_10.csv', 'crypto_data_week_11.csv']

    start_time_date_week_9 = datetime.strptime('2018-02-28 08:00:00.00', '%Y-%m-%d %H:%M:%S.%f')

    currency_predictor = CurrencyPredictor(currency='Bitcoin',
                                           data_files=data_files,
                                           create_single_currency_data_set=True,
                                           prediction_period=1,
                                           start_time_date=start_time_date_week_9)
    currency_predictor.plot_heat_map()
    print()
    currency_predictor.plot_predictions()
    print()

    print(22*'*' + ' BEST REGRESSOR ' + 22*'*')
    best_prediction_dict = currency_predictor.get_best_regressor()
    print('Name: {}'.format(best_prediction_dict['name']))
    print('R2: {0:.3f}'.format(best_prediction_dict['R2']))
    print('MAE: {0:.3f}'.format(best_prediction_dict['MAE']))
    print('MSE: {0:.3f}'.format(best_prediction_dict['MSE']))
    print('{0:.3f}%'.format(best_prediction_dict['accuracy']))
    print(60*'*')
