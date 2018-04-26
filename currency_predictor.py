import numpy as np
import matplotlib
import csv
import os

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from datetime import datetime, timedelta
from sklearn.ensemble import *

import seaborn as sns  # Only used for heatmap

# sns.set()
sns.set(style='ticks', palette='Set2')
sns.despine()

scaler = None


class DataPreprocessor:

    def __init__(self,configurations=None, csv_dir=None, currency=None, start_time_date=None):
        self.configs = configurations
        self.processed_data_list = list()
        self.csv_dir = csv_dir
        self.currency = currency
        self.start_time_date = start_time_date

    def process_data(self):
        for config in configurations:
            data_files = config['data_files']
            test_data_files = config['test_data_files']
            cluster_currencies = config['cluster_currencies']
            prediction_period = config['prediction_period']
            headers_to_remove = config['headers_to_remove']

            #  We have to create the data set
            df = self.create_data_frame_for_currency(self.currency, data_files, cluster_currencies)
            df = self.drop_unwanted_headers(df, headers_to_remove, cluster_currencies)
            df.fillna(0.0, inplace=True)
            df.to_csv('{}/{}.csv'.format(self.csv_dir, self.currency), index=False)
            df = pd.read_csv('{}/{}.csv'.format(self.csv_dir, self.currency), parse_dates=['date'],
                                  index_col='date')

            # Creating training_data set
            testing_df = self.create_data_frame_for_currency(self.currency, test_data_files,
                                                             cluster_currencies)
            testing_df = self.drop_unwanted_headers(testing_df, headers_to_remove, cluster_currencies)
            testing_df.fillna(0.0, inplace=True)
            testing_df.to_csv('{}/{}.csv'.format(self.csv_dir, self.currency), index=False)
            testing_df = pd.read_csv('{}/{}.csv'.format(self.csv_dir, self.currency), parse_dates=['date'],
                                          index_col='date')

            #self.normalize_currency_data_set()
            #self.normalize_testing_data_set()

            df, x_df_fin, feature_names = self.create_labels_based_on_prediction_period(df=df, prediction_period=prediction_period)
            test_x_df, test_df = self.create_testing_data(testing_df=testing_df, prediction_period=prediction_period)


            data = {
                'df': df,
                'x_df': x_df_fin,
                'test_x_df': test_x_df,
                'test_df': test_df,
                'feature_names': feature_names,
                'data_files': data_files,
                'test_data_files': test_data_files,
                'cluster_currencies': cluster_currencies,
                'prediction_period': prediction_period,
                'headers_to_remove': headers_to_remove
            }

            self.processed_data_list.append(data)

        self.normalize_all_data_sets()
        self.create_x_y()

        #for x in self.processed_data_list:
           # df = x['df']
           # print(df.head(2))

        #print(self.processed_data_list)
        #exit()
        return self.processed_data_list

    def normalize_all_data_sets(self):
        max_values = dict()
        min_values = dict()

        for processed_data in self.processed_data_list:
            df = processed_data['df']
            x_df = processed_data['x_df']
            test_x_df = processed_data['test_x_df']
            test_df = processed_data['test_df']

            for header in list(set(list(df)+list(x_df)+list(test_x_df)+list(test_df))):
                if header == 'date':
                    continue
                for data_frame in [df, test_df]:
                    if header in max_values:
                        curr_value = max_values[header]
                        pos_new_val = data_frame[header].max()

                        if curr_value < pos_new_val:
                            max_values[header] = pos_new_val

                    else:
                        max_values[header] = data_frame[header].max()

                    if header in min_values:
                        curr_value = min_values[header]
                        pos_new_val = data_frame[header].min()

                        if curr_value > pos_new_val:
                            min_values[header] = pos_new_val
                    else:
                        min_values[header] = data_frame[header].min()

        for processed_data in self.processed_data_list:
            df = processed_data['df']
            x_df = processed_data['x_df']
            test_x_df = processed_data['test_x_df']
            test_df = processed_data['test_df']

            for header in list(set(list(df)+list(x_df)+list(test_x_df))):
                if header == 'date':
                    continue
                if header in max_values:
                    if header in min_values:
                        if header in list(x_df):
                            x_df[header] = (x_df[header] - min_values[header]) / (max_values[header] - min_values[header])
                        if header in list(test_x_df):
                            test_x_df[header] = (test_x_df[header] - min_values[header]) / (max_values[header] - min_values[header])
                        if header in list(df):
                            df[header] = (df[header] - min_values[header]) / (max_values[header] - min_values[header])
                        if header in list(test_df):
                            test_df[header] = (test_df[header] - min_values[header]) / (max_values[header] - min_values[header])

    @staticmethod
    def create_labels_based_on_prediction_period(df, prediction_period):
        """
        Creating x and y from the data set. For each set of features x we create an label y
         features x -> close, interest_over_time_RU, hourly_relative_change, etc per hour
         label y -> The close price in 'prediction_period' steps ahead of current feature x close price

         Example: A prediction period of 1 will lead to predicting prices 1 hour ahead
        """

        # Setting next hours closing price to be the y
        df['Price_After_Period'] = df['close'].shift(-prediction_period)
        df['Price_After_Period'].fillna(0, inplace=True)
        # Remove last row
        df.drop(df.tail(prediction_period).index, inplace=True)
        x_df = df.drop('Price_After_Period', axis=1)
        # self.x is all columns except the y
        feature_names = list(filter(lambda a: a != 'date', list(x_df)))
        feature_names = np.asarray(feature_names)

        return df, x_df,feature_names

    @staticmethod
    def create_testing_data(testing_df, prediction_period):
        # Setting next hours closing price to be the y
        testing_df['Price_After_Period'] = testing_df['close'].shift(-prediction_period)
        testing_df['Price_After_Period'].fillna(0, inplace=True)
        # Remove last row
        testing_df.drop(testing_df.tail(prediction_period).index, inplace=True)
        test_x_df = testing_df.drop('Price_After_Period', axis=1)

        return test_x_df, testing_df

    def create_x_y(self):
        for processed_data in self.processed_data_list:
            df = processed_data['df']
            x_df = processed_data['x_df']
            x = df.loc[:, x_df.columns != 'date'].values.reshape(-1, len(list(x_df)))
            # self.y is only the y column
            y = df['Price_After_Period']
            processed_data['x'] = x
            processed_data['y'] = y

    def create_data_frame_for_currency(self, currency, data_files, cluster_currencies):
        data_file_count = 0
        list_of_currency_dicts = []

        # Find all hours that is going to be used in df
        if not self.start_time_date:
            start_time_date = datetime.now()
        else:
            start_time_date = self.start_time_date

        for data_file in data_files:
            date_list = []

            df = pd.read_csv('{}/{}'.format(self.csv_dir, data_file))

            for j in range(168 * data_file_count, 168 * (data_file_count + 1)):
                start_time_date = start_time_date + timedelta(hours=1)
                date_list.append(str(start_time_date))

            print('Building data frame for {}st period'.format(data_file_count + 1))

            list_of_currency_dicts = self.build_data_frame_dict_for_selected_currency(df=df,
                                                                                      list_of_currency_dicts=list_of_currency_dicts,
                                                                                      selected_currency=currency,
                                                                                      date_list=date_list,
                                                                                      cluster_currencies=cluster_currencies)
            data_file_count += 1

        return pd.DataFrame(list_of_currency_dicts)

    def build_data_frame_dict_for_selected_currency(self, df, list_of_currency_dicts, selected_currency, date_list,
                                                    cluster_currencies):
        # df.drop(['name'], axis=1, inplace=True)
        headers = list(df.drop(['name'], axis=1, inplace=False))
        currencies = df.set_index('name').T.to_dict('list')

        hour_counter = 0

        country_list = ['US', 'CA', 'SG', 'CN', 'JP', 'KR', 'IN', 'GB', 'DE', 'FR', 'ZA', 'GH', 'NG', 'AU', 'VE', 'BR',
                        'KE', 'RU']

        for date_str in date_list:
            if hour_counter > len(list(df)):
                # The hour counter is now beyound the current data frame
                # The next data frame must be loaded
                break

            new_dict = {}

            # Adding predicted currency to cluster array
            # This is done to merge all data points from the clusters in one row

            all_currencies = cluster_currencies + [selected_currency]

            for currency in all_currencies:

                # Creating the prefix of the column headers
                prefix = '{}_'.format(currency)

                if currency == selected_currency:
                    prefix = ''

                current_currency = currencies[currency]

                currency_dict = dict(zip(headers, current_currency))

                new_dict['{}close'.format(prefix)] = currency_dict['close_{}'.format(str(hour_counter).zfill(4))]
                new_dict['{}open'.format(prefix)] = currency_dict['open_{}'.format(str(hour_counter).zfill(4))]
                new_dict['{}high'.format(prefix)] = currency_dict['high_{}'.format(str(hour_counter).zfill(4))]
                new_dict['{}low'.format(prefix)] = currency_dict['low_{}'.format(str(hour_counter).zfill(4))]
                new_dict['{}volume_to'.format(prefix)] = currency_dict[
                    'volume_to_{}'.format(str(hour_counter).zfill(4))]
                new_dict['{}volume_from'.format(prefix)] = currency_dict[
                    'volume_from_{}'.format(str(hour_counter).zfill(4))]

                for country in country_list:
                    if 'i_o_t_{}_{}'.format(country, str(hour_counter).zfill(4)) in currency_dict:
                        new_dict['{}interest_over_time_{}'.format(prefix, country)] = currency_dict[
                            'i_o_t_{}_{}'.format(country, str(hour_counter).zfill(4))]

                num_tweets, num_retweets, tweet_exposure = self.get_twitter_data_points_from_hour(hour_counter,
                                                                                                  currency_dict)

                new_dict['{}num_tweets'.format(prefix)] = num_tweets
                new_dict['{}num_retweets'.format(prefix)] = num_retweets
                new_dict['{}tweet_exposure'.format(prefix)] = tweet_exposure

            new_dict['date'] = date_str
            hour_counter += 1
            list_of_currency_dicts.append(new_dict)

        return list_of_currency_dicts

    @staticmethod
    def get_twitter_data_points_from_hour(hour_counter, currency_dict):
        # Finding the correct datapoints considering
        # what hour the current data point is in
        day = 0

        if hour_counter < 24:
            day = 0
        elif 24 < hour_counter < 48:
            day = 1
        elif 48 < hour_counter < 72:
            day = 2
        elif 72 < hour_counter < 96:
            day = 3
        elif 96 < hour_counter < 120:
            day = 4
        elif 128 < hour_counter < 144:
            day = 5
        else:
            day = 6

        return currency_dict['tweets_{}'.format(day)], currency_dict['retweets_{}'.format(day)], currency_dict[
            'exposure_{}'.format(day)]

    @staticmethod
    def drop_unwanted_headers(df, headers_to_remove, cluster_currencies):
        for unwanted_header in headers_to_remove:
            if unwanted_header in 'close' or unwanted_header in 'date':
                print('Can not drop header: {}'.format(unwanted_header))
                continue
            df.drop(unwanted_header, axis=1, inplace=True)

            for currency in cluster_currencies:
                df.drop('{}_'.format(currency) + unwanted_header, axis=1, inplace=True)

        return df


class CurrencyPredictor:
    x = None
    y = None

    x_df = None  # Data frame of x - features (instead of numpy array)
    test_x_df = None

    regressors = None

    summary = list()

    image_dir = os.path.dirname(os.path.abspath(__file__)) + '/images'
    start_time_date = None

    feature_names = list()

    def __init__(self,
                 currency,
                 prediction_period=1,
                 start_time_date=None,
                 headers_to_remove=None,
                 cluster_currencies=None,
                 csv_dir=None,
                 feature_names=None,
                 x=None,
                 y=None,
                 x_df=None,
                 test_x_df=None,
                 test_df=None):

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)

        self.x = x
        self.y = y
        self.x_df = x_df
        self.test_x_df = test_x_df
        self.test_df = test_df
        self.feature_names = feature_names
        self.csv_dir = csv_dir
        self.summary = list()

        self.regressors = {
            # 'LinearRegression': LinearRegression(),
            ('Random Forest Regressor', 'RFR'): RandomForestRegressor(n_estimators=500, random_state=101),
            ('Gradient Boosting Regressor', 'GBR'): GradientBoostingRegressor(n_estimators=500, learning_rate=0.1),
            ('Bagging Regressor', 'BR'): BaggingRegressor(n_estimators=500),
            ('AdaBoost Regressor', 'ABR'): AdaBoostRegressor(n_estimators=500, learning_rate=0.1),
            ('Extra Tree Regressor', 'ETR'): ExtraTreesRegressor(n_estimators=500),
        }
        self.currency = currency
        self.prediction_period = prediction_period
        self.testing_files = testing_files
        #self.num_days_of_data = len(data_files) * 7
        self.start_time_date = start_time_date
        self.headers_to_remove = headers_to_remove
        self.cluster_currencies = cluster_currencies
        self.image_dir += '/{}_{}h_{}'.format(currency, self.prediction_period,
                                              'cluster' if len(self.cluster_currencies) > 0 else 'no_cluster')

        if not os.path.isdir(self.image_dir):
            os.makedirs(self.image_dir)


        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=101)

        self.do_training(train_x=self.train_x,
                         train_y=self.train_y,
                         test_x=self.test_x,
                         test_y=self.test_y)

    def do_training(self, train_x, train_y, test_x, test_y):
        for name, regressor in self.regressors.items():
            nada = regressor.fit(train_x, train_y)
            # Sklearn/base.py line 357
            # Returns the coefficient of determination R^2 of the prediction
            score = nada.score(test_x, test_y)
            prediction_on_test = regressor.predict(test_x)
            # We'll take the last period elements to make our predictions on them
            # cost_values = self.x_df.drop(self.x_df.tail(self.prediction_period).index, inplace=False)
            # cost_values = self.x_df
            # cost_values = self.x_df[-(167 * len(self.data_files)) - self.prediction_period:]
            cost_values = self.x_df

            forecast = regressor.predict(cost_values)

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
                'name': name[0],
                'short_name': name[1],
                'forecast': forecast,
                'close': cost_values,
                'regressor': regressor,
                'score': r2 - mse - mae
            })

    def plot_predictions(self, with_untrained_data= False):
        for sum_dict in self.summary:
            if with_untrained_data:
                name = sum_dict['name'] + ' forecast_on_untrained_data_set'
                values_to_predict = self.test_x_df.copy()
            else:
                name = sum_dict['name'] + ' forecast'
                values_to_predict = self.x_df.copy()

            print('Plotting {}_{}.png'.format(name, self.currency))
            regressor = sum_dict['regressor']

            forecast = regressor.predict(values_to_predict)
            original_cost_values = values_to_predict[['close']]

            plot_df = pd.DataFrame(columns=['Close', sum_dict['name'] + ' forecast'])

            for i in range(len(original_cost_values)):
                if i - self.prediction_period <= 0:
                    plot_df.loc[i] = [original_cost_values['close'][i], None]
                else:
                    #  Shift forecast back in time again to plot "Now"
                    plot_df.loc[i] = [original_cost_values['close'][i], forecast[i - self.prediction_period]]

            plot_df.plot(figsize=(12, 6), label=name,
                         title='Currency: {}, Prediction period: {}h. Forecast.'.format(self.currency,
                                                                                        self.prediction_period))
            plt.legend()
            plt.xlabel('Hour')
            plt.ylabel('Close')
            plt.savefig('{}/{} {}.png'.format(self.image_dir, name, self.currency).replace(' ', '_'),
                        bbox_inches='tight', dpi=300)
            plt.gcf().clear()
        # for sum_dict in self.summary:
        #     name = sum_dict['name'] + ' forecast'
        #     print('Plotting {}_{}.png'.format(name, self.currency))
        #     orig_test_df = self.test_df.copy()
        #
        #     actual_values = orig_test_df['Price_After_Period']
        #     actual_values.columns = ['date', 'Price_After_Period']
        #
        #     values_to_predict = orig_test_df.drop('Price_After_Period', axis=1, inplace=False)
        #     regressor = sum_dict['regressor']
        #
        #     forecast = regressor.predict(values_to_predict)
        #     original_cost_values = values_to_predict[['close']]
        #
        #     plot_df = pd.DataFrame(columns=['Close', sum_dict['name'] + ' forecast'])
        #
        #     for i in range(len(original_cost_values)):
        #         if i - self.prediction_period <= 0:
        #             plot_df.loc[i] = [actual_values.iloc[i], None]
        #         else:
        #             plot_df.loc[i] = [actual_values.iloc[i], forecast[i]]
        #
        #     plot_df.plot(figsize=(12, 6), label=name,
        #                  title='Currency: {}, Prediction period: {}h. Forecast.'.format(self.currency,
        #                                                                                 self.prediction_period))
        #     plt.legend()
        #     plt.xlabel('Hour')
        #     plt.ylabel('Close')
        #     plt.savefig('{}/{} {}.png'.format(self.image_dir, name, self.currency).replace(' ', '_'),
        #                 bbox_inches='tight', dpi=300)
        #     plt.gcf().clear()


            # plt.show()

    def plot_regressor_error(self):
        for sum_dict in self.summary:
            name = sum_dict['name'] + ' error'
            print('Plotting {}_{}.png'.format(name, self.currency))
            values_to_predict = self.x_df.copy()
            regressor = sum_dict['regressor']

            forecast = regressor.predict(values_to_predict)
            original_cost_values = values_to_predict[['close']]

            plot_df = pd.DataFrame(columns=['Error'])
            # plot_df['error'] = plot_df[sum_dict['name'] + ' forecast'] - plot_df['Close']

            for i in range(len(original_cost_values)):
                if i - self.prediction_period <= 0:
                    plot_df.loc[i] = [None]
                else:
                    plot_df.loc[i] = [forecast[i - self.prediction_period] - original_cost_values['close'][i]]

            plot_df.plot(figsize=(12, 6), label=name,
                         title='Currency: {}, Prediction period: {}h. Error.'.format(self.currency,
                                                                                     self.prediction_period))
            plt.legend()
            plt.xlabel('Hour')
            plt.ylabel('Error (predicted - actual close)')
            axes = plt.gca()
            abs_min_max = 0.26
            axes.set_ylim([-abs_min_max, abs_min_max])
            plt.savefig('{}/{} {}.png'.format(self.image_dir, name, self.currency).replace(' ', '_'),
                        bbox_inches='tight', dpi=300)
            plt.gcf().clear()

            # plt.show()

    def plot_regressor_results(self, val_name, color):
        name = 'Regressor {} results'.format(val_name)
        print('Plotting {}_{}.png'.format(name, self.currency))

        # Bring some raw data.
        val = [x[val_name] for x in self.summary]

        if val_name == 'R2':
            val = [x * 100 for x in val]

        # In my original code I create a series and run on that,
        # so for consistency I create a series from the list.
        freq_series = pd.Series.from_array(val)

        x_labels = [x['short_name'] for x in self.summary]

        # Plot the figure.
        plt.figure(figsize=(12, 8))
        ax = freq_series.plot(kind='bar', color=color)
        ax.set_title(val_name)
        ax.set_xlabel('Regressors')
        ax.set_ylabel('Score')
        ax.set_xticklabels(x_labels)

        rects = ax.patches

        # For each bar: Place a label
        for rect in rects:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = 5
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with decimal places
            label = "{:.4f}".format(y_value)

            # Create annotation
            plt.annotate(
                label,  # Use `label` as label
                (x_value, y_value),  # Place label at end of the bar
                xytext=(0, space),  # Vertically shift label by `space`
                textcoords='offset points',  # Interpret `xytext` as offset in points
                ha='center',  # Horizontally center label
                va=va)  # Vertically align label differently for

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)

        plt.savefig('{}/{} {}.png'.format(self.image_dir, name, self.currency).replace(' ', '_'), bbox_inches='tight')
        plt.gcf().clear()

        # plt.show()

    def plot_last_predictions(self, with_untrained_data=False):
        for sum_dict in self.summary:

            name = sum_dict['name'] + ' last predictions'
            print('Plotting last predictions of {}_{}.png'.format(name, self.currency))

            period_length = 8
            # points_we_can_remove = self.x_df.shape[0] - (2 * period_length * self.prediction_period)
            # # 384
            # # num_days - num_last_predictions
            # values_to_predict = self.x_df
            if with_untrained_data:
                points_we_can_remove = self.test_x_df.shape[0] - (2 * period_length * self.prediction_period)
                # num_days - num_last_predictions
                values_to_predict = self.test_x_df
            else:
                points_we_can_remove = self.x_df.shape[0] - (2 * period_length * self.prediction_period)
                # num_days - num_last_predictions
                values_to_predict = self.x_df

            # self.x_df.drop(self.x_df.head(delta).index, inplace=False)  # self.x_df[-delta:]
            values_to_predict = values_to_predict.drop(values_to_predict.head(points_we_can_remove).index, inplace=False)
            regressor = sum_dict['regressor']

            forecast = regressor.predict(values_to_predict)
            original_cost_values = values_to_predict[['close']]

            df = pd.DataFrame(
                columns=['Date', 'Close', 'Actual value', sum_dict['name'] + ' forecast', 'Prediction start'])

            if not self.start_time_date:
                start_time_date = datetime.now()
            else:
                start_time_date = self.start_time_date

            for i in range(0, 2 * period_length * self.prediction_period, self.prediction_period):

                if i == (period_length * self.prediction_period):
                    # Make all 3 graph end/start on same place for UI
                    df.loc[i] = [start_time_date, original_cost_values['close'][i], original_cost_values['close'][i],
                                 original_cost_values['close'][i], original_cost_values['close'][i]]
                elif i > period_length * self.prediction_period:

                    # if i - self.prediction_period <= 0:
                    #     print('HEERE')
                    #     print(i)
                    #     df.loc[i] = [None, original_cost_values['close'][i], None, None]
                    # else:
                    #     df.loc[i] = [None, original_cost_values['close'][i], forecast[i - self.prediction_period], None] # Shifting forecast to actual close value

                    df.loc[i] = [start_time_date, None, original_cost_values['close'][i],
                                 forecast[i - self.prediction_period], None]  # Shifting forecast to actual close value
                else:
                    df.loc[i] = [start_time_date, original_cost_values['close'][i], None, None, None]

                start_time_date = start_time_date + timedelta(hours=1 * self.prediction_period)

            df.set_index('Date', inplace=True)
            df.plot(y='Actual value')

            ax = df.plot(y='Close')
            df.plot(y=sum_dict['name'] + ' forecast', ax=ax, subplots=True, color='#808AB3')
            df.plot(y='Actual value', linestyle='dotted', ax=ax, subplots=True, color='#ba645b')
            df.plot(y='Prediction start', marker='o', ax=ax, subplots=True, color='#565656')
            ax.legend()

            if self.prediction_period >= 24:
                x_axis_format = '%d/%m'
                x_axis_name = 'Day'
            else:
                x_axis_format = '%H:%M'
                x_axis_name = 'Hour'

            ax.xaxis.set_major_formatter(mdates.DateFormatter(x_axis_format))

            plt.title('Currency: {}, Prediction period: {}h. Prediction vs actual. '.format(self.currency,
                                                                                            self.prediction_period))

            plt.legend()
            plt.xlabel(x_axis_name)
            plt.ylabel('Close')
            plt.savefig('{}/{} {}.png'.format(self.image_dir, name, self.currency).replace(' ', '_'),
                        bbox_inches='tight', dpi=300)
            plt.gcf().clear()

            # plt.gcf().clear()
            # plt.show()

    def plot_feature_importance(self):
        for sum_dict in self.summary:
            name = sum_dict['name'] + ' feature-importance'
            print('Plotting feature importance ' + name)

            regressor = sum_dict['regressor']
            # Plot feature importance
            if hasattr(regressor, 'feature_importances_'):
                feature_importance = regressor.feature_importances_
            else:
                feature_importance = np.mean([tree.feature_importances_ for tree in regressor.estimators_], axis=0)

                # feature_importance = np.asarray(sorted(feature_importance))
                # if len(feature_importance) > 40:
                # feature_importance = feature_importance[:10000]
                # indices = np.argsort(feature_importance)[::-1]
                # for f in range(self.x.shape[1]):
                # print('%d. feature %d (%f)' % (f + 1, indices[f], feature_importance[indices[f]]))
                # feature_importance[indices[f]]

            # make importances relative to max importance
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            if len(feature_importance) > 35:
                sorted_idx = np.argsort(feature_importance)[-35:]
            else:
                sorted_idx = np.argsort(feature_importance)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 2)
            pos = np.arange(sorted_idx.shape[0]) + .2
            plt.barh(pos, feature_importance[sorted_idx], align='center', color='coral')
            ax = plt.gca()
            ax.set_alpha(0.8)

            for i in ax.patches:
                # get_width pulls left or right; get_y pushes up or down
                ax.text(i.get_width() + 1, i.get_y() + .10,
                        str(round((i.get_width() / feature_importance.max()) * 100, 4)) + '%', fontsize=7,
                        color='dimgrey')

            plt.yticks(pos, self.feature_names[sorted_idx])
            plt.xlabel('Relative Importance')  # (feature_importance / max_feature_importance) * 100
            plt.title('Currency: {}, Prediction period: {}h. Feature importance.'.format(self.currency,
                                                                                        self.prediction_period))

            plt.savefig('{}/{} {}.png'.format(self.image_dir, name, self.currency).replace(' ', '_'),
                        bbox_inches='tight')
            plt.gcf().clear()
            # plt.show()

    def plot_features(self, used_features, removed_features, cluster_currencies):
        name = 'Features'
        print('Plotting {}'.format(name))
        fig, ax = plt.subplots()
        # Hide axes
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        pd_dict = {'Features': used_features, 'Removed features': removed_features,
                   'Cluster currencies': cluster_currencies}
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pd_dict.items()]))
        df.fillna(value=' ', inplace=True)

        ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        plt.savefig('{}/{} {}.png'.format(self.image_dir, name, self.currency).replace(' ', '_'), bbox_inches='tight')
        #  plt.show()
        plt.gcf().clear()

    def write_latex_table(self):
        df = pd.DataFrame(columns=['Regressor', 'Prediction period', 'MAE', 'MSE', 'R2', 'Inc. cluster features'])

        for i, sum_dict in enumerate(self.summary):
            Regressor = sum_dict['name']
            MSE = sum_dict['MSE']
            MAE = sum_dict['MAE']
            R2 = sum_dict['R2']
            inc_cluster = 'yes' if len(self.cluster_currencies) > 0 else 'no'

            df.loc[i] = [Regressor,
                         '{}h'.format(self.prediction_period),
                         '{0:.4f}'.format(MAE),
                         '{0:.4f}'.format(MSE),
                         '{0:.4f}'.format(R2),
                         inc_cluster]

        with open(self.image_dir + '/results.tex', 'a') as tf:
            tf.write('\nprediction_period: {}, currency: {}'.format(self.prediction_period, self.currency))
            best_regressor = self.get_best_regressor(print_it=False)
            tf.write('\nBest regressor based on R2-MSE-MAE: {} - {}\n'.format(best_regressor['name'],
                                                                              best_regressor['score']))
            tf.write(df.to_latex(longtable=True))

    def get_best_regressor(self, print_it):
        highest_score_value = -1
        highest_score_idx = 0

        if print_it: print('**' * 10)

        for idx, sum_dict in enumerate(self.summary):
            if print_it: print(sum_dict['name'] + ' score : ' + ' {}'.format(sum_dict['score']))
            if sum_dict['score'] > highest_score_value:
                highest_score_idx = idx
                highest_score_value = sum_dict['score']

        if print_it: print('**' * 10)

        return self.summary[highest_score_idx]


if __name__ == '__main__':

    # headers_to_remove = [
    #     # 'high', 'interest_over_time_AU', 'interest_over_time_BR', 'interest_over_time_CA',
    #     # 'interest_over_time_DE', 'interest_over_time_FR', 'interest_over_time_GB', 'interest_over_time_GH',
    #     # 'interest_over_time_IN', 'interest_over_time_JP', 'interest_over_time_KE', 'interest_over_time_KR',
    #     # 'interest_over_time_NG', 'interest_over_time_RU', 'interest_over_time_SG', 'interest_over_time_US',
    #     # 'interest_over_time_VE', 'interest_over_time_ZA',  'interest_over_time_CN', 'low', 'num_retweets',
    #     # 'num_tweets', 'open','tweet_exposure', 'volume_from', 'volume_to'
    #                       ]
    # cluster_currencies = []#['NEO', 'EOS', 'Dash', 'ICON']

    # TODO: Load automagically from column explanation.py
    # variable all_headers is not used
    all_headers = [
        'close', 'date', 'high', 'interest_over_time_AU', 'interest_over_time_BR', 'interest_over_time_CA',
        'interest_over_time_DE', 'interest_over_time_FR', 'interest_over_time_GB', 'interest_over_time_GH',
        'interest_over_time_IN', 'interest_over_time_JP', 'interest_over_time_KE', 'interest_over_time_KR',
        'interest_over_time_NG', 'interest_over_time_RU', 'interest_over_time_SG', 'interest_over_time_US',
        'interest_over_time_VE', 'interest_over_time_ZA', 'interest_over_time_CN', 'low', 'num_retweets',
        'num_tweets', 'open', 'tweet_exposure', 'volume_from', 'volume_to'
    ]

    data_files = ['crypto_data_week_9.csv',
                  'crypto_data_week_10.csv',
                  'crypto_data_week_11.csv',
                  #'crypto_data_week_13.csv',
                  #'crypto_data_week_14.csv'
                  ]

    testing_files = ['crypto_data_week_13.csv',
                     'crypto_data_week_14.csv',
                     'crypto_data_week_15.csv'
                     ]

    start_time_date_week_number = datetime.strptime('2018-02-28 08:00:00.00', '%Y-%m-%d %H:%M:%S.%f')

    selected_currency = 'Bitcoin'

    # data_files = config['data_files']
    # test_data_files = config['test_data_files']
    # cluster_currencies = config['cluster_currencies']
    # prediction_period = config['prediction_period']
    # headers_to_remove = config['headers_to_remove']

    configurations = [
        # Prediction: Bitcoin, 1 hour ahead, all Bitcoin features considered
        {
            'headers_to_remove': [],
            'cluster_currencies': [],
            'prediction_period': 1,
            'data_files': data_files,
            'test_data_files': testing_files,
        },
        # # Prediction: Bitcoin, 1 hour ahead, all Bitcoin features and selected features
        # # from cluster currencies considered
        {
            'headers_to_remove': [],
            'cluster_currencies': ['NEO', 'EOS', 'Dash', 'ICON'],
            'prediction_period': 1,
            'data_files': data_files,
            'test_data_files': testing_files,
        },
        # Prediction: Bitcoin, 24 hours ahead, all features considered
        {
            'headers_to_remove': [],
            'cluster_currencies': [],
            'prediction_period': 24,
            'data_files': data_files,
            'test_data_files': testing_files,
        },
        # Prediction: Bitcoin, 24 hours ahead, all Bitcoin features and selected features
        # from cluster currencies considered
        {
            'headers_to_remove': [],
            'cluster_currencies': ['NEO', 'EOS', 'Dash', 'ICON'],
            'prediction_period': 24,
            'data_files': data_files,
            'test_data_files': testing_files,
        },
    ]

    csv_dir = os.path.dirname(os.path.abspath(__file__)) + '/csv_files'

    data_preprocessor = DataPreprocessor(configurations=configurations,
                                         csv_dir=csv_dir,
                                         currency=selected_currency,
                                         start_time_date=start_time_date_week_number)
    processed_data_list = data_preprocessor.process_data()

    for processed_data in processed_data_list:
        x = processed_data['x']
        y = processed_data['y']
        x_df = processed_data['x_df']
        test_x_df = processed_data['test_x_df']
        test_df = processed_data['test_df']
        headers_to_remove = processed_data['headers_to_remove']
        cluster_currencies = processed_data['cluster_currencies']
        prediction_period = processed_data['prediction_period']
        feature_names = processed_data['feature_names']

        used_features = list(filter(lambda a: a not in headers_to_remove and a not in 'date', all_headers))

        currency_predictor = CurrencyPredictor(currency=selected_currency,
                                               csv_dir=csv_dir,
                                               prediction_period=prediction_period,
                                               start_time_date=start_time_date_week_number,
                                               headers_to_remove=headers_to_remove,
                                               cluster_currencies=cluster_currencies,
                                               feature_names=feature_names,
                                               x=x,
                                               y=y,
                                               x_df=x_df,
                                               test_x_df=test_x_df,
                                               test_df=test_df)

        currency_predictor.write_latex_table()
        currency_predictor.plot_regressor_error()
        print()
        currency_predictor.plot_feature_importance()
        print()
        currency_predictor.plot_predictions(with_untrained_data=False)
        print()
        currency_predictor.plot_predictions(with_untrained_data=True)

        print()
        currency_predictor.plot_last_predictions()
        print()
        currency_predictor.plot_regressor_results(val_name='R2', color='#2D898B')
        currency_predictor.plot_regressor_results(val_name='MAE', color='#3587A4')
        currency_predictor.plot_regressor_results(val_name='MSE', color='#88CCF1')
        currency_predictor.plot_regressor_results(val_name='score', color='#C3DFEE')
        print()
        currency_predictor.plot_features(used_features, headers_to_remove, cluster_currencies)
        print()

        best_prediction_dict = currency_predictor.get_best_regressor(print_it=True)
        print(22 * '*' + ' BEST REGRESSOR ' + 22 * '*')
        print('Name: {}'.format(best_prediction_dict['name']))
        print('R2: {0:.4f}'.format(best_prediction_dict['R2']))
        print('MAE: {0:.4f}'.format(best_prediction_dict['MAE']))
        print('MSE: {0:.4f}'.format(best_prediction_dict['MSE']))
        print('{0:.3f}%'.format(best_prediction_dict['accuracy']))
        print(60 * '*')
