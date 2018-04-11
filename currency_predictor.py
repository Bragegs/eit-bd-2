import numpy as np
import matplotlib
import csv
import os

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
#sns.set()
sns.set(style='ticks', palette='Set2')
sns.despine()


class CurrencyPredictor:
    x = None
    y = None

    x_df = None  # Data frame of x - features (instead of numpy array)

    regressors = {
        #'LinearRegression': LinearRegression(),
        ('Random Forest Regressor', 'RFR'): RandomForestRegressor(n_estimators=500, random_state=101),
        ('Gradient Boosting Regressor', 'GBR'): GradientBoostingRegressor(n_estimators=500, learning_rate=0.1),
        ('Bagging Regressor', 'BR'): BaggingRegressor(n_estimators=500),
        ('AdaBoost Regressor', 'ABR'): AdaBoostRegressor(n_estimators=500, learning_rate=0.1),
        ('Extra Tree Regressor', 'ETR'): ExtraTreesRegressor(n_estimators=500),
    }

    summary = list()
    num_days_of_data = None

    image_dir = os.path.dirname(os.path.abspath(__file__)) + '/images'
    start_time_date = None

    csv_dir = os.path.dirname(os.path.abspath(__file__)) + '/csv_files'

    feature_names = []

    def __init__(self, currency,
                 data_files,
                 create_single_currency_data_set=False,
                 prediction_period=1,
                 start_time_date=None,
                 headers_to_remove=None,
                 cluster_currencies=None):
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)

        self.currency = currency
        self.prediction_period = prediction_period
        self.data_files = data_files
        self.num_days_of_data = len(data_files) * 7
        self.start_time_date = start_time_date
        self.image_dir += '/{}'.format(currency)
        self.headers_to_remove = headers_to_remove
        self.cluster_currencies = cluster_currencies

        if not os.path.isdir(self.image_dir):
            os.makedirs(self.image_dir)

        if create_single_currency_data_set:
            #  We have to create the data set
            self.df = self.create_data_frame_for_currency(self.currency, self.data_files, self.cluster_currencies)
            self.df = self.drop_unwanted_headers(self.df)
            self.df.fillna(0.0, inplace=True)
            self.df.to_csv('{}/{}.csv'.format(self.csv_dir, self.currency), index=False)
            self.df = pd.read_csv('{}/{}.csv'.format(self.csv_dir, self.currency), parse_dates=['date'], index_col='date')
        else:
            #  The data set is already created
            self.df = pd.read_csv('{}/{}.csv'.format(self.csv_dir, self.currency), parse_dates=['date'], index_col='date')
            self.df = self.drop_unwanted_headers(self.df)

        self.normalize_currency_data_set()
        self.create_labels_based_on_prediction_period()
        self.df_copy = self.df

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=0.2, random_state=101)

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
            #cost_values = self.x_df.drop(self.x_df.tail(self.prediction_period).index, inplace=False)
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

    def plot_predictions(self):
        for sum_dict in self.summary:
            name = sum_dict['name'] + ' forecast'
            print('Plotting {}_{}.png'.format(name, self.currency))
            values_to_predict = self.x_df.copy()
            regressor = sum_dict['regressor']

            forecast = regressor.predict(values_to_predict)
            original_cost_values = values_to_predict[['close']]

            plot_df = pd.DataFrame(columns=['Close', sum_dict['name'] + ' forecast'])

            for i in range(len(original_cost_values)):
                if i - self.prediction_period <= 0:
                    plot_df.loc[i] = [original_cost_values['close'][i], None]
                else:
                    plot_df.loc[i] = [original_cost_values['close'][i], forecast[i-self.prediction_period]]

            plot_df.plot(figsize=(12, 6), label=name,
                    title='Currency: {}, Prediction period: {} hour'.format(self.currency, self.prediction_period))
            plt.legend()
            plt.xlabel('Hour')
            plt.savefig('{}/{} {}.png'.format(self.image_dir, name, self.currency).replace(' ', '_'), bbox_inches='tight', dpi=300)
            #plt.show()

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
        #plt.show()

    def plot_last_predictions(self):
        for sum_dict in self.summary:
            name = sum_dict['name'] + ' last predictions'
            print('Plotting last predictions of {}_{}.png'.format(name, self.currency))

            num_days = self.num_days_of_data * 24
            num_last_predictions = 20

            if num_last_predictions <= self.prediction_period:
                num_last_predictions *= 2

            delta = num_days - num_last_predictions
            values_to_predict = self.x_df.drop(self.x_df.head(delta).index, inplace=False)  # self.x_df[-delta:]
            regressor = sum_dict['regressor']

            forecast = regressor.predict(values_to_predict)
            original_cost_values = values_to_predict[['close']]

            df = pd.DataFrame(columns=['Close', 'Actual value', sum_dict['name'] + ' forecast', 'Prediction start'])

            show_period = 8

            for i in range(len(original_cost_values)):
                if i == len(original_cost_values) - show_period:
                    # Make all 3 graph end/start on same place for UI
                    df.loc[i] = [original_cost_values['close'][i], original_cost_values['close'][i], original_cost_values['close'][i], original_cost_values['close'][i]]
                elif i > len(original_cost_values) - show_period:

                    # if i - self.prediction_period <= 0:
                    #     print('HEERE')
                    #     print(i)
                    #     df.loc[i] = [None, original_cost_values['close'][i], None, None]
                    # else:
                    #     df.loc[i] = [None, original_cost_values['close'][i], forecast[i - self.prediction_period], None] # Shifting forecast to actual close value

                    df.loc[i] = [None, original_cost_values['close'][i], forecast[i-self.prediction_period], None]  # Shifting forecast to actual close value
                else:
                    df.loc[i] = [original_cost_values['close'][i], None, None, None]

            df.plot(y='Actual value')

            ax = df.plot(y='Close')
            df.plot(y='Actual value', linestyle='dotted', ax=ax, subplots=True, color='#ba645b')
            df.plot(y=sum_dict['name'] + ' forecast', ax=ax, subplots=True, color='#808AB3',)
            df.plot(y='Prediction start', marker='o', ax=ax, subplots=True, color='#565656')
            ax.legend()

            plt.title('Prediction vs actual last {} points for {}'.format(show_period, self.currency))
            #df.plot(y=['Close', 'Actual value', sum_dict['name'] + ' forecast'], figsize=(12, 6), label=name,
                    #title='Currency: {}, Prediction period: {} hour'.format(self.currency, self.prediction_period))
            #df.plot(y=['Prediction point'], marker='x', linestyle='dotted')

            plt.legend()
            plt.xlabel('Hour')
            plt.savefig('{}/{} {}.png'.format(self.image_dir, name, self.currency).replace(' ', '_'), bbox_inches='tight', dpi=300)
            # plt.gcf().clear()
            # exit()
            #plt.show()

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

            #feature_importance = np.asarray(sorted(feature_importance))
            #if len(feature_importance) > 40:
            #feature_importance = feature_importance[:10000]
            #exit()
            #indices = np.argsort(feature_importance)[::-1]
            #for f in range(self.x.shape[1]):
                #print('%d. feature %d (%f)' % (f + 1, indices[f], feature_importance[indices[f]]))
                #feature_importance[indices[f]]

            # make importances relative to max importance
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            if len(feature_importance) > 35:
                sorted_idx = np.argsort(feature_importance)[-35:]
            else:
                sorted_idx = np.argsort(feature_importance)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 2)
            pos = np.arange(sorted_idx.shape[0]) + .2
            plt.barh(pos, feature_importance[sorted_idx], align='center',  color='coral')
            ax = plt.gca()
            ax.set_alpha(0.8)

            for i in ax.patches:
                # get_width pulls left or right; get_y pushes up or down
                ax.text(i.get_width() + 1, i.get_y() + .10, str(round((i.get_width() / feature_importance.max()) * 100, 4)) + '%', fontsize=7,
                        color='dimgrey')

            plt.yticks(pos, self.feature_names[sorted_idx])
            plt.xlabel('Relative Importance')  # (feature_importance / max_feature_importance) * 100
            plt.title(name)

            plt.savefig('{}/{} {}.png'.format(self.image_dir, name, self.currency).replace(' ', '_'), bbox_inches='tight')
            plt.gcf().clear()
            #plt.show()

    def plot_features(self, used_features, removed_features, cluster_currencies):
        name = 'Features'
        print('Plotting {}'.format(name))
        fig, ax = plt.subplots()
        # Hide axes
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        pd_dict = {'Features': used_features, 'Removed features': removed_features, 'Cluster currencies': cluster_currencies}
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in pd_dict.items() ]))
        df.fillna(value=' ', inplace=True)

        ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        plt.savefig('{}/{} {}.png'.format(self.image_dir, name, self.currency).replace(' ', '_'), bbox_inches='tight')
        #  plt.show()
        plt.gcf().clear()

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

    def create_labels_based_on_prediction_period(self):
        """
        Creating x and y from the data set. For each set of features x we create an label y
         features x -> close, interest_over_time_RU, hourly_relative_change, etc per hour
         label y -> The close price in 'prediction_period' steps ahead of current feature x close price

         Example: A prediction period of 1 will lead to predicting prices 1 hour ahead
        """
        # Setting next hours closing price to be the y
        self.df['Price_After_Period'] = self.df['close'].shift(-self.prediction_period)
        self.df['Price_After_Period'].fillna(0, inplace=True)
        # Remove last row
        self.df.drop(self.df.tail(self.prediction_period).index, inplace=True)
        self.x_df = self.df.drop('Price_After_Period', axis=1)
        # self.x is all columns except the y
        self.x = self.df.loc[:, self.x_df.columns != 'date'].values.reshape(-1, len(list(self.x_df)))
        self.feature_names = list(filter(lambda a: a != 'date', list(self.x_df)))
        self.feature_names = np.asarray(self.feature_names)
        # self.y is only the y column
        self.y = self.df['Price_After_Period']

    def normalize_currency_data_set(self):
        scale = MinMaxScaler()
        self.df.fillna(0.0, inplace=True)
        # Normalize all columns except the date column
        used_columns = list(set(list(self.df)) - set('date'))
        self.df[used_columns] = scale.fit_transform(self.df[used_columns])

        self.df['close'].plot(figsize=(12,6),label='Close')
        self.df['close'].rolling(window=self.num_days_of_data).mean().plot(label='{} Day Avg'.format(self.num_days_of_data))
        plt.legend()
        plt.title('avg_price_{}_days'.format(self.num_days_of_data))
        plt.savefig('{}/avg_price_{}_days.png'.format(self.image_dir, self.num_days_of_data).replace(' ', '_'), bbox_inches='tight')
        plt.gcf().clear()
        # plt.legend()
        # plt.show()

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

    def build_data_frame_dict_for_selected_currency(self, df, list_of_currency_dicts, selected_currency, date_list, cluster_currencies):

        # df.drop(['name'], axis=1, inplace=True)
        headers = list(df.drop(['name'], axis=1, inplace=False))
        currencies = df.set_index('name').T.to_dict('list')

        hour_counter = 0

        country_list = ['US', 'CA', 'SG', 'CN', 'JP', 'KR', 'IN', 'GB', 'DE', 'FR', 'ZA', 'GH', 'NG', 'AU', 'VE', 'BR', 'KE', 'RU']

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
                new_dict['{}volume_to'.format(prefix)] = currency_dict['volume_to_{}'.format(str(hour_counter).zfill(4))]
                new_dict['{}volume_from'.format(prefix)] = currency_dict['volume_from_{}'.format(str(hour_counter).zfill(4))]

                for country in country_list:
                    if 'i_o_t_{}_{}'.format(country, str(hour_counter).zfill(4)) in currency_dict:
                        new_dict['{}interest_over_time_{}'.format(prefix, country)] = currency_dict[
                            'i_o_t_{}_{}'.format(country, str(hour_counter).zfill(4))]

                num_tweets, num_retweets, tweet_exposure = self.get_twitter_data_points_from_hour(hour_counter, currency_dict)

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

        return currency_dict['tweets_{}'.format(day)], currency_dict['retweets_{}'.format(day)], currency_dict['exposure_{}'.format(day)]

    def drop_unwanted_headers(self, df):
        for unwanted_header in self.headers_to_remove:
            if unwanted_header in 'close' or unwanted_header in 'date':
                print('Can not drop header: {}'.format(unwanted_header))
                continue
            df.drop(unwanted_header, axis=1, inplace=True)

            for currency in self.cluster_currencies:
                df.drop('{}_'.format(currency) + unwanted_header, axis=1, inplace=True)

        return df

    def write_result(self):
        print('Writing results to result_log.csv')

        results_path = self.csv_dir + '/results_log.csv'

        result = {}

        best_prediction_dict = currency_predictor.get_best_regressor(print_it=False)
        result['Name'] = best_prediction_dict['name']
        result['R2'] = best_prediction_dict['R2']
        result['MAE'] = best_prediction_dict['MAE']
        result['MSE'] = best_prediction_dict['MSE']
        result['Accuracy'] = best_prediction_dict['accuracy']

        result['Currency'] = self.currency
        result['Removed_headers'] = self.headers_to_remove
        result['Prediction_period'] = self.prediction_period
        result['Data_files'] = self.data_files
        result['Cluster_currencies'] = self.cluster_currencies

        results_df = pd.read_csv(results_path)
        flags_df = pd.DataFrame([result])
        concat_df = pd.concat([results_df, flags_df])
        concat_df.to_csv(results_path, index=False)

if __name__ == '__main__':

    # TODO: Load automagically from column explanation.py
    # variable all_headers is not used
    all_headers = [
        'close', 'date', 'high', 'interest_over_time_AU', 'interest_over_time_BR', 'interest_over_time_CA',
        'interest_over_time_DE', 'interest_over_time_FR', 'interest_over_time_GB', 'interest_over_time_GH',
        'interest_over_time_IN', 'interest_over_time_JP', 'interest_over_time_KE', 'interest_over_time_KR',
        'interest_over_time_NG', 'interest_over_time_RU', 'interest_over_time_SG', 'interest_over_time_US',
        'interest_over_time_VE', 'interest_over_time_ZA',  'interest_over_time_CN', 'low', 'num_retweets',
        'num_tweets', 'open','tweet_exposure', 'volume_from', 'volume_to'
    ]

    data_files = ['crypto_data_week_9.csv',
                  'crypto_data_week_10.csv',
                  'crypto_data_week_11.csv',
                  'crypto_data_week_13.csv',
                  'crypto_data_week_14.csv'
                  ]

    start_time_date_week_number = datetime.strptime('2018-02-28 08:00:00.00', '%Y-%m-%d %H:%M:%S.%f')


    headers_to_remove = [
        # 'high', 'interest_over_time_AU', 'interest_over_time_BR', 'interest_over_time_CA',
        # 'interest_over_time_DE', 'interest_over_time_FR', 'interest_over_time_GB', 'interest_over_time_GH',
        # 'interest_over_time_IN', 'interest_over_time_JP', 'interest_over_time_KE', 'interest_over_time_KR',
        # 'interest_over_time_NG', 'interest_over_time_RU', 'interest_over_time_SG', 'interest_over_time_US',
        # 'interest_over_time_VE', 'interest_over_time_ZA',  'interest_over_time_CN', 'low', 'num_retweets',
        # 'num_tweets', 'open','tweet_exposure', 'volume_from', 'volume_to'
                          ]

    used_features = list(filter(lambda a: a not in headers_to_remove and a not in 'date', all_headers))

    # Note that higher prediction period gets higher accuracy if cluster currencies are appended to data set
    # This increase in accuracy is not as big in when predition periods are smaller

    cluster_currencies = ['NEO', 'EOS', 'Dash', 'ICON']#'Litecoin', 'Ethereum Classic']#'Cardano', 'Monero', 'Bitcoin Gold', 'Qtum', 'Zcash']

    selected_currency = 'Bitcoin'

    currency_predictor = CurrencyPredictor(currency=selected_currency,
                                           data_files=data_files,
                                           create_single_currency_data_set=True,
                                           prediction_period=24,
                                           start_time_date=start_time_date_week_number,
                                           headers_to_remove=headers_to_remove,
                                           cluster_currencies=cluster_currencies)
    currency_predictor.plot_feature_importance()
    print()
    currency_predictor.plot_predictions()
    print()
    currency_predictor.plot_last_predictions()
    print()
    currency_predictor.plot_regressor_results(val_name='R2', color='#2D898B')
    currency_predictor.plot_regressor_results(val_name='MAE', color='#3587A4')
    currency_predictor.plot_regressor_results(val_name='MSE', color='#88CCF1')
    currency_predictor.plot_regressor_results(val_name='score', color='#C3DFEE')
    print()
    currency_predictor.plot_features(used_features, headers_to_remove, cluster_currencies)


    best_prediction_dict = currency_predictor.get_best_regressor(print_it=True)
    print(22*'*' + ' BEST REGRESSOR ' + 22*'*')
    print('Name: {}'.format(best_prediction_dict['name']))
    print('R2: {0:.4f}'.format(best_prediction_dict['R2']))
    print('MAE: {0:.4f}'.format(best_prediction_dict['MAE']))
    print('MSE: {0:.4f}'.format(best_prediction_dict['MSE']))
    print('{0:.3f}%'.format(best_prediction_dict['accuracy']))
    print(60*'*')

    print()
    currency_predictor.write_result()
