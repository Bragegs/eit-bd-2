import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

import time
from datetime import datetime
from datetime import timedelta
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from datetime import datetime, timedelta
from sklearn.ensemble import *
#import xgboost as xgb

sns.set()


def create_data_frame_for_currency(currency):
    df = pd.read_csv('./crypto_data.csv')  # , parse_dates=['date'], index_col='date'

    # df.drop(['name'], axis=1, inplace=True)
    headers = list(df.drop(['name'], axis=1, inplace=False))
    currencies = df.set_index('name').T.to_dict('list')

    ethereum = currencies[currency]

    currency_dict = dict(zip(headers, ethereum))
    # print(headers)
    # print(currency_dict.keys())
    # exit()
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

        # if new_dict['close'] == 1:
        #     print(key)
        #     print(val)
        # print('hei')
        # exit()

        new_dict['hourly_relative_change'] = currency_dict['h_r_c{}'.format(str(counter).zfill(4))]
        # new_dict['close'] = currency_dict['close_{}'.format(str(counter).zfill(4))]
        # i_o_t_CN_0053
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


# df = create_data_frame_for_currency('Ethereum')
# df.to_csv('ethereum.csv', index=False)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=np.nan)
df = pd.read_csv('./ethereum.csv', parse_dates=['date'], index_col='date')
df.fillna(0.0)

# df['close'].plot(figsize=(12,6),label='Close')
# df['close'].rolling(window=7).mean().plot(label='7 Day Avg')
# plt.legend()
# plt.show()


original_input_values = df.loc[:, df.columns != 'date'].values.reshape((-1,len(list(df))))


scaler = MinMaxScaler()
t = list(set(list(df)) - set('date'))
df_copy = df#.drop('date', axis=1)
df[t] = scaler.fit_transform(df[t])

# df['close'].plot(figsize=(12,6),label='Close')
# df['close'].rolling(window=7).mean().plot(label='7 Day Avg')
# plt.legend()
# plt.show()

#print(df.iloc[:, 0].values.reshape((-1, 1)))
#exit()

# print(df.loc[:, df.columns != 'date'].values.reshape(-1, len(list(df))))
# exit()
period = 1 # How far would you like to predict?? 1 hour
#close_reshaped = df['close'].values.reshape((-1, 1))
#close_df = pd.DataFrame(close_reshaped)
df['Price_After_period'] = df['close'].shift(-period)
df.drop(df.tail(1).index,inplace=True)
df.to_csv('norm.csv', index=False)
# print(df)
# exit()
 # <- normalized -> Real numbersdf.iloc[:, 0].values.reshape((-1,1)))#
#normalized.dropna(inplace=True)


# #df.loc[:, df.columns != 'date'].values.reshape(-1, len(list(df)))
# minmax = MinMaxScaler().fit(df.iloc[:, 0].values.reshape((-1,1)))
# close_normalize = minmax.transform(df.iloc[:, 0].values.reshape((-1, 1)))
# normalized = pd.DataFrame(close_normalize) # <- normalized -> Real numbersdf.iloc[:, 0].values.reshape((-1,1)))#
# normalized['Price_After_period'] = normalized[0].shift(-period)
# normalized.dropna(inplace=True)
# normalized.to_csv('norm.csv', index=False)
#
# df.drop('close', axis=1, inplace=True)
# df.drop(df.tail(1).index,inplace=True)
#
# X = normalized.drop('Price_After_period', axis=1)
#
# df['close'] = pd.Series(list(X[0]), index=df.index)
#
X_df = df.drop('Price_After_period', axis=1)
X = df.loc[:, X_df.columns != 'date'].values.reshape(-1, len(list(X_df)))

y = df['Price_After_period']

#print(X.shape)

train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=101)

#print(test_X.shape)
#print(original_input_values.shape)
#exit()
classifiers = {
    'LinearRegression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=101),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=500, learning_rate=0.1),
    'Bagging Regressor': BaggingRegressor(n_estimators=500),
    'AdaBoost Regressor': AdaBoostRegressor(n_estimators=500, learning_rate=0.1),
    'Extra Tree Regressor': ExtraTreesRegressor(n_estimators=500),

}

summary = list()
for name, clf in classifiers.items():
    #print(name)
    nada = clf.fit(train_X, train_Y)
    score = nada.score(test_X, test_Y)
    prediction = clf.predict(test_X)
    cost_values = X_df[-167-period:]  # We'll take the last period elements to make our predictions on them
    #print(cost_values)
    #print(cost_values)
    forecast = clf.predict(cost_values)
    print(forecast)
    accuracy = score * 100
    mae = mean_absolute_error(test_Y, clf.predict(test_X))
    mse = mean_squared_error(test_Y, clf.predict(test_X))
    r2 = r2_score(test_Y, clf.predict(test_X))

    print(f'R2: {r2:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'MSE: {mse:.2f}')
    print('{0:.4f}%'.format(accuracy))
    print()

    summary.append({
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'accuracy': accuracy,
        'prediction': prediction,
        'name': name,
        'forecast': forecast,
        'close': cost_values
    })


# bagging_out = bagging.predict(test_X)
# et_out = et.predict(test_X)
# gb_out = gb.predict(test_X)
# rf_out = rf.predict(test_X)
#
#original_df = pd.read_csv('./ethereum.csv', parse_dates=['date'], index_col='date')
#print(original_df.shape)

#df_copy.loc[168] = [0 for n in range(len(list(df_copy)))]
print(df_copy.shape)

for sum_dict in summary:
    print(len(sum_dict['forecast']))

    name = sum_dict['name'] + '_forecast'
    df_copy[name] = sum_dict['forecast']
    df_copy['close'].plot(figsize=(12,6),label='Close')
    df_copy[name].plot(figsize=(12,6),label=name)

    #df['close'].rolling(window=7).mean().plot(label='7 Day Avg')
    plt.legend()
    plt.show()

# forecast_df = pd.DataFrame(summary)
# print(forecast_df['forecast'])
# #exit()
#
# forecast_df['close'].plot(figsize=(12,6),label='Close')
# forecast_df['forecast'].plot(label='forecast')
#
# #forecast_df[]
#
# exit()
# predictions = [(sum_dict['prediction'], sum_dict['name']) for sum_dict in summary]
# names = [x[1] for x in predictions] + ['test']
#
# stack_predict = np.vstack([x[0] for x in predictions] + [test_Y]).T
# corr_df = pd.DataFrame(stack_predict, columns=names)
# plt.figure(figsize=(10,5))
# sns.heatmap(corr_df.corr(), annot=True)
# plt.show()




# df = pd.read_csv('./ethereum.csv', parse_dates=['date'], index_col='date')
#
# X = df.iloc[:, 0].values.reshape((-1,1))
# X_30 = X[-period:]  #  We'll take the last period elements to make our predictions on them
# forecast = reg.predict(X_30)

# print(corr_df.head())
# for name in names:
#     corr_df[name] = minmax.inverse_transform(corr_df[name].values.reshape((-1, 1))).flatten()
# print(corr_df.head())
#
# print(corr_df.head())
# corr_df.ada = minmax.inverse_transform(corr_df.ada.values.reshape((-1,1))).flatten()
# corr_df.bagging = minmax.inverse_transform(corr_df.bagging.values.reshape((-1,1))).flatten()
# corr_df.et = minmax.inverse_transform(corr_df.et.values.reshape((-1,1))).flatten()
# corr_df.gb = minmax.inverse_transform(corr_df.gb.values.reshape((-1,1))).flatten()
# corr_df.rf = minmax.inverse_transform(corr_df.rf.values.reshape((-1,1))).flatten()
# corr_df.test = minmax.inverse_transform(corr_df.test.values.reshape((-1,1))).flatten()
# print(corr_df.head())
# #
# # params_xgd = {
# #     'max_depth': 7,
# #     'objective': 'reg:linear',
# #     'learning_rate': 0.033,
# #     'n_estimators': 10000
# #     }
# # clf = xgb.XGBRegressor(**params_xgd)
# # stack_train = np.vstack([ada.predict(train_X),
# #                            bagging.predict(train_X),
# #                            et.predict(train_X),
# #                            gb.predict(train_X),
# #                           rf.predict(train_X)]).T
# #
# # stack_test = np.vstack([ada.predict(test_X),
# #                            bagging.predict(test_X),
# #                            et.predict(test_X),
# #                            gb.predict(test_X),
# #                           rf.predict(test_X)]).T
# #
# # clf.fit(stack_train, train_Y, eval_set=[(stack_test, test_Y)],
# #         eval_metric='rmse', early_stopping_rounds=20, verbose=True)
#
# df = pd.read_csv('./ethereum.csv', parse_dates=['date'], index_col='date')
#
# forecast=rf.predict(df['close'])
