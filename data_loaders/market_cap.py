from pymarketcap import *
import pandas as pd

coinmarketcap = Pymarketcap()


def get_top_x_market_cap(x):
    all_coinmarketcap_coins = coinmarketcap.ticker(limit=0)

    df = pd.DataFrame(all_coinmarketcap_coins)
    df = df.sort_values('market_cap_usd', ascending=False)
    sorted_crypto_currencies = list(df.T.to_dict().values())

    filtered_crypto_currencies = sorted_crypto_currencies[:x]

    print('Top {} market cap crypto currencies'.format(x))
    for cc in filtered_crypto_currencies:
        print('"{}",'.format(cc['name']))

    return filtered_crypto_currencies
