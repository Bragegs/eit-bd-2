from pymarketcap import *
import pandas as pd

coinmarketcap = Pymarketcap()
all_coinmarketcap_coins = coinmarketcap.ticker(limit=0)

df = pd.DataFrame(all_coinmarketcap_coins)
df = df.sort_values('market_cap_usd', ascending=False)
sorted_crypto_currencies = list(df.T.to_dict().values())

filtered_crypto_currencies = sorted_crypto_currencies[:25]

for cc in filtered_crypto_currencies:
    print(cc['name'])
