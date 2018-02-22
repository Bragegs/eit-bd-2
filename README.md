# eit-bd-2

Must haves:

* Python 3

Should haves:

* Virtualenv


# How to run

Activate your virtual environment. (Differs from virtualenv distribution)
```
source activate virtualenv
```
Install all project dependecies (third party) listed in requirements file
```
pip install -r requirements.txt
```
Run python programs (example)
```
python ./data_loaders/weekly_data.py
```

#### Currency predictor

To run the currency predictor the 'crypto_data.csv' must be downloaded and placed in folder as 'currency_predictor.py'.


Run currency predictor
```
python currency_predictor.py
```

Two parameters to you can change:

* Currency (Upper cased first letter currency found in the 'crypto_data.csv')
* Prediction period (How many hours ahead the predictor should predict)

(Scroll to bottom of currency_predictor.py to replace values)