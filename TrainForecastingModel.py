

import datetime
import warnings

import matplotlib.pyplot as plt
# import MultiTrade
import numpy as np
import pandas as pd
from IPython.display import display
from neuralforecast.auto import AutoBiTCN, AutoNBEATS, AutoTFT
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import RMSE


from configs import defaults
from utils.utils import build_market_image

warnings.filterwarnings("ignore")
import optuna


optuna.logging.set_verbosity(optuna.logging.WARNING)

COIN_PAIRS=defaults.COIN_PAIRS
target_pair=defaults.target_pair
time_frame=defaults.time_frame



horizon = 7
context_length=5*horizon
start_dt=pd.Timestamp('2024-01-01')
split_dt=datetime.datetime.now()-pd.Timedelta(days=7)
end_dt=datetime.datetime.now()
split_buffer=pd.Timedelta(unit=time_frame[-1],value=context_length)



data=build_market_image(target_pair=target_pair,time_frame=time_frame,axis=0)


train_data=data.groupby('symbol').apply(lambda x: x[start_dt:split_dt])
test_data=data.groupby('symbol').apply(lambda x: x[split_dt-split_buffer:end_dt])

train_data=train_data.reset_index(level=0,drop=True).reset_index()
test_data=test_data.reset_index(level=0,drop=True).reset_index()

data['symbol'].unique()
id_col='symbol'


front=['y','ds','symbol']
exo_gen_cols=data.filter(like='feature_').columns.tolist()
cols=front+[c for c in exo_gen_cols if c not in front]

train_data=train_data[cols]
test_data=test_data[cols]



forecast_horizon=6
backend='optuna'
BiTCN_config=AutoBiTCN.get_default_config(h=forecast_horizon,backend=backend)

TFT_config=AutoTFT.get_default_config(h=forecast_horizon,backend=backend)
NBEATS_config=AutoNBEATS.get_default_config(h=forecast_horizon,backend=backend)


BiTCN_MODEL= AutoBiTCN(h=forecast_horizon,
                  loss=RMSE(),
                  config=BiTCN_config,
                  search_alg=optuna.samplers.TPESampler(),
                  backend='optuna',
                  num_samples=10)

TFT_MODEL= AutoTFT(h=forecast_horizon,
                  loss=RMSE(),
                  config=TFT_config,
                  search_alg=optuna.samplers.TPESampler(),
                  backend='optuna',
                  num_samples=10)

NBEATS_MODEL= AutoNBEATS(h=forecast_horizon,
                  loss=RMSE(),
                  config=NBEATS_config,
                  search_alg=optuna.samplers.TPESampler(),
                  backend='optuna',
                  num_samples=10)

models=[BiTCN_MODEL,TFT_MODEL,NBEATS_MODEL]



model = NeuralForecast(models=models,
                    local_scaler_type='robust',
                    
                     freq='1h')


model.fit(train_data,
        val_size=horizon,
        time_col='ds',                    
        target_col='y',
        id_col=id_col,
        
        )



model.save(defaults.forecasting_model_path,overwrite=True)


