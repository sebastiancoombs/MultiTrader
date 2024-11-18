from configs import spot_defaults

import datetime
from utils.utils import prepare_forecast_data,build_market_image,train_test_split_data
from neuralforecast.core import NeuralForecast
import pandas as pd
import numpy as np
from utils.reward_functions import sharpe_reward_function
from environments.environments import NormTradingEnvironment
import copy
## CHANGE THIS DEFAULT TO CHANGE PARAMS FROM CONFIGS
from configs import spot_defaults as defaults
from IPython.display import display
COIN_PAIRS=defaults.COIN_PAIRS
target_pair=defaults.target_pair
time_frame=defaults.time_frame
model_path=defaults.forecasting_model_path
env_config=defaults.env_config
DATA_DIR=defaults.DATA_DIR

def n_trades(history):
        return sum(np.abs(np.diff(history['position'])))

def make_envs(reward_function):
    model=NeuralForecast.load(model_path)
    ohlv_data=build_market_image(data_dir=DATA_DIR,target_pair=target_pair,time_frame='1h',axis=0,verbose=1,only_target=True)
    display(ohlv_data.head())
    print(ohlv_data.columns)
    
    data=prepare_forecast_data(model,ohlv_data,plot=True)
    data.index=pd.to_datetime(data['ds'])
    train_data,test_data=train_test_split_data(data,n_days=30)
    
    train_env_config=copy.deepcopy(env_config)
    test_env_config=copy.deepcopy(env_config)
    train_env_config['name']=f'{target_pair}_train'
    train_env_config['reward_function']=reward_function
    train_env_config['df']=train_data

    
    
    test_env_config['name']=f'{target_pair}_test'
    test_env_config['reward_function']=reward_function
    test_env_config['df']=test_data
    train_env=NormTradingEnvironment(**train_env_config)
    test_env=NormTradingEnvironment(**test_env_config)
    train_env.add_metric(name="position_change", function=n_trades)
    train_env.add_metric(name="position_change", function=n_trades)
    return train_env,test_env


    




