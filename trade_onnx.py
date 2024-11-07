
import datetime
import logging
import sqlite3 as db
import threading
import time
import warnings
from pprint import pprint

import pandas as pd
import pytorch_lightning
from neuralforecast.core import NeuralForecast

from environments.live_environments import AlpacaTradingEnv, OandaTradingEnv
from Keys import *
from utils.utils import sharpe_reward

import onnxruntime as ort
pytorch_lightning._logger.setLevel(0)
logging.getLogger('lightning').setLevel(0)


trade_target='USD_JPY'

trade_interval='1h'
api_key=oanda_paper_api_key
account_id=oanda_paper_account
history_path='Trade_history/trade.db'

conn=db.connect(history_path)

model=NeuralForecast.load("FX_forecasting_model",verbose=False)

agent_dir=f'FX_Agent/{trade_target}_production_agent_onnx/'

agent = ort.InferenceSession(agent_dir+'model.onnx')
# agent= Algorithm.from_checkpoint(agent_dir)

live_env_config=dict(
                name=f'{trade_target}_live',
                model=model,
                agent=agent,
                api_key=api_key,
                account_id=account_id,
                test_net=True,
                restore_trading=False,
                target_symbol=trade_target,
                time_frame='1h',
                reward_function=sharpe_reward,
                positions = [ -.5,-.25,0,.25, .5], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
                trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
                borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
                max_episode_duration=168,
                verbose=0,
                onnx_model=True
                )
live_env=OandaTradingEnv(**live_env_config)
obs,info=live_env.reset(reset_account=True)
time_format='%I:%M %p %m-%d-%Y'
                        


if __name__ == '__main__':
        live_env.live_trade()
        start_time=datetime. datetime.now().strftime(time_format)
        print (f'START trading session at {start_time}')
        while True:
                try:
                        live_env.stream_trade()
                except Exception as e:
                        print(e)