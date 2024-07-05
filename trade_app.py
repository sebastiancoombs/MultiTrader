
import datetime
import logging
import sqlite3 as db
import threading
from pprint import pprint

import pandas as pd
import pytorch_lightning
from alpaca.data.live.crypto import CryptoDataStream
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from environments.live_environments import AlpacaTradingEnv,OandaTradingEnv
from Keys import *
from neuralforecast.core import NeuralForecast
from ray.rllib.policy.policy import Policy

from utils.utils import sharpe_reward

pytorch_lightning._logger.setLevel(0)

import logging

logging.getLogger('lightning').setLevel(0)
import time

trade_target='ETH/USD'
trade_interval='1h'
api_key=alpaca_api_key
api_secret=alpaca_api_secret
history_path='Trade_history/trade.db'

conn=db.connect(history_path)

model=NeuralForecast.load("FX_forecasting_model",verbose=False)

agent_dir='Agent/final_checkpoints/policies/default_policy'
agent= Policy.from_checkpoint(agent_dir)

live_env_config=dict(
                name=f'{trade_target}_live',
                model=model,
                agent=agent,
                api_key=api_key,
                api_secret=api_secret,
                test_net=True,
                restore_trading=True,
                target_symbol=trade_target,
                time_frame=trade_interval,
                reward_function=sharpe_reward,
                positions = [ -.5,-.25,.25, .5], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
                trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
                borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
                max_episode_duration=168,
                verbose=0,
                )

live_env=OandaTradingEnv(**live_env_config)
obs,info=live_env.reset(reset_account=True)
time_format='%I:%M %p %m-%d-%Y'
                        


if __name__ == '__main__':
        start_time=datetime. datetime.now().strftime(time_format)
        print (f'START trading session at {start_time}')

        sock=CryptoDataStream(api_key=api_key,secret_key=api_secret,raw_data=True)
        sock.subscribe_bars(live_env.live_stream_step,trade_target)
        sock.run()
