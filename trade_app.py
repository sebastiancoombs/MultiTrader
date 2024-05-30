
import ast
import datetime
import json
import logging
import sqlite3 as db
import threading
import warnings
from pprint import pprint

import pandas as pd
import pytorch_lightning
from binance.spot import Spot
# WebSocket API Client
from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from IPython.display import display
from Keys import *
from MultiTrade.live_environments import LiveTradingEnv
from neuralforecast.core import NeuralForecast
from ray.rllib.policy.policy import Policy
from utils.rendering import LiveRenderer
from utils.utils import (binanace_col_map, build_market_image, sharpe_reward,
                         symbol_map)

pytorch_lightning._logger.setLevel(0)

import logging

logging.getLogger('lightning').setLevel(0)
import time

trade_target='BTC/USDT'
api_key=binanace_api_key
api_secret=binance_api_secret


symbol="ETHUSDT"
history_path='Trade_history/trade.db'

conn=db.connect(history_path)

model=NeuralForecast.load("forecasting_model",verbose=False)

agent_dir='Agent/final_checkpoints/policies/default_policy'
agent= Policy.from_checkpoint(agent_dir)

live_env_config=dict(
                name='ETHUSDT_train',
                model=model,
                agent=agent,
                api_key=api_key,
                api_secret=api_secret,
                test_net=True,
                restore_trading=True,
                target_symbol='ETH/USDT',
                time_frame='1h',
                reward_function=sharpe_reward,
                positions = [ -.5,-.25,.25, .5], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
                trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
                borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
                max_episode_duration=168,
                verbose=0,
                )

live_env=LiveTradingEnv(**live_env_config)

live_env.reset(reset_account=False)


def on_start(message):
        obs,info=live_env.reset(reset_account=False)
        start_time=datetime. datetime.now().strftime("%m-%d-%Y %H:%M")
        print (f'START trading session at {start_time}')
        action,_,states=live_env.agent.compute_single_action(obs,explore=False)
        obs, reward, terminated, truncated, info=live_env.live_step(action,wait=False)

def on_close(_):
        end_time=datetime. datetime.now().strftime("%m-%d-%Y %H:%M")
        print (f'FINISH trading session at {end_time}')
        live_env.reset_account()
        
        
my_client = SpotWebsocketStreamClient(
                                    on_close=on_close,
                                    on_open=on_start,
                                    on_error=on_close,
                                    on_message=live_env.stream_step,
                                    stream_url='wss://testnet.binance.vision')



if __name__ == '__main__':
        #trade for a fulll week the socket manager will time out after 24 hours
        for i in range(7):
                # Subscribe to a single symbol stream
                my_client = SpotWebsocketStreamClient(
                                        on_close=on_close,
                                        on_open=on_start,
                                        on_error=on_close,
                                        on_message=live_env.stream_step,
                                        stream_url='wss://testnet.binance.vision')
                trade_monitor=LiveRenderer('Trade_history/trade.db')
                app=threading.Thread(target=trade_monitor.run)
                wait_time=pd.Timedelta(hours=24).total_seconds()
                my_client.kline(symbol="ETHUSDT",interval="1h")
                app.start()
                print('waiting to reconnect')
                time.sleep(wait_time)
