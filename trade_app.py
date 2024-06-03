
import datetime
import logging
import sqlite3 as db
import pandas as pd
from pprint import pprint
import pytorch_lightning


from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient

from Keys import *
from environments.live_environments import LiveTradingEnv
from neuralforecast.core import NeuralForecast
from ray.rllib.policy.policy import Policy
from utils.rendering import LiveRenderer
from utils.utils import sharpe_reward
import threading
pytorch_lightning._logger.setLevel(0)

import logging

logging.getLogger('lightning').setLevel(0)
import time

trade_target='ETH/USDT'
trade_interval='1h'
api_key=binanace_api_key
api_secret=binance_api_secret
history_path='Trade_history/trade.db'

conn=db.connect(history_path)

model=NeuralForecast.load("forecasting_model",verbose=False)

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

live_env=LiveTradingEnv(**live_env_config)

time_format="%m-%d-%Y %H:%M"


def on_start(message):
        obs,info=live_env.reset(reset_account=False)
        start_time=datetime. datetime.now().strftime(time_format)
        print (f'START trading session at {start_time}')
        action,_,states=live_env.agent.compute_single_action(obs,explore=False)
        obs, reward, terminated, truncated, info=live_env.live_step(action,wait=False)

def on_close(_,message):
        end_time=datetime. datetime.now().strftime(time_format)
        print (f'FINISH trading session at {end_time}')
        live_env.reset_account()
        
        
# my_client = SpotWebsocketStreamClient(
#                                     on_close=on_close,
#                                     on_open=on_start,
#                                     on_error=on_close,
#                                     on_message=live_env.stream_step,
#                                     stream_url='wss://testnet.binance.vision')

class LiveTradingApp(LiveRenderer):
        def __init__(self,render_logs_dir,trading_env):
                super().__init__(render_logs_dir,
                                 base_asset=trading_env.base_asset,
                                 quote_asset=trading_env.quote_asset
                                 )
                self.trading_env=trading_env
                

        def run(self,**kwargs):
                my_client = SpotWebsocketStreamClient(
                        on_close=on_close,
                        on_open=on_start,
                        on_error=on_close,
                        on_message=self.trading_env.stream_step,
                        stream_url='wss://testnet.binance.vision')
                app=threading.Thread(target=super().run,kwargs=kwargs)
                app.start()
                # reconnect every 12 hours
                while True:
                        my_client.kline(symbol=self.trading_env.symbol,interval=self.trading_env.time_frame)
                        wait_time=pd.Timedelta(hours=12).total_seconds()
                        time.sleep(wait_time)
                        


if __name__ == '__main__':
        #trade for a full week the socket manager will time out after 24 hours

        app=LiveTradingApp('Trade_history/trade.db',live_env)
        app.run(host='0.0.0.0',port=5000)
        # app=threading.Thread(target=trade_monitor.run)
