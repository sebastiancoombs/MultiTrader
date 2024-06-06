
import sqlite3 as db
from utils.rendering import LiveRenderer

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
# from alpaca.data.bars import
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from utils.clients import AlpacaClient
from utils.mappings import alpaca_stream_message_map,alpaca_stream_col_map,symbol_map
from alpaca.data.live.crypto import CryptoDataStream
from pandas.tseries.offsets import Hour

import sys
import json
import datetime as dt
import Keys
import asyncio
import threading
import asyncio

api_key=Keys.alpaca_api_key
api_secret=Keys.alpaca_api_secret
sock=CryptoDataStream(api_key=api_key,secret_key=api_secret,raw_data=True)
pair="ETH/USD"
symbol=pair.replace('/','')
print()
async def on_bar(message):
    print('BAR',message)

async def _stream_data_handler(message):
        # print(message,type(message))
        
        bar_time=message.get('t')
        if bar_time:
            bar_time=pd.Timestamp(bar_time.seconds,unit='s')
            message['t']=bar_time


        data=pd.DataFrame([message])
        data=data[[c for c in alpaca_stream_col_map.keys() if c in data.columns]]
        data=data.rename(columns=alpaca_stream_col_map)  

        data['bar_type']=data['bar_type'].map(alpaca_stream_message_map)

        data["ds"]=data["date_close"].copy()
        data=data.set_index('date_close')
        data['symbol'] = data['symbol'].str.replace('/','')
        data['unique_id']=symbol_map[symbol]


        with db.connect('Trade_history/trade.db') as conn:
            data.to_sql(f'{symbol}_candle_history',conn,if_exists='replace',index=True)
        print(data)

        do_trade=bar_time.minute==0
        print(bar_time.strftime("%H:%M"),"####### is on hour?",do_trade)
        return do_trade

sock.subscribe_bars(_stream_data_handler,pair)
# sock.subscribe_quotes(on_quote,pair)
sock.run()
