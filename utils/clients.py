
import datetime
import logging
import threading
from decimal import Decimal
from typing import Optional

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest

from alpaca.data.live.crypto import CryptoDataStream

from .mappings import alpaca_time_map


class AlpacaSocketManager(threading.Thread):
    paper_url='wss://stream.data.sandbox.alpaca.markets/{version}/{stream}'
    live_url='wss://stream.data.sandbox.alpaca.markets/{version}/{stream}'
    def __init__(
                    self,
                    paper=True,
                    stream_url=None,
                    on_message=None,
                    on_open=None,
                    on_close=None,
                    on_error=None,
                    on_ping=None,
                    on_pong=None,
                    logger=None,
                    timeout=None,
                ):
        threading.Thread.__init__(self)
        if not logger:
            logger = logging.getLogger(__name__)
        self.logger = logger

        self.stream_url = self.paper_url if paper else self.live_url
        self.on_message = on_message
        self.on_open = on_open
        self.on_close = on_close
        self.on_ping = on_ping
        self.on_pong = on_pong
        self.on_error = on_error
        self.timeout = timeout
    


class BaseClient():

    def get_historical_data():
        NotImplemented
    def update_account():
        NotImplemented

    def get_balance(self,symbol):
        NotImplemented

    def update_account():
        NotImplemented
    def account():
        NotImplemented

    def get_trade_rules():
        NotImplemented
         
class AlpacaClient():
    def __init__(self,api_key,api_secret,time_frame,symbol,paper=True) -> None:
        self.trade_client=TradingClient(api_key, api_secret,paper=paper)
        self.data_client=CryptoHistoricalDataClient()
        self.time_frame= alpaca_time_map[time_frame]
        self.base_asset=symbol.split('/')[0]
        self.quote_asset=symbol.split('/')[1]
        self.symbol=symbol
        self._positions=None
        self._account=None
        self.update_positions()

    def get_historical_data(self,start_date):
        request_params = CryptoBarsRequest(
                        symbol_or_symbols=self.symbol,
                        timeframe=self.time_frame,
                        start=start_date
                 )
        
        bars = self.data_client.get_crypto_bars(request_params)
        data=bars.df.reset_index()
        data=data.rename(columns={'timestamp':'date_close'})
        return data
    
    def klines(self,symbol,time_frame,limit):
        delta=pd.Timedelta(time_frame)*limit
        start_date=(datetime.datetime.now()-delta)
        data=self.get_historical_data(start_date=start_date)
        return data
    
    def get_balance(self,symbol):

        
        if symbol.lower() in ['usd','usdt']:
            bal=self._account.get('cash')

        elif symbol==self.base_asset:
            bal=self._account.get(self.base_asset+self.quote_asset)
        return float(bal)

    def update_account(self):
        self.update_positions()
        account_obj=self.trade_client.get_account()
        account=account_obj.dict()
        pos_frame=self._positions
        asset_quanities=pos_frame['qty_available'].to_dict()
        account.update(asset_quanities)
        self._account=account
        
    
    def account(self):
        self.update_account()
        return self._account
    
    def update_positions(self):
        positions = self.trade_client.get_all_positions()
        position_list=[p.dict() for p in positions]
        pos_frame=pd.DataFrame.from_dict(position_list).set_index('symbol')
        
        self._positions=pos_frame
    
    def get_trade_rules(self):
        trade_info=self.trade_client.get_asset(self.symbol)
        trade_info=trade_info.dict()
        trade_rules=dict(
                        min_quote_size=1,
                        max_quote_size=1_000_000,
                        min_asset_size=trade_info['min_order_size'],
                        max_asset_size=1_000_000,
                        base_asset_precision=abs(round(Decimal(trade_info['min_trade_increment']).log10())),
                        quote_asset_precision=abs(round(Decimal(trade_info['price_increment']).log10())),
                        )
        return trade_rules
    
    def ticker_price(self,symbol):
        start_date=datetime.datetime.now()-pd.Timedelta(minutes=4).to_pytimedelta()
        request_params = CryptoBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame.Minute,
                        start=start_date
                 )
        
        bars = self.data_client.get_crypto_bars(request_params)
        data=bars.df.reset_index()

        price=data['close'].values[-1]
        return price
    
    def new_order(self,**kwargs):
        # preparing market order
        kwargs=self.check_params(**kwargs)
        
        market_order_data = MarketOrderRequest(
                            **kwargs
                            )
        
        market_order = self.trade_client.submit_order(
                order_data=market_order_data
               )
        return market_order_data
    
    def check_params(self,**kwargs):
        if 'quoteOrderQty' in kwargs:
            kwargs['notional']=kwargs.pop('quoteOrderQty')

        if 'quantity' in kwargs:
            kwargs['qty']=kwargs.pop('quantity')
        kwargs['side']=kwargs['side'].lower()
        kwargs['time_in_force']='ioc'
        return kwargs

    def new_listen_key(self):
        key={'listenKey':None}
        return key