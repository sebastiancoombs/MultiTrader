
import datetime
import logging
import threading
from decimal import Decimal
from typing import Dict, Optional

import numpy as np
import oandapyV20 as oanda
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
import pandas as pd
from oandapyV20.contrib.requests import MarketOrderRequest as OandaMarketOrder

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.live.crypto import CryptoDataStream
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest
from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory

from .mappings import alpaca_time_map,oanda_time_map


class BaseClient():
    def __init__(self,
                 api_key=None,
                 api_secret=None,
                 time_frame=None,
                 symbol=None,paper=True) -> None:

        self.time_frame= time_frame
        self.base_asset=None
        self.quote_asset=None
        self.symbol=symbol
        self._positions=None
        self._account=None


    def get_historical_data():
        NotImplemented

    def klines(self,symbol,time_frame,limit):
        delta=pd.Timedelta(time_frame)*(limit**2)
        start_date=(datetime.datetime.now()-delta)
        data=self.get_historical_data(start_date=start_date)
        return data

    def get_balance(self,symbol)->float:

        if symbol.lower() in ['usd','usdt']:
            bal=self._account.get('cash')

        elif symbol==self.base_asset:
            bal=self._account.get(self.base_asset+self.quote_asset)
        return float(bal)
    
    def update_account(self)->Dict:
        NotImplemented

    def update_positions(self):
        NotImplemented

    def account(self)->Dict:
        
        self.update_account()
        return self._account
    
    def check_params(self,**kwargs):
        
        return kwargs  
    
    def new_order(self,**kwargs):
        NotImplemented

    def get_trade_rules(self):
        NotImplemented
         
    def ticker_price(self,symbol)->float:
        NotImplemented

    def new_listen_key(self):
        key={'listenKey':None}
        return key


class OandaClient(BaseClient):
    def __init__(self,api_key,account_id,time_frame,symbol,paper=True) -> None:
        super().__init__()
        self.api=API(access_token=api_key,environment='practice' if paper else 'live',)
        self.account_id=account_id
        self.data_client=None
        self.oanda_time_frame= oanda_time_map[time_frame]
        self.time_frame= time_frame
        self.base_asset=symbol.split('_')[0]
        self.quote_asset=symbol.split('_')[1]
        self.symbol=symbol
        self._positions=None
        self._account=None
        self.update_positions()

    def get_historical_data(self,start_date)->pd.DataFrame:


        now=datetime.datetime.now()

        start_date=start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        now=now.strftime('%Y-%m-%dT%H:%M:%SZ')

        params={
            "granularity": oanda_time_map[self.time_frame],
            "from": start_date,
            "to": now,
        }

        klines=InstrumentsCandlesFactory(instrument=self.symbol, params=params)
        candles=[]
        for batch in klines:
            resp = self.api.request(batch)
            
            for candle in resp.get('candles'):

                ctime = candle.get('time')[0:19]
                
                try:
                    row = dict(
                        date_close=ctime,
                        open=candle['mid']['o'],
                        high=candle['mid']['h'],
                        low=candle['mid']['l'],
                        close=candle['mid']['c'],
                        volume=candle['volume'],
                    )
                
                    candles.append(row)
                except Exception as e:

                    print(e, klines)
                    
        return pd.DataFrame(candles)
    
    def get_balance(self,symbol)->float:

        if symbol.lower() in ['usd','usdt']:
            bal=self._account.get('cash')

        # elif symbol==self.base_asset:
        else:
            bal=self._account.get(self.base_asset+self.quote_asset)
        


        return float(bal)
    
    def update_account(self):
        self.update_positions()
        req=accounts.AccountDetails(self.account_id)
        account=self.api.request(req)['account']
        account.pop('positions')
        account.pop('trades')

        account[self.base_asset+self.quote_asset]=account['positionValue']
        account['cash']=account['marginAvailable']
        self._account=account

    def update_positions(self):
        req=accounts.AccountDetails(self.account_id)
        resp=self.api.request(req)['account']
        positions = resp.get('positions')
        
        position_list=[self.format_position(p) for p in positions]
        positions=[]
        for pos in position_list:
            for p in pos:
                positions.append(p)
        pos_frame=pd.DataFrame.from_dict(positions)
        if len (pos_frame)>0:
            pos_frame.set_index('symbol')
            pos_frame=pos_frame[sorted(pos_frame.columns)]
        self._positions=pos_frame

    def format_position(self,position)->Dict:
        position_list=[]
        long=position.pop('long')
        long['side']='buy'
        short=position.pop('short')
        short['side']='sell'
        symbol=position.pop('instrument')
        long['symbol']=symbol
        short['symbol']=symbol

        if int(long['units'])!=0:
            position_list.append(long)

        if int(short['units'])!=0:
            position_list.append(short)

        return position_list
    
    def account(self)->Dict:
        self.update_account()
        return self._account
    
    def check_params(self,**kwargs):

        symbol=kwargs.get('symbol')

        qty=kwargs.get('notional')
        side=kwargs.get('side')
        
        if side.lower()=='sell':
            qty=-qty
        
        mktOrder=OandaMarketOrder(
                        instrument=symbol,
                        units=qty,
                        
                        )
        
        order_params = orders.OrderCreate(self.account_id, data=mktOrder.data)

        return order_params
    
    def new_order(self,**kwargs):
        order_params=self.check_params(**kwargs)

        try:
            # create the OrderCreate request
            resp = self.api.request(order_params)
        except oanda.exceptions.V20Error as err:
            print(order_params.status_code, err)
    
    def get_symbol_info(self):

        params={"instruments":self.symbol}

        req=pricing.PricingInfo(self.account_id, params=params)
        trade_info=self.api.request(req)['prices'][0]
        return trade_info
    
    def get_trade_rules(self):
        trade_info=self.get_symbol_info()['quoteHomeConversionFactors']
        
        trade_rules=dict(
                        min_quote_size=float(trade_info['negativeUnits']),
                        max_quote_size=1_000_000,
                        min_asset_size=float(trade_info['positiveUnits']),
                        max_asset_size=1_000_000,
                        base_asset_precision=abs(round(Decimal(trade_info['positiveUnits']).log10())),
                        quote_asset_precision=abs(round(Decimal(trade_info['negativeUnits']).log10())),
                        )
        return trade_rules
         
    def ticker_price(self,symbol):
        params={"instruments":symbol}

        req=pricing.PricingInfo(self.account_id, params=params)
        resp=self.api.request(req)
        bid=float(resp['prices'][0]['closeoutBid'])
        ask=float(resp['prices'][0]['closeoutAsk'])
        return float(np.mean([bid,ask]))




class AlpacaClient():
    def __init__(self,api_key,api_secret,time_frame,symbol,paper=True) -> None:
        self.trade_client=TradingClient(api_key, api_secret,paper=paper)
        self.data_client=CryptoHistoricalDataClient()
        self.time_frame= alpaca_time_map[time_frame]
        self.base_asset=symbol.split('/')[0]
        self.quote_asset=symbol.split('/')[1]
        if self.quote_asset.lower()=='usdt':
            self.quote_asset=self.quote_asset.replace('T','')
        self.symbol=symbol
        self._positions=None
        self._account=None
        self.update_account()

    def get_historical_data(self,start_date)->pd.DataFrame:
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
        delta=pd.Timedelta(time_frame)*limit**2
        start_date=(datetime.datetime.now()-delta)
        data=self.get_historical_data(start_date=start_date)
        return data
    
    def get_balance(self,symbol)->float:

        if symbol.lower() in ['usd','usdt']:
            bal=self._account.get('cash')

        elif symbol==self.base_asset:
            
            bal=self._account.get(self.base_asset+self.quote_asset)
            if not bal:
                bal=0
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