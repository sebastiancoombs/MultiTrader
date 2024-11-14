
import datetime
import logging
import threading
from decimal import Decimal
from typing import Dict, Optional
import re
import numpy as np
import pandas as pd
from IPython.display import display
from tqdm.autonotebook import tqdm
try:
    from oandapyV20 import API
    from oandapyV20.contrib.factories import InstrumentsCandlesFactory
    from oandapyV20.contrib.requests import MarketOrderRequest as OandaMarketOrder
    from oandapyV20.endpoints import accounts, orders, pricing
except:
    print('need to install oandapyV20')


try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.live.crypto import CryptoDataStream
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest
except:
    print('need to install alpaca')
try:
    from coinbase.rest import RESTClient

except:
    print('need to install coinbase-advanced-py')



from .mappings import alpaca_time_map, oanda_time_map,coinbase_time_map
import math

class BaseClient():
    def __init__(self,

                 time_frame=None,
                 symbol=None,
                 paper=True,
                 positions=[0,1],
                 product_type='SPOT',
                 **kwargs) -> None:

        self.time_frame= time_frame
        self.paper=paper
        self.product_type=product_type.upper()
        self.connect(**kwargs)
        self.base_asset=None
        self.quote_asset=None
        self.symbol=None
        self.symbol_list=[]
        self.exchange=None

        self.set_base_quote_assets(symbol)

        self._positions=None
        self._account=None
        self.trade_rules=None

    def klines(self,symbol=None,time_frame=None,limit=100,start_date=None):
        delta=pd.Timedelta(time_frame)*(limit*2)
        if not symbol:
            symbol=self.symbol
        if not start_date:
            start_date=(pd.Timestamp.now(tz='utc')-delta)
        # print(symbol,time_frame,limit,start_date)
        #
        data=self.get_historical_data(symbol,start_date=start_date,end_date=datetime.datetime.now())
        return data

    def account(self):
        self.update_account()
        return self._account
  
    def set_base_quote_assets(self,symbol):
        symbol_split=re.split(r'/|_|-', symbol)
        self.symbol_list=symbol_split
        self.quote_asset=symbol_split[-1]
        self.base_asset=symbol_split[0]

        self.symbol=''.join([self.base_asset,self.quote_asset])

    def make_random_id(self):
        id=''.join([str(np.random.randint(0,9)) for i in range (7)])
        return id
    
    def round_down(self,num, precision):
        multiplier = pow(10,precision)
        return math.floor(num * multiplier) / multiplier

    def normalize_asset_size(self,size):
        balance=self.get_balance(self.base_asset)
        if self.trade_rules==None:
            self.trade_rules=self.get_trade_rules()        
        size=float(size)
        size=max([self.trade_rules['min_asset_size'],size]) # max of smallest possible trades
        size=min([self.trade_rules['max_asset_size'],size,balance])# min of largest possible trades eg. if we dont have enough currency to make a big trade sell everything
        size=self.round_down(size,self.trade_rules['base_asset_precision'])
        return size
    
    def normalize_quote_size(self,size):
        balance=self.get_balance(self.quote_asset)
        if self.trade_rules==None:
            self.trade_rules=self.get_trade_rules()

        size=float(size)
        size=max([self.trade_rules['min_quote_size'],size]) # max of smallest possible trades
        size=min([self.trade_rules['max_quote_size'],size,balance]) # min of largest possible trades eg. if we dont have enough currency to make a big trade sell everything 
        size=round(size,self.trade_rules['quote_asset_precision'])
        size=self.round_down(size,self.trade_rules['quote_asset_precision'])

        return size
    
    def get_precision(self,decimal):
        precision=abs(round(Decimal(decimal).log10()))
        return precision
    
    def get_spot_position(self) -> float:
        ## spot positions can only be between 0 and 1
        self.update_account()
        asset_bal=self.get_balance(self.base_asset)
        quote_bal=self.get_balance(self.quote_asset)

        asset_value=self.convert_to_quote(asset_bal)
        portfolio_value=self.get_spot_value()

        asset_ratio=asset_value/portfolio_value
        return asset_ratio
    
    def convert_to_base(self,quote_size):
        price=self.get_price(self.symbol)
        base_size=quote_size/price
        return base_size
    
    def convert_to_quote(self,base_size):
        price=self.get_price(self.symbol)
        quote_size=base_size*price
        return quote_size
    
    def get_balance(self,symbol):
        self.update_account()
        try:
            bal=self._account.loc[symbol,'balance']
        except:
            bal=0.0
        return float(bal)
    
    def get_spot_value(self,asset_bal=None,quote_bal=None):
        asset_bal=self.get_balance(self.base_asset)
        quote_bal=self.get_balance(self.quote_asset)
        asset_value=self.convert_to_quote(asset_bal)
        port_value=quote_bal+asset_value

        return port_value
    
    def get_portfolio_value(self):
        self.update_account()
        if self.product_type=='SPOT':
            port_value=self.get_spot_value()
        else:
            port_value=self.get_futures_value()
        return port_value
    
    def get_current_position(self):
        self.update_account()
        if self.product_type=='SPOT':
            position=self.get_spot_position()
        else:
            position,qty=self.get_futures_position()
        return position  

    def get_date_range(self,start_date,end_date,freq='1h'):
        end_date=min([end_date,pd.Timestamp.now(tz='utc')])
        date_range=pd.date_range(start=start_date,end=end_date,freq=freq,tz='UTC')
        return date_range.to_pydatetime()

    def get_start_dts(self,date_range,limit):
        
        start_ids=[i for i in range (0,len(date_range),limit) ]
        start_dts=date_range[start_ids]
        start_dts=[dt for dt in start_dts]

        return start_dts

    def bulk_download(self,symbol,start_date,end_date,limit,symbol_name=None,verbose=False):
        date_range=self.get_date_range(start_date,end_date)
        start_dts=self.get_start_dts(date_range,limit)
        data_list=[]
        for dt in tqdm(start_dts):
            data=self.get_historical_data(symbol=symbol,start_date=dt,verbose=verbose)
            data_list.append(data)
        data=pd.concat(data_list)
        if symbol_name is not None:
            data['symbol_name']=symbol_name
        data=data.drop_duplicates()
        return data
    
    def download_data(self,product_ids,start_date,end_date,data_dir='data'):
        for idx in tqdm(product_ids,leave=True):
            symbol_base=idx.split('-')[0]
            symbol_name=symbol_base
            
            data=self.bulk_download(idx,start_date,end_date,limit=300,symbol_name=symbol_name)
            data.index=pd.to_datetime(data['date_close'])
            data.to_pickle(f'data/{self.exchange}-{symbol_name}-{self.time_frame}.pkl')
    ## to add a new client, you need to implement the following methods
    def connect(self,**kwargs):
        NotImplemented

    def get_trade_rules(self)->Dict:
        """returns a dictionary with the following keys['min_quote_size','max_quote_size','min_asset_size','max_asset_size','base_asset_precision','quote_asset_precision']
        used for submiting tades later
        """
        NotImplemented

    def get_futures_value(self):
        NotImplemented

    def get_futures_position(self):
        NotImplemented
    
    def check_params(self,**kwargs):
        '''This method should check the parameters of the order and return the correct parameters for the order of whatever echange youre using'''
        return kwargs  
    
    def close_positions(self):
        NotImplemented

    def get_historical_data():
        NotImplemented
    
    def get_positions(self,symbol):
        NotImplemented
    
    def update_account(self)->pd.DataFrame:
        '''This must return a data frame with the index as the currency and the columns as the balance of the currency''' 
        NotImplemented

    def update_positions(self):
        NotImplemented

    def get_price(self,symbol)->float:
        NotImplemented

    def buy(self,symbol,qty):
        NotImplemented

    def sell(self,symbol,qty):
        NotImplemented

    def close(self,symbol):
        NotImplemented

    def close_all(self):
        NotImplemented
    
    def new_order(self,**kwargs):
        NotImplemented
        
    def new_listen_key(self):
        key={'listenKey':None}
        return key


class CoinbaseClient(BaseClient):
    
    def __init__(self,*args,**kwargs) -> None:
        
        self.paper_url='api-sandbox.coinbase.com'
        super().__init__(*args,**kwargs)

        self.trade_rules=self.get_trade_rules()
        self.exchange='coinbase'
        self.portfolio_id=self.get_portfolio_id()

    def set_base_quote_assets(self,symbol=None):
        if symbol==None:
            symbol=self.symbol
        if self.product_type=='SPOT':
            print(f' got symbol {symbol}')
            super().set_base_quote_assets(symbol)
            self.symbol='-'.join([self.base_asset,self.quote_asset])
            print(f'final symbol {self.symbol}')

        else:
            symbol_split=re.split(r'/|_|-', symbol)
            self.symbol_list=symbol_split
            print(symbol_split)
            self.base_asset='-'.join(symbol_split)
            self.quote_asset='USD'
            self.symbol='-'.join(symbol_split)

    def connect(self,api_key,api_secret,**kwargs):
        
        if self.paper:
            self.trade_client=RESTClient(api_key,api_secret,base_url=self.paper_url)
        else:
            self.trade_client=RESTClient(api_key,api_secret)
        print(f'Paper Trading{self.paper} on coinbase at {self.trade_client.base_url}')
   
    def get_futures_position(self,symbol):
        if symbol==None:
            symbol=self.symbol
        positions=self.trade_client.list_futures_positions()
        positions=pd.DataFrame(positions['positions'])
        positions=positions.set_index('product_id')
        if len(positions)==0:
            return 0
        else:
            side=positions.loc[symbol,'side']
            qty=positions.loc[symbol,'number_of_contracts']
            if side in ['LONG','BUY']:
                return 1,qty
            elif side in ['SHORT','SELL']:
                return -1,qty
                
    def get_futures_value(self):
        summary=self.trade_client.get_futures_balance_summary() 
        summary=summary['balance_summary']
        summary.pop('intraday_margin_window_measure')
        summary.pop('overnight_margin_window_measure')
        futures_account=pd.DataFrame(summary).T
        futures_account.rename(columns={'value':'balance'},inplace=True)
        futures_account['available']
        quote_bal=self._account.loc['available_margin','balance']
        asset_value=self._account.loc['unrealized_pnl','balance']
        port_value=quote_bal+asset_value
        return port_value 
        
    def update_account(self):
        accounts=self.trade_client.get_accounts()
        account_list=self.convert_to_dict_list(accounts['accounts'])
        self._account=pd.DataFrame(account_list)
        balances=self._account['available_balance'].apply(lambda x: pd.Series(x))
        positions=self._account['hold'].apply(lambda x: pd.Series(x))
        self._account['position_value']=positions['value']
        self._account=self._account.drop('available_balance',axis=1)
        self._account['balance']=balances['value']
        self._account=self._account.drop(self._account.filter(like='_at').columns,axis=1)
        self._account=self._account.set_index('currency')
        self._account['balance']=self._account['balance'].astype(float)
        self._account=self._account[self._account['balance']>0]

    def get_historical_data(self,symbol=None,start_date=None,end_date=None,verbose=False)->pd.DataFrame:
        if start_date==None:
            start_date=datetime.datetime.now()-pd.Timedelta(hours=350)
        if end_date==None:
            end_date=start_date+pd.Timedelta(hours=350)

        if symbol==None:
            symbol=self.symbol


        start_ts=int(start_date.timestamp())
        end_ts=int(end_date.timestamp())
        download_info={'symbol':symbol,
                        'start_date':start_date.strftime('%Y-%m-%d'),
                        'end_date':end_date.strftime('%Y-%m-%d'),
                        'start_ts':start_ts,
                        'end_ts':end_ts
                        }    

        try:
            candles=self.trade_client.get_candles(symbol,
                                                    start=start_ts,
                                                    end=end_ts,
                                                    granularity=coinbase_time_map[self.time_frame],
                                                    )
            candle_list=self.convert_to_dict_list(candles['candles'])
            klines=pd.DataFrame(candle_list)            
            klines['start']=pd.to_datetime(klines['start'],unit='s')
            klines=klines.rename(columns={'start':'date_close'})
            klines['symbol']=symbol

            if verbose:
                display(klines)
            
        except Exception as e:
            print('FAILED TO DOWNLOAD DATA')
            print(download_info)
            print(e)
            klines=pd.DataFrame()

        return klines
    
    def get_orders(self):
        fills=self.trade_client.get_fills()
        fill_list=self.convert_to_dict_list(fills['fills'])

        filled_orders=pd.DataFrame(fill_list)
        return filled_orders
    
    def get_products(self,product_type=None)->pd.DataFrame:
        if not product_type:
            product_type=self.product_type
        else:
            product_type=product_type.upper()
        products=self.trade_client.get_products(product_type=product_type)
        product_list=self.convert_to_dict_list(products['products'])
        # print(products)
        # print(products['products'])
        # n_products=len(products['products'])
        # print('type',type(products['products'][0]))
        product_frame=pd.DataFrame.from_records(product_list)
        # print(f'got {n_products} products returned')

        try:
            details=product_frame['future_product_details'].apply(pd.Series)
            product_frame[details.columns]=details

        except:
            pass
        only_cols=product_frame.filter(like='_only').columns.to_list()
        only_cols=[col for col in only_cols if col != 'view_only']
        drop_cols=['volume_percentage_change_24h',
                   'volume_24h',
                   'price_percentage_change_24h',
                   'future_product_details',
                   'watched',
                   'is_disabled',
                   'new',
                   'contact_expiry',
                   'twenty_four_seven',
                    'approximate_quote_24h_volume','venue',
                    'risk_managed_by',
                    'auction_mode',
                    'trading_disabled',
                   'non_crypto']+only_cols
        
        drop_cols=[col for col in drop_cols if col in product_frame.columns]
        drop_cols=list(set(drop_cols))
        product_frame=product_frame.drop(drop_cols,axis=1)
        # product_frame=product_frame[product_frame['view_only']==False]

        return product_frame
    
    def convert_to_dict_list(self,obj_list):
        return [p.__dict__ for p in obj_list]

    def get_trade_rules(self,symbol=None):
        if symbol==None:
            symbol=self.symbol
        product_frame=self.get_products()
        
        try:
            product_frame.set_index('product_id',inplace=True)
            trade_info=product_frame.loc[symbol]
            trade_rules=dict(
                            min_quote_size=float(trade_info['quote_min_size']),
                            max_quote_size=float(trade_info['quote_max_size']),
                            min_asset_size=float(trade_info['base_min_size']),
                            max_asset_size=float(trade_info['base_max_size']),
                            base_asset_precision=self.get_precision(trade_info['base_increment']),
                            quote_asset_precision=self.get_precision(trade_info['quote_increment']),
                            )
        except Exception as e:
            display(product_frame)
            print('Failed to set trade rules, target symbol is probabaly wrong:',e)

            trade_rules=None
        return trade_rules

    def get_price(self, symbol):

        product=self.trade_client.get_product(product_id=self.symbol)
        price=float(product['price'])
        return price
    
    def get_portfolio_id(self):

        ports=self.trade_client.get_portfolios()
        id=ports['portfolios'][0]['uuid']
        return id
    
    def buy(self,symbol=None,quote_size=1,order_id=None):
        if symbol==None:
            symbol=self.symbol
        order_id=self.make_random_id() if order_id==None else order_id
        order_id=str(order_id)
        order_args={
                    'client_order_id':order_id,
                    'product_id':symbol,
                    # 'rfq_enabled':True,
                    'retail_portfolio_id':self.portfolio_id 
                    }
        quote_size=self.normalize_quote_size(quote_size)
        quote_size=str(quote_size)
        if self.product_type=='SPOT':
            order_args['quote_size']=quote_size
        if self.product_type=='FUTURES':
            order_args['base_size']=quote_size

        order = self.trade_client.market_order_buy(**order_args)
        print(order)
        print('ok...returning now....')

        return self.convert_order_to_info(order)

    def sell(self,symbol=None,base_size=1,order_id=None):
        if symbol==None:
            symbol=self.symbol

        base_size=self.normalize_asset_size(base_size)
        base_size=str(base_size)
        order_id=self.make_random_id() if order_id==None else order_id
        order_id=str(order_id)
        order = self.trade_client.market_order_sell(
            client_order_id=order_id,
            product_id=symbol,
            base_size=base_size,
            retail_portfolio_id=self.portfolio_id, 

        )
        print(order)
        print('ok...returning now....')

        return self.convert_order_to_info(order)
    
    def convert_order_to_info(self,order):
        order=order.__dict__
        response=order.pop('response')
        order.update(response)

        fill_config=order.pop('order_configuration').__dict__

        fill_type=[c for c in fill_config.keys()][0]
        fill_details=fill_config.pop(fill_type).__dict__
        order['fill_type']=fill_type
        order.update(fill_details)
        print('fixed order:',order)
        return order

    def close(self,symbol=None):
        ## only use for Futures positions
        if symbol==None:
            symbol=self.symbol
        position,qty=self.get_futures_position(symbol)
        if position==0:
            order={'order':'no position to close'}
        
        elif position==1:
            order=self.sell(symbol,qty=qty)

        elif position==-1:
            order=self.buy(symbol,qty=qty)
        return order

class BinanceClient(BaseClient):
    
    def __init__(self,api_key,api_secret,time_frame,symbol,paper=True,product_type='SPOT',*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.paper_url=''

class OandaClient(BaseClient):

    def __init__(self,api_key,account_id,time_frame,symbol,paper=True,*args,**kwargs) -> None:

        
        super().__init__(*args,**kwargs)
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
        self.update_account(self)

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
        # self.update_account()

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