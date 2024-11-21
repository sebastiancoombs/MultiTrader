import datetime
import json
import sqlite3 as db
import time
import re

import numpy as np
import pandas as pd
# from binance.spot import Spot
from gym_trading_env.utils.history import History
from gym_trading_env.utils.portfolio import Portfolio
from tqdm.asyncio import tqdm
from utils.utils import preprocess_data,prepare_forecast_data,plot_insample_forecasts
from utils.mappings import binanace_col_map, symbol_map,alpaca_stream_col_map,alpaca_stream_message_map

from utils import clients 
from .environments import NormTradingEnvironment
from IPython.display import display
# from oandapyV20.endpoints import pricing 
from utils.discord_utils import send_message_to_channel,send_picture_to_channel
import json
import traceback
import matplotlib.pyplot as plt

class BaseLiveTradingEnv(NormTradingEnvironment):


    def __init__(self, 
            api_key,
            api_secret=None,
            account_id=None,
            paper=True,
            time_frame='1h',
            symbol='ETH/USDT',
            restore_trading=False,
            streaming=False,
            history_path='Trade_history/trade.db',
            exchange='coinbase',
            forecast_model=None,
            product_type='Spot',
            discord_webhook=None,
            *args, **kwargs):
        
        ## account info
        self.api_key=api_key
        self.api_secret=api_secret
        self.account_id=account_id
        self.exchange=exchange.lower()
        self.product_type=product_type.upper()
        self.discord_webhook=discord_webhook
        self.allow_trade_submit=True
        
        self.supported_exchanges=['coinbase',
                                #   'binance','alpaca','oanda'
                                  ]
        self.pred_df=None
        self.raw_df=None
        ## history path for saving information... possibly not neccesary
        self._history_path=history_path
        ## load for do not load a previous trade history
        self._restore_trading=restore_trading

        ## model_precition_info
        self.forecast_model=forecast_model
        self.context_length=max([model.hparams['input_size'] for model in self.forecast_model.models])

        kwargs['max_episode_duration']='max'

        ### trading info
        self.loading_bar=None
        self.action_map={x:pct for x,pct in enumerate(kwargs['positions'])}
        self.side_map={-1:'SELL',1:'BUY',0:'close'}
        self.position_map={x:y for y,x in self.side_map.items()}

        self.paper=paper
        
        ## symbol information
        self.quote_asset=None
        self.base_asset=None
        self.symbol=None
        self.client=None

        self.time_frame=time_frame
        client_args=dict(api_key=api_key,api_secret=api_secret,account_id=account_id,product_type=product_type,symbol=symbol,paper=paper,time_frame=time_frame)
        self.connect_client(**client_args)
        self.set_base_quote_assets()

        self._streaming=streaming

        ### information about reward suff... probably not neccesary only needed for training
        self._rew=0
        self._current_valuation=0
        self.listen_key=None
        self.stream_client =None

        df=self.get_data()
        kwargs['df']=df
        super().__init__(*args,**kwargs)

    def set_base_quote_assets(self):
        self.quote_asset=self.client.quote_asset
        self.base_asset=self.client.base_asset
        self.symbol=self.client.symbol

    def connect_client(self ,**kwargs):
        assert self.exchange in self.supported_exchanges, f'{self.exchange} not supported'
        if self.exchange=='coinbase':
            self.client=clients.CoinbaseClient(**kwargs)
        elif self.exchange=='binance':
            self.client=clients.BinanceClient(**kwargs)
        elif self.exchange=='alpaca':
            self.client=clients.AlpacaClient(**kwargs)
        elif self.exchange=='oanda':
            self.client=clients.OandaClient(**kwargs)

        print(f'Connected to {self.exchange} client')
        #### add any other clients here

    def connect_to_db(self):
        conn = db.connect(self._history_path)
        return conn

    def get_info(self,old_info):
            
        new_info={
            'date':pd.Timestamp.now(tz='US/Pacific').strftime('%m-%d-%Y %I:%M %p %Z'),
            'base_asset':self.base_asset,
            'quote_asset':self.quote_asset,
            'data_symbol':self.symbol,
            'position': self.client.get_current_position(),
            'time_frame':self.time_frame,
            'exchange':self.exchange,
            'product_type':self.product_type,
            'portfolio_distribution_asset':self.client.get_balance(self.base_asset),
            'portfolio_distribution_fiat':self.client.get_balance(self.quote_asset),
            'portfolio_valuation':self.client.get_portfolio_value(),
            
        }
        if old_info==None:
            info=new_info
        else:
            old_info.update(new_info)
            info=old_info

        info['idx']=self._idx
        info['real_position']=info['position']
        return info

    def get_data(self):
        print('Getting data')
        data=self.client.klines(symbol=self.symbol,time_frame= self.time_frame)
        self.raw_df=data.copy()
        data=preprocess_data(data)
        data,pred_df=prepare_forecast_data(model=self.forecast_model,
                                           data=data,
                                           time_frame=self.time_frame,
                                           real=True,
                                           symbol=self.base_asset)
        price=self.client.get_price(self.symbol)
        data['close']=price
        
        
        self.pred_df=pred_df
        
        return data
    
    def prepare_plot_df(self):
        self.raw_df['ds']=self.raw_df['date_close'].copy()
        plot_df=pd.concat([self.raw_df[['ds','close']],self.pred_df],axis=0).reset_index(drop=True)
        plot_df['symbol']=plot_df['symbol'].fillna(method='ffill').fillna(method='bfill')
        plot_df=plot_df[-60:]
        plot_df['close']=pd.to_numeric(plot_df['close'])
        return plot_df 
    
    def plot_forecasts(self,plot_df,symb):
        fig,axes=plt.subplots(sharex=True, sharey=True, figsize=(10, 5))
        time=pd.Timestamp(plot_df['ds'].values[-1]).strftime('%m-%d-%Y %I:%M%p')

        axes.plot(plot_df['ds'], plot_df['close'], label='Close Price')
        for model in plot_df.select_dtypes(np.number).columns:
            if model=='close':
                continue
            axes.plot(plot_df['ds'], plot_df[model], label=f'{model} Forecast')
        
        axes.set_xlabel('Timestamp [t]')
        axes.set_ylabel(f'{symb} Price')
        axes.set_title(f'{symb} Forecast through {time}')
        axes.grid()
        fig.legend()
        return fig
    
    def _get_obs(self):
        data=self.get_data()
        self._set_df(data)
        obs=super()._get_obs()
        return obs

    def get_trade_size(self,change_ratio):
        price=self.client.get_price(self.symbol)
        porfolio_value=self.client.get_portfolio_value()
        size_in_dollars=porfolio_value*change_ratio
        size_in_asset=size_in_dollars/price

        return size_in_dollars,size_in_asset

    def close(self):
        if self.product_type.upper()=='SPOT':
            asset_val=self.client.get_balance(self.base_asset)
            self.client.sell(self.base_asset,base_size=asset_val)
            
        elif self.product_type.upper()=='FUTURES':
            self.client.close_position()
        else:
            print(f'{self.product_type} Not supported')

    def _trade(self, new_position):
        ## get current asset to dollar ratio closest to self.positions
        self._position=self.client.get_current_position()
        current_position_idx=int(np.argmin(np.abs(np.array(self.positions)-self._position)))
        current_position=self.positions[current_position_idx]

        ## get size of assets we have
        trade_from_to=[self._position,new_position]
        position_change=float(np.diff([trade_from_to]))
        ## decide which direction to go
        
        buy_sell=int(np.sign(position_change))
        change_ratio=float(np.abs(position_change))
        
        ## get the target potfolio distribution
        position_target_id=int(np.argmin(np.abs(np.array(self.positions)-position_change)))
        position_target=self.positions[position_target_id]

        dollar_val,n_asset=self.get_trade_size(change_ratio)
        
        info={
        'Current_position':current_position,
        'New_position':position_target,

        'Trade_from':trade_from_to[0],
        'Trade_to':trade_from_to[1],
        'Change_size':float(position_change),
        'Change_direction':buy_sell,
        'Place_order':self.side_map[buy_sell],
        'Change needed to get to target':change_ratio,
        'Size in dollars':dollar_val,
        'Size in asset':n_asset,
        }
        order_id=self.get_order_number()
        print(info)
        if info['Current_position'] == info['New_position']:
            print('No trade')
            info['success']=False
            info['error']='No Position Change so did not Submit Trade'
            return info
        
        elif buy_sell==-1:
            print('Sell')
            order_info=self.client.sell(base_size=n_asset,order_id=order_id)
            info.update(order_info)

        elif buy_sell==1:
            print('Buy')
            order_info=self.client.buy(quote_size=dollar_val,order_id=order_id)
            info.update(order_info)

        return info

    def live_step(self,position_index=None,wait=False):
        done, truncated=False,False
        trade_info={}
        if self.allow_trade_submit==True:
            if position_index is not None: 
                trade_info=self._trade(self.positions[position_index])

        live_info=self.get_info(trade_info)
        live_info.update(trade_info)

        reward = self.get_reward()
        obs=self._get_obs()
        if self. allow_trade_submit==True:
            self.save_history(live_info)
        return obs,  reward, done, truncated,live_info

    def step(self,action):
        obs,rew,done,truncated,info=self.live_step(action)
        return obs,rew,done,truncated,info

    def load_history(self):
        try:
            conn=self.connect_to_db()
            query=f"SELECT * FROM {self.base_asset}_trade_history"
            history=pd.read_sql(query,conn)
            conn.close()
        except:
            history=pd.DataFrame()
        return history  
    
    def save_history(self,info):
        conn=self.connect_to_db()
        if self.discord_webhook!=None:
            message=json.dumps(info, indent=2)
            try:

                send_message_to_channel(self.discord_webhook,message)
                plot_df=self.prepare_plot_df()
                display(plot_df)
                fig=self.plot_forecasts(plot_df=plot_df,symb=self.base_asset)
                fig_time=pd.Timestamp.now().strftime('%m-%d-%Y %I:%M%p')
                fig_name=f'{fig_time}_forecasts.png'
                fig.savefig(fig_name)
                send_picture_to_channel(self.discord_webhook,file=fig_name)
                
            except Exception as e:
                traceback.print_exc()

        history=pd.DataFrame([info])

        try:
            order=history['response'].apply(lambda x: pd.Series(x))
            history[order.columns]=order
        except:
            pass
        display(history)
        try:
            history.to_sql(f'{self.base_asset}_trade_history',conn,if_exists='append',index=False)
        except Exception as e:
            print(e)
            old_history=self.load_history()
            new_history=pd.concat([old_history,history])
            new_history.to_sql(f'{self.base_asset}_trade_history',conn,if_exists='replace',index=False)
        conn.close()
        return
    
    def get_reward(self):
        history=self.load_history()
        try:
            reward=self.reward_function(history)
        except Exception as e:
            print(e)
            reward=0

        return reward

    def get_order_number(self):
        history=self.client.get_orders()
        new_order_number=len(history)+1
        order_id=f'{self.base_asset}_order_{new_order_number}'
        return order_id

    def reset(self,**kwargs):

        obs,info=super().reset()
        self._idx=len(self.df)-1
        print(self._idx)
        obs=self._get_obs()
        info=self.get_info(info)
        
        return obs,info
        
    def wait_for_reward(self):
        
        current_time=pd.Timestamp(datetime.datetime.now())
        

        next_time=current_time+pd.Timedelta( self.time_frame)
        next_time=next_time.floor(self.time_frame)
        wait_time=(next_time-current_time).seconds

        valuation=self.client.get_portfolio_value()
        bar=tqdm(range(wait_time))
        reward=round(self.historical_info["reward", -1],2)
        position=self.client.get_current_position()
        valuation=self.client.get_portfolio_value()
        reward=1.00

        for i in bar:
            time.sleep(1)
            bar.set_description(f'position {self._position} value: {valuation:.2f},reward: {reward:.2f} waiting {int((wait_time-i)/60)} min for next reward')
            if i%60==0:
                
                valuation=self.client.get_portfolio_value()

        return
    
class CoinbaseTradingEnv(BaseLiveTradingEnv):
    def __init__(
            self,
            
            *args,
            **kwargs,
            ):
        super().__init__(*args,**kwargs)
    
class BinanceTradingEnv(BaseLiveTradingEnv):
    def __init__(
            self,
            
            *args,
            **kwargs,
            ):

        self._preprocess_data=preprocess_data
        client=self.connect_client()
        listen_key=client.new_listen_key()['listenKey']
        self.client=client
        self._listen_key=listen_key
        self.account=self.get_account()
        self.initial_base_balance=self.get_balance(self.base_asset)
        self.initial_quote_balance=self.get_balance(self.quote_asset)
        self.set_trade_rules()
        self.cash=self.initial_base_balance
        data=self.get_data(initial_pull=True)
        kwargs['df']=data
        kwargs['portfolio_initial_value']=self.initial_quote_balance

        super().__init__(*args,**kwargs)
        self.spec.id='LiveNeuralForecastingTradingEnv'
        self._fake_trade=super()._trade
        if self._restore_trading:
            if self._load_history():
                print('Restored trading history')
                

    def connect_client(self):
        if self.client!=None:
            del self.client

        if self.paper:

            client=Spot(api_key=self.api_key,api_secret=self.api_secret,base_url='https://testnet.binance.vision')

        else:
            client=Spot(api_key=self.api_key,api_secret=self.api_secret,base_url='https://api3.binance.com')
   
        return client
      
    def set_trade_rules(self):
        '''
        Set the min and max trade_sizes and the values to round by for order precision rounding
        '''
        trade_info=pd.DataFrame(self.client.exchange_info(self.symbol)['symbols'][0]['filters'])
        trade_info=trade_info.set_index('filterType')
        self._trade_info=trade_info
        
        self.min_quote_size=float(trade_info.loc['LOT_SIZE','minQty'])
        self.max_quote_size=float(trade_info.loc['LOT_SIZE','maxQty'])
        self.min_asset_size=float(trade_info.loc['MARKET_LOT_SIZE','minQty'])
        self.max_asset_size=float(trade_info.loc['MARKET_LOT_SIZE','maxQty'])
        
        self.quote_asset_precision=self.client.exchange_info(self.symbol)['symbols'][0]['quotePrecision']
        self.base_asset_precision=self.client.exchange_info(self.symbol)['symbols'][0]['baseAssetPrecision']

    def live_step(self,position_index,wait=True):

        if position_index is not None: self._take_action(self.positions[position_index])
        self._n_steps+=1
        self._step += 1
        self._idx = -1
        done, truncated = False, False
        if wait:
            self.wait_for_reward()
        
        portfolio_value = self._portfolio.valorisation(self._get_price())
        portfolio_distribution = self._portfolio.get_portfolio_distribution()

        if portfolio_value <= 1000:
            done = True

        if isinstance(self.max_episode_duration,int) and self._step >= self.max_episode_duration - 1:
            truncated = True

        reward = self.reward_function(self.historical_info)
        self.historical_info["reward", -1] = reward
        
        hist_config=self.build_history(reward=reward)
      
        self.historical_info.add(**hist_config)
        self._save_history()
                
        self.set_forecast_df()
        obs=self._get_obs()
        print(self.pred_df.tail(1))
        print(self.df.tail(1))
        # display(obs)

        return self._get_obs(),  self.historical_info["reward", -1], done, truncated, self.historical_info[-1]
    
    def set_forecast_df(self):
        self._set_df(self.get_data())
        self._prep_forecasts()
        self._set_df(self.get_data())
        

    
    def get_account(self):
        account=self.client.account()
        account_frame=pd.DataFrame(account['balances']).set_index('asset')
        return account_frame
    
    def get_balance(self,symbol):
        bal=self.account.loc[symbol]
        bal=bal['free']
        # print(bal)
        bal=float(bal)
        return bal
    

    def get_position(self,history):
        current_position=history['position',-1]
        return current_position


            
 

        data['ds']=data.index.copy()
        data['symbol'] = self.symbol
        data['unique_id']=symbol_map[self.symbol]
        data['is_closed']=True
        with self.connect_to_db() as conn:
            if initial_pull:
                data['bar_type']='Bar'
                data.to_sql(f'{self.symbol}_candle_history',conn,if_exists='replace',index=True)
            data=data.drop(['is_closed'],axis=1)


        return data
    
    def get_klines(self):

        data_list=self.client.klines(self.symbol, self.time_frame, limit=self.context_length*2)
        columns=['date_open','open','high','low','close','volume','date_close','QA_volume','N_trades','BA_volume','BQ_volume','unused']
        data=pd.DataFrame(data_list,columns=columns)

        data['date_close']=data['date_close'].apply(pd.to_datetime,unit='ms')

        data=data.set_index('date_close')
        data=data[['open','high','low','close','volume']]
        return data
    
    def get_stream_data(self):
         
        query=f"""SELECT * FROM f'{self.symbol}_candle_history' 
        ORDER BY ds ASC
        WHERE Is_Closed= true 
        LAST {self.context_length*2}"""
        conn=self.connect_to_db()
        
        data=pd.read_sql(query,conn)
        data=data.drop(['Is_Closed'],axis=1)

        data=data.set_index('date_close')
        data=data.resample('1h')
        

        data=self._preprocess_data(data)
        data['ds']=data.index.copy()

        data['symbol'] = self.symbol
        data['unique_id']=symbol_map[self.symbol]
        
        conn.close()
        return data
    
    def _stream_data_handler(self, message):
        try:
            # data=ast.literal_eval(message)

            # print(message)
            message_data=json.loads(message)
            k_data=message_data['k']


            # print(type(data),data)
            data=pd.DataFrame([k_data])
            
            data.columns=data.columns.map(binanace_col_map)
            data=data[[c for c in binanace_col_map.values() if c in data.columns]]

            data=data.drop('date_open',axis=1)
            data["date_close"]=pd.to_datetime(data["date_close"],unit='ms')
            data["ds"]=data["date_close"].copy()
            data=data.set_index('date_close')
            data['symbol'] = self.symbol
            data['unique_id']=symbol_map[self.symbol]

            conn=db.connect(self._history_path)
            data.to_sql(f'{self.symbol}_candle_history',conn,if_exists='append',index=True)
            return k_data['x']
        
        except Exception as e: 
            print('bad_data',message)
            print(e)
            return False
        
    def wait_for_reward(self):
        self.client=self.connect_client()
        current_time=pd.Timestamp(datetime.datetime.now())
        

        next_time=current_time+pd.Timedelta( self.time_frame)
        next_time=next_time.floor(self.time_frame)
        wait_time=(next_time-current_time).seconds
        self.update_portfolio()
        valuation=round(self._portfolio.valorisation(self._get_price()),2)
        bar=tqdm(range(wait_time))
        reward=round(self.historical_info["reward", -1],2)

        for i in bar:
            time.sleep(1)
            bar.set_description(f'position {self._position} value: {valuation:.2f},reward: {reward:.2f} waiting {int((wait_time-i)/60)} min for next reward')
            if i%60==0:
                self.update_portfolio()
                valuation=self._portfolio.valorisation(self._get_price())

        return

    def reset_account(self):
        ##update account
        self.account=self.get_account()
        price=self._get_price()
        ##Get current_valuation
        current_size=self.get_balance(self.base_asset)
        current_dols=self.get_balance(self.quote_asset)
        total_valuation=current_dols+current_size*price
        ##split in half to know how much to distribute to each side
        half_valuation=total_valuation/2
        split_asset=half_valuation/price
        
        dump_size=np.diff([split_asset,current_size])
        trade_size=abs(dump_size)
        dollar_val=trade_size*price
        dollar_val=self.normalize_quote_size(dollar_val)

        if dump_size>self.min_asset_size:
            
            params = {
                'symbol': self.symbol,
                'side':'SELL',
                'type': 'MARKET',
                'quoteOrderQty':dollar_val,
                # 'quantity':trade_size,
                }
            
            print('reset_account')
            print(params)
            self.client.new_order(**params)


        elif dump_size<self.min_asset_size:
            params = {
                'symbol': self.symbol,
                'side':'BUY',
                'type': 'MARKET',
                'quoteOrderQty':dollar_val,
                }
            
            print('reset_account')
            print(params)
            self.client.new_order(**params)
        else:
            print('no need to reset')
            print(f'{self.base_asset}: {current_size} {self.quote_asset}: {current_dols}')

    def close_positions(self):
        ##update account
        self.account=self.get_account()

        price=self._get_price()
        ##Get current_valuation
        current_size=self.get_balance(self.base_asset)
        current_dols=self.get_balance(self.quote_asset)
        total_valuation=current_size*price
        ##split in half to know how much to distribute to each side
        trade_size=self.normalize_asset_size(current_size)

        params = {
            'symbol': self.symbol,
            'side':'SELL',
            'type': 'MARKET',
            # 'quoteOrderQty':dollar_val,
            'quantity':trade_size,
            }
        
        print('Close out account')
        print(params)
        self.client.new_order(**params)

    def format_history(self, **kwargs):
        values = []
        columns = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                columns.extend([f"{name}_{i}" for i in range(len(value))])
                values.extend(value[:])
            elif isinstance(value, dict):
                columns.extend([f"{name}_{key}" for key in value.keys()])
                values.extend(list(value.values()))
            else:
                columns.append(name)
                values.append(value)

        if columns == self.historical_info.columns:
            hist_dict=dict(zip(columns,values))
        else:
            col_ids=[]
            updated_values=[]
            for col in self.historical_info. columns:
                if col in columns:
                    c_id=columns.index(col)
                    val=values[c_id]
                    updated_values.append(val)
                else:
                    updated_values.append(0)
            hist_dict=dict(zip(self.historical_info.columns,updated_values))
        return hist_dict

    def build_history(self,reward=None,initial_record=False):
        self.get_account()
        self._portfolio=Portfolio(
            asset = self.get_balance(self.base_asset),
            fiat = self.get_balance(self.quote_asset),
        )
        hist_config=dict(idx = self._idx,
                step = self._step,
                date = pd.Timestamp(datetime.datetime.now()) ,
                position_index = 0 if self._position ==0 else self.positions.index(round(self._position,2)),
                position = self._position,
                real_position = self._position,
                portfolio_valuation = self._portfolio.valorisation(self._get_price()),
                portfolio_distribution = self._portfolio.get_portfolio_distribution(),
                reward = reward if reward != None else self.reward_function(self.historical_info))
        data_dict=self.df.iloc[[self._idx]].to_dict('records')[0]
        hist_config.update(data_dict)
        if not initial_record:
            hist_config=self.format_history(**hist_config)
    
        return hist_config
    
    def _load_history(self,position=None):
        try:
            self.historical_info = History(max_size=1000)
            conn=self.connect_to_db()

            history_df=pd.read_sql(f'select * from {self.symbol}_trade_history',conn)
            history_df=history_df.tail(len(self.df))
            history=history_df.to_dict('records')
            for i,h in enumerate(history):
                col_list=sorted(h)
                h={c:h[c] for c in col_list}

                if i==0:
                    self.historical_info.set(**h)
                else:
                    self.historical_info.add(**h)
            
            self._position=position if position!=None else round(self.historical_info["position", -1],2)
            conn.close()
            return True
        except Exception as e:
            print(e)
            return False
        
    def _save_history(self):
        history_df=pd.DataFrame([self.historical_info[-1]])
        conn=self.connect_to_db()

        if self._restore_trading==False and self._step==0:
            
            history_df.to_sql(f'{self.symbol}_trade_history',conn,if_exists='replace',index=False)
        else:
            history_df.to_sql(f'{self.symbol}_trade_history',conn,if_exists='append',index=False)
        conn.close()
    
    def reset(self,seed = None, options=None,reset_account=None):
        self._idx=-1
        self._step = 0
        self._limit_orders = {}
        self._position=0
        reward=0
        

        if self._restore_trading:
            if self._load_history():
                reward=self.reward_function(self.historical_info)

            if reset_account:
                try:
                    self.reset_account()
                except Exception as e:
                    print(e)
        elif self._restore_trading==False:
            try:
                self.reset_account()
            except Exception as e:
                print(e)

        self._set_df(self.get_data())
        self.get_account()
        self._portfolio  = Portfolio(
                                    asset = self.get_balance(self.base_asset),
                                    fiat = self.get_balance(self.quote_asset),
                                    )
                                
        
        if self._restore_trading:
            position=None
            if reset_account:
                position=0
            
            if  self._load_history(position=position):
                hist_config=self.build_history(reward=reward)
                self.historical_info.add(**hist_config)

            else:
                self.historical_info = History(max_size=1000)
                hist_config=self.build_history(reward=reward,initial_record=True)
                self.historical_info.set(**hist_config)

        else:
            self.historical_info = History(max_size=1000)
            hist_config=self.build_history(reward=reward,initial_record=True)
            self.historical_info.set(**hist_config)
            
        self._save_history()
        
        return self._get_obs(), self.historical_info[-1]


class AlpacaTradingEnv(BaseLiveTradingEnv):
    def __init__(
            self,
            *args,
            **kwargs,
            ):
        
        super().__init__(**kwargs)
    
    def connect_client(self):
        if self.client!=None:
            del self.client

        client=AlpacaClient(api_key=self.api_key,
                            api_secret=self.api_secret,
                            paper=self.paper,
                            symbol='/'.join([self.base_asset,self.quote_asset]),
                            time_frame=self.time_frame)

        return client
    
    def set_trade_rules(self):
        '''
        Set the min and max trade_sizes and the values to round by for order precision rounding
        '''
        trade_rules=self.client.get_trade_rules()

        
        self.min_quote_size=trade_rules.get('min_quote_size')
        self.max_quote_size=trade_rules.get('max_quote_size')

        self.min_asset_size=trade_rules.get('min_asset_size')
        self.max_asset_size=trade_rules.get('max_asset_size')

        self.base_asset_precision=trade_rules.get('base_asset_precision')
        self.quote_asset_precision=trade_rules.get('quote_asset_precision')
    
    def get_account(self):
        account=self.client.account()
        return account
    
    def get_balance(self,symbol):

        bal=self.client.get_balance(symbol)

        return bal

    def _get_price(self):
        symb='/'.join([self.base_asset,self.quote_asset])
        price=float(self.client.ticker_price(symb))
        self._current_price=price
        return price

    def _trade(self, position, price = None):
        self.account=self.get_account()
        ## get size of assets we have
        n_dollars=self.get_balance(self.quote_asset)
        n_asset=self.get_balance(self.base_asset)
        price=self._get_price()
        
        new_position=position
    
        # position=self._portfolio.position(price)

        trade_from_to=[self._position,new_position]
        position_change=np.diff([trade_from_to])
        position_change=round(float(position_change),2)

        print(f'',position_change)

        ## decide which direction to go
        side=np.sign(position_change)
        side_str=self.side_map[side]
        self.update_portfolio()
        current_portfolio=self._portfolio.get_portfolio_distribution()
        current_portfolio['valuation']=self._portfolio.valorisation(price)
        

        params = {
            'symbol': self.symbol,
            'side':side_str,
            'type': 'MARKET',
            ## later we can expirement with limit orders
            # 'type': 'LIMIT',
            'time_in_force': 'ioc',

            }

        if side_str.upper()=='BUY':
            dollar_val=abs(position_change)*n_dollars
            asset_size=n_dollars/price
            trade_size=self.normalize_asset_size(asset_size)

        elif side_str.upper()=='SELL':
            asset_size=abs(position_change)*n_asset
            dollar_val=asset_size*price
            trade_size=self.normalize_quote_size(dollar_val)
        else:
            print('WTF',side_str)
            return
        params['notional']=trade_size
        print('portfoliio',current_portfolio)

        print(f'''Trade {trade_from_to[0]}->{trade_from_to[1]}
              {position_change}, so {round(dollar_val,2)}{self.quote_asset} / {asset_size:.8f} {self.base_asset},
              {side_str} {self.base_asset}:{asset_size} for {dollar_val:.2f}$ at {price} {self.symbol}'''
              )
        
        print(params)
        try: 
            self.client.new_order(**params)
        except Exception as e:
            print(e)
        self.update_portfolio()
        self._position = position
        return
    
    def get_klines(self):

        data=self.client.klines(self.symbol, self.time_frame, limit=self.context_length*2)
        data['date_close']=data['date_close'].apply(pd.to_datetime,unit='ms')

        data=data.set_index('date_close')
        data=data[['open','high','low','close','volume']]
        return data
    
    def live_trade(self):
        
        self.set_forecast_df()
        obs = self._get_obs()
        if self. onnx_model:
            action=self.onnx_infer(obs)        
        else:    
            action,_,states=self.agent.compute_single_action(obs,explore=False)
        obs, reward, terminated, truncated, info=self.live_step(action,wait=False)


    
    async def _live_stream_data_handler(self,message):

            do_trade=False
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
            data['unique_id']=symbol_map[self.symbol]

            with db.connect('Trade_history/trade.db') as conn:
                data.to_sql(f'{self.symbol}_candle_history',conn,if_exists='append',index=True)
  
            if bar_time.minute==0:
                
                do_trade=True
                print('--------------------------------')
                print(bar_time.strftime('%I:%M %p %m-%d-%Y'))
                
            
            return do_trade
    
    async def live_stream_step(self,message):
        # Get the current timestamp
        now = pd.Timestamp.now()

        # Round the timestamp to the nearest hour
        next_hour = now.floor(self.time_frame) + pd.Timedelta(self.time_frame)
        wait_time=(next_hour-now).total_seconds()
        wait_time=int(wait_time/60)
        wait_str=next_hour.strftime('%I:%M %p %m-%d-%Y')
        if not self.loading_bar:
            self.live_trade()
            self.loading_bar=tqdm(range(wait_time),desc=f'Next trade at {wait_str}',leave=True)
            self.loading_bar.update(1)

        do_trade= await self._live_stream_data_handler(message)

        if do_trade:
            self.loading_bar=tqdm(range(wait_time),desc=f'Next trade at {wait_str}',leave=True)
            self.live_trade()
            if self.render_forecasts:
                print(self.pred_df)
                

        else:
            self.loading_bar.set_description(f'Next trade at {wait_str} {wait_time} minutes:')
            self.loading_bar.update(1)
        

class OandaTradingEnv(BaseLiveTradingEnv):
    
    def __init__(
            self,
            account_id=None,
            *args,
            **kwargs,
            ):
        
        self.account_id=account_id
        
        super().__init__(**kwargs)
        self.symbol='_'.join([self.base_asset,self.quote_asset])

    def connect_client(self):
        if self.client!=None:
            del self.client

        client=OandaClient(api_key=self.api_key,
                            account_id=self.account_id,
                        
                            paper=self.paper,
                            symbol='_'.join([self.base_asset,self.quote_asset]),
                            time_frame=self.time_frame)

        return client
    
    def _get_price(self):
        symb='_'.join([self.base_asset,self.quote_asset])
        price=float(self.client.ticker_price(symb))
        self._current_price=price
        return price
    
    def _trade(self, position, price = None):
        self.account=self.get_account()
        ## get size of assets we have
        n_dollars=float(self.client._account['cash'])

        price=self._get_price()
        new_position=position
    
        trade_from_to=[self._position,new_position]
        position_change=np.diff([trade_from_to])
        position_change=round(float(position_change),2)

        print('--------------- TRADE ------------------')
        ## decide which direction to go
        side=np.sign(position_change)
        side_str=self.side_map[side]
        self.update_portfolio()
        
        current_portfolio=self._portfolio.get_portfolio_distribution()
        current_portfolio['valuation']=self._portfolio.valorisation(price)
        symbol='_'.join([self.base_asset,self.quote_asset])

        params = {
            'symbol': symbol,
            'side': side_str,
            'type': 'MARKET',
            'time_in_force': 'ioc',

            }
        print('Position_change',position_change)

        dollar_val=abs(position_change)*n_dollars
        
        asset_size=dollar_val*price
        dollar_val=self.normalize_quote_size(dollar_val)
        trade_size=self.normalize_asset_size(asset_size)

        params['notional']=round(dollar_val,0)
        
        print('portfoliio',current_portfolio)

        print(f'''
              Trade {trade_from_to[0]}->{trade_from_to[1]} \n
              so {side_str} {dollar_val} {self.base_asset} units or \n
              {trade_size} {self.quote_asset} 
              at {price} {self.quote_asset} per {self.base_asset}
            '''
              )
        
        print(params)
        try: 
            self.client.new_order(**params)
        except Exception as e:
            print(e)
        self.update_portfolio()
        self._position = position
        return
    
    def get_trade_params(self):
        pass

    def reset_account(self):
        self.client.update_positions()
        pos_frame=self.client._positions
        print(pos_frame)
        for symbol,row in pos_frame.iterrows():

            dump_size=abs(float(row['units']))
            side=row['side']

            
            params = {
                'symbol': self.symbol,
                'type': 'MARKET',
                'notional':dump_size,
                # 'quantity':trade_size,
                }
            if side.upper()=='SELL':
                params['side']='BUY'

            if side.upper()=='BUY':
                params['side']='SELL'   

            print('reset_account')
            print(params)

            self.client.new_order(**params)

    def close_positions(self):
        ## close all positions
        self.reset_account()

    def get_klines(self):


        data=self.client.klines(self.symbol, self.time_frame, limit=self.context_length*2)
        data['date_close']=data['date_close'].apply(pd.to_datetime)

        data=data.set_index('date_close')
        data=data[['open','high','low','close','volume']]
        return data
    
    def stream_trade(self):
        r = pricing.PricingStream(accountID=self.client.account_id, params={"instruments": "_".join([self.base_asset,self.quote_asset])})
        while True:
            first_tick=0
            try:
                # the stream requests returns a generator so we can do ...
                for tick in self.client.api.request(r):
                    time_stamp=pd.Timestamp(tick['time'])
                    if time_stamp.minute==0:
                        if first_tick==0:
                            self.live_trade()
                            first_tick+=1
                            self.update_loading_bar(reset=True)
                        else:
                            self.update_loading_bar(reset=False)
                    else:
                        first_tick=0
                        self.update_loading_bar(reset=False)
            except StreamTerminated as err:
                print(f"Stream processing ended because {err} ")

    def update_loading_bar(self, reset=False):
        now = pd.Timestamp.now()

        # Round the timestamp to the nearest hour
        next_hour = now.floor(self.time_frame) + pd.Timedelta(self.time_frame)
        wait_time=(next_hour-now).total_seconds()
        wait_time=int(wait_time)
        wait_str=next_hour.strftime('%I:%M %p %m-%d-%Y')
        if not isinstance(self.loading_bar,tqdm):
            self.loading_bar=tqdm(range(wait_time),desc=f'Next trade at {wait_str}',leave=True)
        
        if reset:
            self.loading_bar=tqdm(range(wait_time),desc=f'Next trade at {wait_str}',leave=True)

        self.loading_bar.update(1)
        if self.loading_bar.total==self.loading_bar.n:
            self.loading_bar.reset()