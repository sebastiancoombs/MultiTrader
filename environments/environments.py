
import matplotlib.dates
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
import gymnasium as gym
from gymnasium import spaces
from IPython.display import clear_output
import warnings
from gym_trading_env.environments import TradingEnv
from gym_trading_env.downloader import download
from gym_trading_env.utils.history import History
import matplotlib
from datetime import datetime, timedelta
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
import re
import glob
import copy
import os
from tqdm import tqdm
from IPython.display import display


class BoxTradingEnv(TradingEnv):

    def __init__(self,*args: Any, **kwargs: Any) -> None:
        super().__init__(*args,**kwargs)

        self. original_action_space=self.action_space
        self.box_positions=np.array(sorted(self.positions))


        self.action_space=gym.spaces.Box(
            high=max(self.box_positions),
            low=min(self.box_positions),
            shape=(1,),
        )

    def step(self,position_array=None):
        position_diff=self.box_positions-position_array
        position_index=np.argmin(position_diff)
        return super().step(position_index)
    

class NormTradingEnvironment(TradingEnv):
    def __init__(self, margin=1,*args, **kwargs):
        kwargs['positions']=[p*margin for p in kwargs['positions']]
        self.margin=margin
        super().__init__(*args, **kwargs)
        
    def _get_obs(self):
        obs=super()._get_obs()
        obs = self.norm_obs(obs)
        return obs
    
    def norm_obs(self, obs):
        return obs / np.linalg.norm(obs)
    
class NeuralForecastingTradingEnv(TradingEnv):
    def __init__(self,model=None,forecasts=None,forecast_horizon=7,context_length=35,pre_prep=True,*args,**kwargs):
        warnings.filterwarnings("ignore")
        warnings.simplefilter('ignore')
        super().__init__(*args,**kwargs)
        self.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
        self.add_metric('Episode Length', lambda history : len(history['position']) )
        self.portfolio_initial_value
        self.spec=gym.envs.registration.EnvSpec(
            id='NeuralForecastingTradingEnv',
            max_episode_steps=self.max_episode_duration,
            reward_threshold=100,
            entry_point="MultiTrade.environments:NeuralForecastingTradingEnv"
        
                )
        self._original_obs=super()._get_obs
        self.model=model
        self.pred_df=None

        if forecasts:
            pre_prepped=True
        else:
            pre_prepped=False

        self._pre_prep=pre_prep
        if not model:
            self.forecast_horizon=forecast_horizon
            self.context_length=context_length
        else:
            self.forecast_horizon=model.h,
            self.context_length=model.models[0].input_size
                
        self._forecast_array=forecasts
        if self._pre_prep:
            if not pre_prepped:
                self._prep_forecasts()

        forecast=self._get_forecast()
        # print(self._nb_features+len(forecast))
        
        self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape = [self._nb_features+len(forecast)]
            )
        
    def regularize(self,obs):
        norm=np.linalg.norm(obs,ord=2)
        normed_obs=obs/norm
        return normed_obs
    
    def _get_obs(self):
        feature_obs= self._original_obs()
        forecast=self._get_forecast()
        obs = np.concatenate([forecast,feature_obs]).flatten()
        obs = np.nan_to_num(obs)
        obs = self.regularize(obs)
        return obs
    
    def _prep_forecasts(self):
        forecast_array=[]
        # print(self.df.columns)
        self.model.dataset, self.model.uids, self.model.last_dates, self.model.ds=self.model._prepare_fit(self.df[['ds','unique_id','y']],
                                                                                            static_df=None, 
                                                                                            sort_df=None,
                                                                                            predict_only=False,
                                                                                            id_col='unique_id', 
                                                                                            time_col='ds', 
                                                                                            target_col='y')
        forecasts=self.model.predict_insample()
        self.pred_df=self.model.predict(self.df[['ds','unique_id','y']])
        
        forecasts_series=forecasts.groupby('cutoff').apply(lambda x: x.select_dtypes(np.number).values.flatten())
        forecast_array=[c for c in forecasts_series]
        self._forecast_array=forecast_array

        new_df=self.df[self.df['ds'].isin([c for c in forecasts_series.index])]
        if self.verbose:
            print(len(self.df),len(self._forecast_array),len(new_df))

        self._set_df(new_df)

    def make_pred(self,end):
        if end !=-1:
            end=end if end>0 else 1
        start=end-self.context_length if (end-self.context_length)>0 else 0
        
        data=self.df.iloc[start:end]
        data=data[['ds','unique_id','y']]
        self.model.dataset, self.model.uids, self.model.last_dates, self.model.ds=self.model._prepare_fit(data,
                                                                                            static_df=None, 
                                                                                            sort_df=None,
                                                                                            predict_only=False,
                                                                                            id_col='unique_id', 
                                                                                            time_col='ds', 
                                                                                            target_col='y')
        preds_array=np.array([model.predict(self.model.dataset) for model in self.model.models])
        preds_array=preds_array.flatten()
        clear_output()
        return preds_array

    def _get_forecast(self):
        if hasattr(self,'_idx'):
            _step_idx=self._idx
        else:
            _step_idx=0

        if self._pre_prep:
            forecast=self._forecast_array[_step_idx]
        else :
            forecast=self.make_pred(end=_step_idx)

        return forecast


## Define our ENV who will make the trades for the agent
class MultiSymbolTradingEnv(gym.Env):
    def __init__(self,
                 instrument=['Forex'], 
                 target_symbols=['ETHUSDT'],
                 base_currency='USD',
                 model=None,
                 reward_function=None,
                 forecast_horizon=7,
                 context_length=35,
                 pre_prep=True,
                 actions=[-1,0,1],
                 initial_cash=10000,
                 leverage=200,
                 fees=.00003,
                 spread=.00003,
                 fixed_trades=False,
                 trade_size_pct=.2,
                 fixed_trade_size=1.1,
                 interval=None,
                 download_data=True,
                 process_data=True,
                 process_data_func=None,
                 verbose=1,
                 data_dir='data/',
                 max_episode_duration=168,
                 seed=4,
                        ):
        
        super().__init__()
        self.verbose=verbose
        ## save init variables
        ## the instrument you want to trade Forex, Crypto, Stocks
        self.instrument=instrument
        self.sim_kwargs= {'reward_function':reward_function,
                    'forecast_horizon':forecast_horizon,
                    'context_length':context_length,
                    'pre_prep':pre_prep,
                    'positions':actions
                    }
        self.simulators={}
        self.no_action_idx=actions.index(0)
        ## the symbols you want to trade BNBUSD, BTCUSD, GPBUSD
        self.target_symbols=target_symbols
        ## base_curenncy USD, EUR, GBP
        self.base_currency=base_currency
        self.forecast_model=None

        self. model=model
        ## START_DATa and END_DATE for trading
        
        self.start_date=start_date
        self.end_date=end_date
        self.actions=actions
        self.initial_cash=initial_cash
        ## the amount of leverage you want to use per trade
        self.leverage=leverage

        ## the fees and spread you want
        self.fees=fees
        self.spread=spread

        ## used a fixed trade size or a percentage of the portfolio
        self.fixed_trades=fixed_trades

        ## the percentage of the portfolio you want to trade if not fixed
        self.trade_size_pct=trade_size_pct  

        ## the fixed trade size if you want to use if fixed
        self.fixed_trade_size=fixed_trade_size

        ## trading interval 15min, 1h, 1d, 1w
        self.interval=interval
        
        self.download_data=download_data
        self.process_data=process_data
        self.process_data_func=process_data_func
        self.processed_dir=data_dir+'/processed'
        self.data_dir=data_dir

        self.sim_data_dir=self.create_save_dir()
        self.signal_features=None
        self.best_found = 0
        
        start_date=pd.Timestamp(start_date)
        end_date=pd.Timestamp(end_date)

        ## SAVE THE DATE
        self.process_date=start_date-timedelta(days=60)
        self.start_date=start_date
        self.end_date=end_date
        ## info about your account is need frst to initialize the simulators

        self.timeframe=interval
        self.trade_week_start=None

        ## info about your account
        self.balance=initial_cash
        self.equity= initial_cash
        self.old_equity=initial_cash
        ## The minimum balance you want to keep in your account before you stop trading
        self.min_balance=initial_cash*.2

        ## profit you remove from your account and save
        self.savings=0
        ## total profit you have made
        self.profit=0
        self.margin= initial_cash
        self.free_margin=initial_cash
        self.unrealized_profit=0
        self.margin_level= None
        self.orders=None
        self.step_profit=0

        ## data transformers
        self.scaler=MaxAbsScaler()
        self.encoder=OneHotEncoder()
        self.done=False
        self.tz=None

        self.total_sim_times=None
        self.save_path=None

        # TODO CHANGE THIS TO GYM-TRADING-ENV
        sim=self.load_simulators()

        self.original_sim = sim ## original simulator to use during reset
        self.sim = copy.deepcopy(sim) ## this explictly need to be the MT trade env
        self.full_reset=False

        ## terminated_truncated
        self.truncated=False
        self.terminated=False
        self.max_episode_duration=self.data[target_symbols[0]].shape[0]

        self.spec = gym.envs.registration.EnvSpec(
            id='MultiSymbolTradingEnv',
            max_episode_steps=self.max_episode_duration,
            entry_point="TradeGymAny.envs.TradeGymMulti:MultiSymbolTradeEnv"
                )

        encode_list,encode_dict=self.make_encode_list()

        self.encoder.fit(encode_list)

        self.actions_list=actions
        # dict of discrete actions based on what symbols you have defined as avaiable to trade
        self.positions=actions
        # dict of discrete actions coresponding to the symbol and buy/sell action
        self.actions_mapping=encode_dict
        self.encode_list=encode_list
        
        # dict of current positons updated to be size trade and sort/long as trading happens all start at 0
        self.current_positions = {symbol:0 for symbol in self.target_symbols}


        ## discrete action spaces idx = [0] used to determine symbol to trade, idx = [1] buy/hold/sell, idx = [2] size of trade

        self.action_space = spaces.Discrete(len(encode_list)) ## number of actions and size of trade 0-10 units

        self.reward = np.zeros(1)
        self.last_obs=None
        obs=self.get_obs()
        
        self.observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape,dtype = np.float32),

        ## history_frames
        self.history= None
        self.make_history()
        
    def make_history(self):
        self.historical_info = History(max_size= len(self.df))
        history_args=dict(idx = self._idx,
            step = self._step,
            date = self.df.index.values[self._idx],
            position_index =self.positions.index(self._position),
            position = self._position,
            real_position = self._position,
            data =  dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation = self.portfolio_initial_value,
            portfolio_distribution = self._portfolio.get_portfolio_distribution(),
            reward = 0,)
        
        for symbol in self.target_symbols:
            history_args[symbol]=0

        self.historical_info.set(
            **history_args
        )
        return pd.DataFrame(columns=['equity','savings','balance']+self.target_symbols)

    def create_save_dir(self):

        symbol_file_path=f'{self.instrument[0]}/simulaton/{self.interval}'
        file_path=os.path.join(self.data_dir,symbol_file_path)
        if not os.path.isdir(file_path):
            os.makedirs(file_path)

        return file_path + '/Sim_data.pkl'

    def make_encode_list(self):
        encode_list=[]
        encode_dict={}
        j=0
        for symbol in enumerate( self.target_symbols):
            for i,action in enumerate(self.actions):
                act_dict={'symbol' : symbol,
                        'action' : action,
                        'action_idx' : i,
                        }
                encode_list.append([act_dict['symbol'],act_dict['action']]) 
                
                encode_dict[j]=act_dict
                j+=1
        j+=1
        ## add the no action/ close all trades action
        encode_dict[j]={'symbol':'all','action':self.no_action_idx}

        return encode_list,encode_dict
 
    def load_simulators(self):
        
        start_date=self.check_datetime(self.process_date)
        end_date=self.check_datetime(self.end_date)
        instrument=self.instrument
        timeframe=self.timeframe

        # get the base currencies of the symbols
        reg_code=fr'_|-|/|{self.base_currency}'

        symbols_striped=[re.sub(sym,reg_code,'') for sym in self.target_symbols]
        symbols=list(set(symbols_striped))+[self.base_currency]
        available_symbols=glob.glob(f'{self.data_dir}/*.pkl')

        associated_pairs=[ pair for pair in available_symbols if any([sym in pair for sym in symbols])]

        all_symbs=list(set(associated_pairs))

        self.symbols=all_symbs

        if self.download_data:
            
            download(exchange_names = ["binance-us"],
                symbols= tqdm(self.target_symbols),
                timeframe= self.timeframe,
                dir = self.data_dir,
                since= self.start_date,
            )

        if self.model:
            env_class=NeuralForecastingTradingEnv
        else:
            env_class=TradingEnv

        available_data=glob.glob(f'{self.data_dir}/*.pkl')
        for symbol in self.target_symbols:
            sim_kwargs=copy.deepcopy(self.sim_kwargs)
            
            data_path=[d for d in available_data if symbol in d][0]
            data=pd.read_pickle(data_path)
            if self.process_data:
                data=self.process_data_func(data)
            
            self.simulators[symbol]=env_class(
                model=self.model,
                df = data, # Your dataset with your custom features
                
                max_episode_duration=168,
                verbose=0,
                **sim_kwargs
                )

        data_path=self.sim_data_dir

            
        data=self.simulators[self.target_symbols[0]].df
        self.tz=data.index.tz
        start_date=self.check_datetime(self.start_date)
        end_date=self.check_datetime(self.end_date)
        total_sim_times=data[start_date:end_date].index
        if self.verbose:
            print('------------------Loaded data---------------------')
            print('start date:',start_date)
            print('actual start date:',total_sim_times[0])
            print('end date:',end_date)
            print('actuall end date:',total_sim_times[-1])
            print('total time steps:',len(total_sim_times))


        self.total_sim_times=total_sim_times
        
    def get_obs(self):
        big_obs= np.array([env._get_obs() for env in self.simulators.values()])
        obs=big_obs.flatten()
        obs=np.nan_to_num(obs)
        pos_list=[[p,np.sign(self.current_positions[p])] for p in self.current_positions]
        
        return obs
    
    ##TODO
    def get_reward(self,info_list):      
        ## TODO
        ## loop through the info list and get the rewards
        ## for each symbol
        reward=0
        return reward
    
    ## TODO
    def update_env_infos(self, verbose=True):
        for env in self.simulators.values():
            env.get_info(verbose=verbose)
            env.update_info(verbose=verbose)
        maininfo=''
        return 
    
    ## TODO
    def step(self,action,verbose=False): 
        action_dict=self.actions_dict[action]
        act_symbol=action_dict['symbol']
        action=action_dict['action']
        action_idx=action_dict['action_idx']
        info_list=[]
        obs_list=[]
        for symbol,env in self.simulators.items():
            if symbol==act_symbol:
                obs,rew,trunc,term,info=env.step(action_idx)
            else:
                obs,rew,trunc,term,info=env.step(self.no_action_idx)
            info_list.append(info)
        
        # move forward in time
        terminated,truncated=self.compute_terminated()

        if terminated:
            self.close_all_trades()
        
        info=self.update_info(verbose=verbose)

        #recieve the reward
        obs=self.make_obs()

        if self.observation_type=='Dictionary':
            reward=self.compute_reward(obs["achieved_goal"], obs["desired_goal"],info=info)
        else: 
            reward=self.get_reward()

        step_reward=self.get_reward()
        info['is_success']=self._trade_success
        info['truncated']= truncated
        info['terminated']= terminated
        # print(rew,reward)

        return obs, reward, terminated,truncated, info

    def reset(self,
                    seed,
                    options
                ):
        for env in self.simulators.values():
            obs,info=env.reset(seed,**options)
        
        self.current_positions = {symbol:self.no_action_idx for symbol in self.target_symbols}
        self.history=self.make_history()

        first_time=self.total_sim_times[0]
        first_time=self.check_datetime(first_time)

        self.full_reset=False
        self.current_trades=np.zeros(len(self.target_symbols))-.00001

        info=self.update_info(verbose=False)
        obs=self.make_obs()
        return obs ,info

    def render(self,mode="human",save=False):
        if mode !='human':
            display(self.history)
        else:
            orders=self.orders
            equity=self.history['equity']
            savings=self.history['savings']
            rows=len(orders.Symbol.unique())
            plot=int(f'{rows}{2}{1}')
            fig=plt.figure(figsize=(15, 12),layout ='tight')

            ax=fig.add_subplot(plot)
            fig.suptitle('Account + Orders')
            date_form = matplotlib.dates.DateFormatter("%m-%d")
            ax.xaxis.set_major_formatter(date_form)
            ## plot balance
            ax.set_title(f'Balance: \${self.balance :.2f} Equity: \${self.equity:.2f} Savings  \${self.savings :.2f} Profit on {len(orders)} trades')
            ax.plot(equity,label='agent equity')
            ax.set_ylabel('$ Agent equity balance', color='Blue')

            ## plot savings
            axes3 = ax.twinx()
            axes3.plot(savings,label='Agent Savings Balance',color='green')
            axes3.set_ylabel('$ Savings', color='green')
            axes3.legend()

            try:
                start=self.check_datetime(orders['Entry Time'].values[-1])
                end=self.check_datetime(equity.index.values[-1])
                display(start,end)

                for symbol ,data in orders.groupby('Symbol'):
                    plot+=1   
                    ax=fig.add_subplot(plot)
                    ax.xaxis.set_major_formatter(date_form)
                    ax.set_title(f'{symbol} Orders')
                    buys=data[data.Type=='Buy']
                    sells=data[data.Type=='Sell']
                    prices=self.sim.symbols_data[symbol].loc[start:end,'Close']
                    for i, row in buys.iterrows():
                        x=[row['Entry Time'],row['Exit Time']]
                        y=[row['Entry Price'],row['Exit Price']]

                        ax.scatter(row['Entry Time'],row['Entry Price'],marker='^')
                        ax.scatter(row['Exit Time'],row['Exit Price'],marker='v')
                        profit=row['Profit']
                        if profit >0:
                            ax.plot(x,y,'--',c='green')
                        else:
                            ax.plot(x,y,'--',c='red')

                    for i ,row in sells.iterrows():
                        x=[row['Entry Time'],row['Exit Time']]
                        y=[row['Entry Price'],row['Exit Price']]
                        ax.scatter(row['Entry Time'],row['Entry Price'],marker='v')
                        ax.scatter(row['Exit Time'],row['Exit Price'],marker='^')
                        profit=row['Profit']
                        if profit >0:
                            ax.plot(x,y,'--',c='green')
                        else:
                            ax.plot(x,y,'--',c='red')
                    ax.plot(prices,label=f'{symbol} Close Prices')
                    ax.legend()
                plt.xlabel('Time')
                fig.show()
                if save:
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    plt.savefig(self.save_path)

            except Exception as e:
                print(e)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """All the available environments are currently continuing tasks and non-time dependent. The objective is to reach the goal for an indefinite period of time."""
        return False
    
    def compute_truncated(self, achievec_goal, desired_goal, info):
        """The environments will be truncated only if setting a time limit with max_steps which will automatically wrap the environment in a gym TimeLimit wrapper."""
        return False
    
