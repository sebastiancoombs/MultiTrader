
import pandas as pd

import numpy as np
from typing import Any
import gymnasium as gym
from gymnasium import spaces
from tqdm.autonotebook import tqdm
from IPython.display import display,clear_output
from utils.utils import build_dataset, build_market_image, preprocess_data,symbol_map,sharpe_reward
from utils.forecast_utils import create_ts_preprocessor,create_ts_dataset

import warnings
from neuralforecast.core import NeuralForecast
import datetime

import sqlite3 as db
from gym_trading_env.environments import TradingEnv

from gym_trading_env.utils.history import History

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
    
class NeuralForecastingTradingEnv(TradingEnv):
    def __init__(self,model=None,forecasts=None,forecast_horizon=7,context_length=35,pre_prep=True,*args,**kwargs):
        warnings.filterwarnings("ignore")
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
        
    
    def _get_obs(self):
        feature_obs= self._original_obs()
        forecast=self._get_forecast()
        forecast=np.concatenate([forecast,feature_obs]).flatten()
        forecast=np.nan_to_num(forecast)
        return forecast
    
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
    

class MultiTradeEnv(gym.Env):
    def __init__(self,
                 pairs=[],
                 data_dict={},
                 positions=[-.25,0,.25],

                 ):
        self.envs={}
        self.pairs=pairs
        self.n_pairs=len(pairs)
        positions=positions/self.n_pairs

        for name,data in data_dict.items():
            self.envs[name]=self.make_env(name,data,positions=positions)
        
        self.action_space=NotImplemented
        self.observation_space=NotImplemented


        def step(self,action):
            action=action/len(action)## normalize actions by the total so they are percentages of the total portfolio
            observations=[]
            rewards=[]
            dones=[]
            truncateds=[]
            infos={}
            for name,act in zip(action,self.pairs):
                obs, rew, dun,trunc, info=self.envs[name].step(act)
                observations.append(obs)
                rewards.append(rew)
                dones.append(dun)
                truncateds.append(trunc)
                for i in info.keys():
                    infos[f'{name}_{i}']=info[i]
            
            observation=np.array(observations).flatten()
            reward=sum(rewards)
            done=any(done)
            truncated=any(truncateds)
            return observation,reward,done,truncated,infos
        


        def reset(self):
            observations=[]
            for name,env in self.envs.items():
                obs=env.reset()
                observations.append(obs)

            observation=np.array(observations).flatten()
            return observation

        def make_env(self,name,data,positions=[ -1, 0, 1], 
                     trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
                borrow_interest_rate= 0.0003/100,# 0.0003% per timestep (one timestep = 1h here)
                ):
            env=gym.make("BoxTradingEnv",
                name=name,
                df = data, # Your dataset with your custom features
                positions = positions, # -1 (=SHORT), 0(=OUT), +1 (=LONG)
                trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
                borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
            )
            return env
