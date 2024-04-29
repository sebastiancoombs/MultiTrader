# %%
import datetime
import glob
from functools import lru_cache, partial
from pprint import pprint

import gym_trading_env
import gymnasium as gym
import matplotlib.pyplot as plt
import MultiTrade
import numpy as np
import pandas as pd
import torch
from gym_trading_env.downloader import download
from gym_trading_env.environments import TradingEnv
from gym_trading_env.renderer import Renderer
from IPython.display import display
from ray import train, tune
from tqdm.autonotebook import tqdm
from utils.utils import build_dataset, build_market_image,preprocess_data,stack_arrays
from utils.forecast_utils import create_ts_preprocessor,create_ts_dataset
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.time_feature import get_lags_for_frequency
from datasets import load_dataset,Dataset,DatasetDict
from MultiTrade.environments import ForecastingTradingEnv

from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
)
import ray

from pearl.pearl_agent import PearlAgent
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.neural_networks.sequential_decision_making.q_value_networks import VanillaQValueNetwork
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.policy_learners.sequential_decision_making.deep_q_learning import DeepQLearning
from pearl.policy_learners.sequential_decision_making.double_dqn import DoubleDQN
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.action_representation_modules.one_hot_action_representation_module import OneHotActionTensorRepresentationModule

COIN_PAIRS=['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT','XRP/USDT','ADA/USDT',
            'ETH/BTC','SOL/ETH','BNB/ETH','XRP/ETH',"ADA/ETH",
            'SOL/BTC','SOL/BNB',
            'XRP/BTC','XRP/BNB',
            'ADA/BTC','ADA/BNB',
            ]
target_pair='ETHUSDT'
time_frame="1h"

model=PatchTSTForPrediction.from_pretrained("C:/Users/standard/Git/MultiTrader/forecaster_pretrain/output/checkpoint-19392")

import warnings
warnings.filterwarnings('ignore')

start_date=datetime.datetime(year= 2024, month= 2, day=1)
split_date=start_date+datetime.timedelta(days=30)
end_date=split_date+datetime.timedelta(days=30)
target_pair='ETH/USDT'

data=build_market_image(target_pair=target_pair,time_frame='1h',axis=0)
data=data[data['symbol']==target_pair.replace('/','')].copy()


split_date=datetime.datetime(year= 2024, month= 3, day=1)
end_date=split_date+datetime.timedelta(days=14)

# data=build_market_image(target_pair='ETH/USDT',time_frame='1h')

hf_data=data.copy()
hf_train_data=hf_data.groupby('symbol').apply(lambda x: x[:split_date])
hf_test_data=hf_data.groupby('symbol').apply(lambda x: x[split_date:end_date])
hf_test_data=hf_test_data.reset_index(level=0,drop=True).reset_index()

def get_train_test_envs(data,symbol='ETHUSDT',look_back=7):
    start_date=datetime.datetime(year= 2024, month= 1, day=1)
    split_date=datetime.datetime(year= 2024, month= 3, day=1)
    end_date=split_date+datetime.timedelta(days=14)

    
    hf_data=data.copy()

    time_series_preprocessor=create_ts_preprocessor(hf_data.reset_index())
    time_series_preprocessor=time_series_preprocessor.train(hf_data.reset_index())

    hf_data=time_series_preprocessor.preprocess(hf_data)
    
    hf_train_data=hf_data.groupby('symbol').apply(lambda x: x[:split_date])
    hf_test_data=hf_data.groupby('symbol').apply(lambda x: x[split_date:end_date])

    hf_train_data=hf_train_data.reset_index(level=0,drop=True).reset_index()
    hf_test_data=hf_test_data.reset_index(level=0,drop=True).reset_index()


    trade_data=data[data['symbol']==symbol].copy()
    train_data=trade_data[:split_date]
    test_data=trade_data[split_date:end_date]


    train_env = ForecastingTradingEnv(
                                        model=model,
                                        hf_data=hf_train_data,
                                        name='ETHUSDT_train',
                                        df = train_data, # Your dataset with your custom features
                                        positions = [ -.25, 0, .25], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
                                        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
                                        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
                                        max_episode_duration=168,
                                        )
    
    test_env = ForecastingTradingEnv(
                                        model=model,
                                        hf_data=hf_test_data,
                                        
                                        name='ETHUSDT_test',
                                        df = test_data, # Your dataset with your custom features
                                        positions = [ -.25, 0, .25], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
                                        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
                                        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
                                        max_episode_duration=168,
                                        
                                    )
    return train_env,test_env


train_env,test_env=get_train_test_envs(data,look_back=7)
train_pearl_env=GymEnvironment(train_env)
test_pearl_env=GymEnvironment(test_env)

obs=train_env.reset()

search_space={
        # "look_back" : tune.choice([7,14,21,30,45,60]),

        "hidden_dims" : tune.choice([[64,64],[128,128],[256,256]]),

        'learning_rate':tune.uniform(1e-6, 1e-2),

        'discount_factor': tune.uniform(1e-6, 1),

        'training_rounds': tune.choice([c for c in range(2,64,2)]),

        'batch_size': tune.choice([64,128,256]),
        
        'target_update_freq':tune.choice([c for c in range(2,64,2)]),

        'soft_update_tau': tune.uniform(1e-6, 1),  # a value of 1 indicates no soft updates
        
        "replay_buffer_size":tune.choice([c for c in range(10,1_000,10)]),
        }


def plot_pearl(pearl_env):
    naked_env=pearl_env.env.unwrapped
    value_history=naked_env.historical_info['portfolio_valuation']
    x=np.arange(len(value_history))
    y=value_history
    plt.plot(x,y)



@ray.remote
def objective(config):

    hidden_dims=list(config["hidden_dims"])
    replay_buffer_size=config["replay_buffer_size"]


    # Instead of using the 'network_type' argument, use the 'network_instance' argument.
    # Pass Q_value_network as the `network_instance` to the `DeepQLearning` policy learner.
    # We will be using a one hot representation for representing actions. So take action_dim = num_actions.
    Q_network_DoubleDQN = VanillaQValueNetwork(state_dim=train_env.observation_space.shape[0],  # dimension of the state representation
                                                action_dim=train_env.action_space.n,                        # dimension of the action representation
                                                hidden_dims=hidden_dims,                       # dimensions of the intermediate layers
                                                output_dim=1)  
    # Instead of using the 'network_type' argument, use the 'network_instance' argument.
    # Pass Q_value_network as the `network_instance` to the `DoubleDQN` policy learner.
    DoubleDQNagent = PearlAgent(
                                policy_learner=DoubleDQN(
                                                            state_dim=train_env.observation_space.shape[0],
                                                            action_space=train_env.action_space,

                                                            network_instance=Q_network_DoubleDQN,   # pass an instance of Q value network to the policy learner.
                                                            action_representation_module=OneHotActionTensorRepresentationModule(
                                                                                                                                    max_number_actions=train_env.action_space.n
                                                                                                                                ),
                                                                                                                                
                                                            **config
                                                        ),
                                replay_buffer=FIFOOffPolicyReplayBuffer(replay_buffer_size),
                                ) 
    ## train dat bitch               
    info = online_learning(
                            agent=DoubleDQNagent ,
                            env=train_pearl_env,
                            number_of_episodes=20_000,
                            print_every_x_episodes=100,   # print returns after every 10 episdoes
                            learn_after_episode=True,    # updating after every environment interaction, Q networks are updates at the end of each episode
                            seed=0
                            )
    # plot_results(info)
    agent=DoubleDQNagent
    observation, action_space = test_pearl_env.reset()
    agent.reset(observation, action_space)
    done = False
    while not done:
        action = agent.act(exploit=True)
        action_result = test_pearl_env.step(action)
        agent.observe(action_result)
        agent.learn()
        done = action_result.done

    # plot_pearl(test_env)
    score=action_result.info['portfolio_valuation']/1000
    loss={"score": score,
            "_metric": score}
    print(loss)
    return loss


def trial_str_creator(trial):
    return "{}_{}_trading_agent".format(trial.trainable_name, trial.trial_id)

tune_config=tune.TuneConfig(num_samples=100,mode="max",search_alg='hyperopt',
                                trial_name_creator=trial_str_creator,
                                trial_dirname_creator=trial_str_creator,

                            )
run_config=train.RunConfig(
                            storage_path='C:/Users/standard/OneDrive/Documents/Git/MultiTrader/tune_results', 
                            name="DDQN_experiments"
                            )
scaling_config=train.ScalingConfig(num_workers=20
                                   
                                   )
# objective_with_resources = tune.with_resources(objective, {"cpu": 0.5})
objective_with_resources = tune.with_resources(objective, {"gpu": 1})


if not ray.is_initialized():
    ray.init()
else:
    ray.shutdown()
    ray.init()


tuner = tune.Tuner(objective_with_resources ,
                   tune_config=tune_config,
                   run_config=run_config,
                #    scaling_config=scaling_config,
                    param_space=search_space)  




results = tuner.fit()
print(results.get_best_result(metric="score", mode="max").config)

best_params=results.get_best_result(metric="score", mode="max").config
best_params

test_env.env.unwrapped.save_for_render(dir = "test_render_logs")

renderer = Renderer(render_logs_dir = "test_render_logs")

