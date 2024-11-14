
import warnings
import logging


warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


from utils import make_environments
from utils import pearl_utils
from configs.defaults import forecasting_model_path
from configs.defaults import target_pair
from utils.reward_functions import log_reward_function,cumulative_reward_function,sharpe_reward_function
from utils. utils import make_hidden_dims
import optuna
from optuna.samplers import TPESampler
from neuralforecast.core import NeuralForecast
from Pearl.pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from Pearl.pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
import datetime

import numpy as np
import pickle

reward_functions=[log_reward_function,cumulative_reward_function,sharpe_reward_function]
train_env,test_env=make_environments.make_envs(reward_function=log_reward_function)


today=datetime.datetime.now().strftime('%Y-%m-%d')
study = optuna.create_study(study_name=f"pearl-2024-11-12-hp-search",
                            directions=["maximize", "maximize"],
                            storage="sqlite:///pearl_hyper_parameters.sqlite3",
                            load_if_exists=True,
                            sampler=TPESampler()
                            )



# print(f"Best value: {study.best_value} (params: {study.best_params})")
best_trials=study.best_trials
best_trials


if -1 in [np.sign(p) for p in test_env.positions]:
    market_type='Futures'
else:
    market_type='Spot'
symb=target_pair.replace('/','')

agent_path=f'Agent/pearl_{symb}_{market_type}_model.pkl'

agent,learning_params=pearl_utils.load_agent_from_study(study_path="sqlite:///pearl_hyper_parameters.sqlite3",
                                            study_name='pearl-2024-11-12-hp-search',
                                            action_space_dim=2,
                                            observation_space_dim=30,)

agent=pearl_utils.train_production_agent(agent,
                             learning_params,
                             train_env=train_env,
                             test_env=test_env,
                             save_path=agent_path)


agent,learning_params=pearl_utils.load_agent_from_study(study_path="sqlite:///pearl_hyper_parameters.sqlite3",
                                        study_name='pearl-2024-11-12-hp-search',
                                        action_space_dim=2,
                                        observation_space_dim=30)
agent=pearl_utils.load_agent_weights(agent,weight_path=agent_path)


profit,n_trades=pearl_utils.test_pearl_model(agent,test_env)
print(f"Testing Return AVG Profit: {profit}, AVG Number of Trades: {n_trades}")





