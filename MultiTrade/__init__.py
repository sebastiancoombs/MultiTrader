
from gymnasium.envs.registration import register
import gym_trading_env
import numpy as np
from typing import Any
import utils.forecast_utils as forecast_utils
import utils

register(
    id='BoxTradingEnv',
    entry_point='MultiTrade.environments:BoxTradingEnv',
    disable_env_checker = True
)

register(
    id='MultiPairTradingEnv',
    entry_point='MultiTrade.environments:MultiPairTradingEnv',
    disable_env_checker = True
)
register(
    id='NeuralForecastingTradingEnv',
    entry_point='MultiTrade.environments:NeuralForecastingTradingEnv',
    disable_env_checker = True
)