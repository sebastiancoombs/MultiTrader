
from gymnasium.envs.registration import register

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