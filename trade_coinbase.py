

import warnings
warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore",category=ResourceWarning)


from environments.live_environments import BaseLiveTradingEnv
from neuralforecast.core import NeuralForecast
from configs import defaults
from Keys import *
import pickle
import numpy as np
from utils import pearl_utils
from Pearl.pearl.utils.instantiations.environments.gym_environment import \
    GymEnvironment
import boto3
import shutil
import tempfile
from IPython.display import display
from configs import defaults
with tempfile.TemporaryDirectory() as temp_dir:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")


        s3= boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        s3.download_file('coinbasetradehistory','trade.db','Trade_history/trade.db')
        forecast_model=NeuralForecast.load(defaults.forecasting_model_path)


        base_asset='DOGE'
        quote_asset='USDC'
        test_net=False
        time_frame='1h'
        # product_type='FUTURE'
        product_type='Spot'
        futures_target='DOG-29NOV24-CDE'
        exchange='coinbase'
        trade_target='/'.join([base_asset,quote_asset]) if product_type.upper()=='SPOT' else  futures_target
        trade_target


        live_env=BaseLiveTradingEnv(
                    api_key=coinbase_api_key,
                    api_secret=coinbase_api_secret,
                    paper=test_net,
                    symbol=trade_target,
                    time_frame=time_frame,
                    product_type=product_type,
                    positions=[0,1],
                    history_path='Trade_history/trade.db',
                    exchange=exchange,
                    forecast_model=forecast_model,
                    discord_webhook=disord_webhook,

                    )



        agent_path=f'Agent/pearl_{defaults.model_name}_model.pkl'
        s3.download_file('coinbasetradehistory',f'pearl_{defaults.model_name}_model.pkl',agent_path)

        agent=pickle.load(open(agent_path,'rb'))

        live_pearl_env=GymEnvironment(live_env)


                
        observation,action_space=live_pearl_env.reset()
        # agent.reset(observation, action_space)
        current_position=int(live_env.client.get_current_position())
        # action=agent.act(exploit=True)
        live_pearl_env.env.allow_trade_submit=False
        action=live_env.action_map[current_position]
        action_result=live_pearl_env.step(action=current_position)

        live_pearl_env.env.allow_trade_submit=True
        agent.observe(action_result)
        action=agent.act(exploit=True)
        live_pearl_env.step(action)
        
        s3.upload_file('Trade_history/trade.db','coinbasetradehistory','trade.db',)
        s3.upload_file(agent_path,'coinbasetradehistory',agent_path.split('/')[-1],)

        display(live_env.client.account())










