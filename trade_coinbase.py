

import warnings
warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore",category=ResourceWarning)


from environments.live_environments import BaseLiveTradingEnv
from neuralforecast.core import NeuralForecast

from Keys import *
import pickle
import numpy as np
from utils import pearl_utils,save_utils
from Pearl.pearl.utils.instantiations.environments.gym_environment import \
    GymEnvironment
import boto3
import shutil
import datetime
import tempfile

from IPython.display import display
with tempfile.TemporaryDirectory() as temp_dir:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        from configs import spot_defaults as defaults



        base_asset=defaults.base_asset
        quote_asset=defaults.quote_asset
        test_net=False
        time_frame=defaults.time_frame

        product_type=defaults.product_type
        futures_target='DOG-29NOV24-CDE'
        exchange=defaults.exchange
        trade_target='/'.join([base_asset,quote_asset]) 
        agent_path=defaults.agent_path


        s3= boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)


        print('Downloading files from S3')
        print(f'USING {defaults.forecasting_model_path}')
        s3.download_file('coinbasetradehistory','trade.db','Trade_history/trade.db')

        print(f'Downloading {agent_path}')
        s3.download_file('coinbasetradehistory',agent_path,agent_path)

        agent=save_utils.load_agent(agent_path)
        forecast_model=NeuralForecast.load(defaults.forecasting_model_path)

        agent=pickle.load(open(agent_path,'rb'))
        forecast_model=NeuralForecast.load(defaults.forecasting_model_path)

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
                    discord_webhook=discord_crypto_webhook,

                    )


        



        live_pearl_env=GymEnvironment(live_env)


                
        observation,action_space=live_pearl_env.reset()

        if datetime.datetime.now().hour==0 and datetime.datetime.now().weekday()==1:
            agent.learn()
            agent.reset(observation,action_space)

        
        else:
            last_action=live_env.get_last_action()
            # action=agent.act(exploit=True)
            live_pearl_env.env.allow_trade_submit=False
            action_result=live_pearl_env.step(action=last_action)
            agent.observe(action_result)


        live_pearl_env.env.allow_trade_submit=True
        action=agent.act(exploit=True)
        live_pearl_env.step(action)
        

        s3.upload_file('Trade_history/trade.db','coinbasetradehistory','trade.db',)
        s3.upload_file(agent_path,'coinbasetradehistory',agent_path)

        display(live_env.client.account())










