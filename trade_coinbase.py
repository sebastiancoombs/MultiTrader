

import warnings
warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore",category=ResourceWarning)


from environments.live_environments import BaseLiveTradingEnv
from neuralforecast.core import NeuralForecast
from configs import spot_defaults
from Keys import *
import pickle
import numpy as np
from utils import pearl_utils
from Pearl.pearl.utils.instantiations.environments.gym_environment import \
    GymEnvironment
import boto3
import shutil
import tempfile
with tempfile.TemporaryDirectory() as temp_dir:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")


        s3= boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        s3.download_file('coinbasetradehistory','trade.db','Trade_history/trade.db')
        forecast_model=NeuralForecast.load('MultiHeadForecastingModel/')


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



        agent_path='Agent/pearl_DOGEUSDT_Spot_model.pkl'

        agent=pickle.load(open(agent_path,'rb'))






        live_pearl_env=GymEnvironment(live_env)

        if __name__ == '__main__':
                
                observation,action_space=live_pearl_env.reset()
                agent.reset(observation, action_space)
                action=agent.act(exploit=True)
                action_result=live_pearl_env.step(int(action))
                agent.observe(action_result)
                action=agent.act(exploit=True)
                live_pearl_env.step(action)
                s3.upload_file('Trade_history/trade.db','coinbasetradehistory','trade.db',)
                # shutil.rmtree('/var/folders/nv/')

# live_env.client.account()








