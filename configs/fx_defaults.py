COIN_PAIRS=['EUR_USD','USD_JPY','GBP_USD','AUD_USD','USD_CAD','USD_CHF','NZD_USD'
            ]
base_asset='USD'
quote_asset='JPY'
target_pair = ''.join([base_asset,quote_asset])
time_frame = '1h'
product_type='FOREX'
forecasting_model_path=f'ForecastingModels/{target_pair}ForecastingModel/'

DATA_DIR='data_forex'
model_name=target_pair
agent_path=f'Agent/pearl_{model_name}_model.pkl'
exchange='oanda'
env_config = dict(
                name='Symbol_train',
                positions = [ -20, 20], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
                trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
                borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
                max_episode_duration=168,
                margin=20,
                verbose=0
                )
