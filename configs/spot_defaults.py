COIN_PAIRS=['DOGE/USDT','LTC/USDT','SHIB/USDT','ETH/USDT','BTC/USDT','XRP/USDT','BNB/USDT'
            ]
target_pair = 'DOGEUSDT'
time_frame = '1h'
forecasting_model_path=f'{target_pair}ForecastingModel/'
DATA_DIR='data'
model_name=f'{target_pair}SPOT'


env_config = dict(
                name='Symbol_train',
                
                positions = [ 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
                trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
                borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
                max_episode_duration=168,
                verbose=0
                )
