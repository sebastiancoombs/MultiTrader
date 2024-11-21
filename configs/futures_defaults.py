COIN_PAIRS=['DOGE/USDT','LTC/USDT','SHIB/USDT','ETH/USDT','BTC/USDT','XRP/USDT','BNB/USDT'
            ]
target_pair = 'DOGEUSD_Futures'
time_frame = '1h'
forecasting_model_path=f'ForecastingModels/{target_pair}ForecastingModel/'
DATA_DIR='data_futures'
model_name=target_pair
env_config = dict(
                name='Symbol_train',
                
                positions = [ -1,0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
                trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
                borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
                max_episode_duration=168,
                verbose=0
                )
