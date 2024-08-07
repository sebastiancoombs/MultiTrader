
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from .mappings import *

import datetime
from gluonts.time_feature import time_features_from_frequency_str


def preprocess_data(data,time_frame='1h'):
    returns=[2,5,10,15,20,25,30]
    time_funcs=time_features_from_frequency_str(time_frame)
    data['close']=data['close'].apply(pd.to_numeric)
    data['open']=data['open'].apply(pd.to_numeric)
    data['high']=data['high'].apply(pd.to_numeric)
    data['low']=data['low'].apply(pd.to_numeric)
    data['volume']=data['volume'].apply(pd.to_numeric)
    data['y']=data['close'].copy()

    if 'date_close' in data.columns:
        data=data.reset_index(drop=True)
        data['date_close']=data['date_close'].apply(pd.Timestamp)

        data=data.set_index('date_close')
        
    if 'date_open' in data.columns:
        data=data.drop(['date_open'],axis=1)

    for i ,t_func  in enumerate(time_funcs):
        # print (i,t_func)
        f_name=t_func.__repr__().split(' ')[1]

        data[f'feature_{f_name}']=t_func(data.index)


    data['feature_MA_20'] = data['close'].rolling(window=20).mean()
    data['feature_MA_50'] = data['close'].rolling(window=50).mean()
    data['feature_MA_200'] = data['close'].rolling(window=200).mean()
    
    for r in returns:
        data[f'feature_log_return_{r}'] = np.log(1 + data.close.pct_change(r)+1e-15)
        data[f'feature_log_volume_{r}'] = np.log(1 + data.volume.pct_change(r)+1e-15)

    data = data.replace((np.inf, -np.inf,np.nan), 0)
    # data=data-data.min(0)/(data.max(0)-data.min(0))
    data['ds']=data.index.values
    data['ds']=data['ds'].apply(pd.Timestamp)
    return data

def stack_arrays(data,name=None,n_samples=24,prediction_window=4,feature_id=1):
    n_samples=n_samples-1
    train_tensor=[]
    train_targets=[]
    i=0
    data_indexs=data.index
    max_id=len(data_indexs)-(n_samples+prediction_window)-1

    train_list=[]
    val_list=[]
    n_features=len(data.filter(like='feature').columns)

    while i<max_id:
        train_dict={}   
        val_dict={}

        start_id=data_indexs[i]
        end_id=data_indexs[i+n_samples]
        pred_id=data_indexs[i+(n_samples+prediction_window)]

        train_dict['start']=start_id
        val_dict['start']=start_id
        
        train_slice=data[start_id:end_id]['close'].values
        pred_slice=data[start_id:pred_id]['close'].values

        train_dict['target']=train_slice
        val_dict['target']=pred_slice
        n_other_features=len(data.filter(like='feature').columns)
        train_features=data[start_id:end_id].filter(like='feature').values
        pred_features=data[start_id:pred_id].filter(like='feature').values


        train_dict['feat_static_real']=train_features.reshape(train_slice.shape[0],1,n_other_features)
        val_dict['feat_static_real']=pred_features.reshape(pred_slice.shape[0],1,n_other_features)
        
        train_dict['feat_static_cat']=[feature_id]
        val_dict['feat_static_cat']=[feature_id]

        # val_dict['index']=i
        # train_dict['index']=i


        if name:
            train_dict['item_id']=f'{name}_T{i}'
            val_dict['item_id']=f'{name}_T{i}'
        
        train_list.append(train_dict)
        val_list.append(val_dict)        

        i+=1

    return train_list,val_list

def build_market_image(target_pair='ETH/USDT',time_frame='1h',axis=1,dir='data'):

    files=glob.glob(f'{dir}/**{time_frame}.pkl',recursive=True)
    # print(files)
    big_data=[]
    
    for i,file in enumerate(files):
        pair=file.split('-')[1]
        data=pd.read_pickle(file)
        
        data=preprocess_data(data,time_frame=time_frame)
        # display(data.head(1))
        if axis==1:
            if target_pair.replace('/','') != pair:
                feauture_data=data
                feauture_data.columns=[f'{c}_{pair}' for c in feauture_data.columns]
            else :
                feauture_data=data
        else:
            data['unique_id']=symbol_map[pair.replace('/','')]
            data['symbol']=pair
            feauture_data=data
                
        big_data.append(feauture_data)
        # data=data.to_pickle(file)
    big_data=pd.concat(big_data,axis=axis)
    return big_data

def sharpe_reward(history):
    current_position=history['position',-1]
    portfolio_return=history["portfolio_valuation"]
    portfolio_pct_change=pd.Series(np.array(portfolio_return)).pct_change().fillna(0)
    reward = (portfolio_pct_change.mean() / portfolio_pct_change.std())
    reward = 0 if np.isnan(reward) else reward
    return float(reward)



def prep_forecasts(df:pd.DataFrame,model):
    forecast_array=[]
    # print(self.df.columns)

    model.dataset, model.uids, model.last_dates, model.ds=model._prepare_fit(df[['ds','unique_id','y']],
                                                                                        static_df=None, 
                                                                                        sort_df=None,
                                                                                        predict_only=False,
                                                                                        id_col='unique_id', 
                                                                                        time_col='ds', 
                                                                                        target_col='y')
    forecasts=model.predict_insample()
    forecasts_series=forecasts.groupby('cutoff').apply(lambda x: x.select_dtypes(np.number).values.flatten())
    new_df=df[df['ds'].isin([c for c in forecasts_series.index])]
    forecasts_series=forecasts_series[new_df.index]
    forecast_array=[c for c in forecasts_series]
    return forecast_array,new_df

