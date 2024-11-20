
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from .mappings import *
from finta import TA
from tqdm import tqdm
import datetime
from gluonts.time_feature import time_features_from_frequency_str
from .ta_mapping import get_ta_funcs
from IPython.display import display

def add_indicator(func_name,data):
    try:
        func=getattr(TA,func_name)
        indicator=func(data)
        try:
            col_name=f"feature_{indicator.name}".strip('.').replace(' ','_')
            data[col_name]=indicator.values
        except:
            cols=[f'feature_{func_name}_{n}'.strip('.').replace(' ','_') for n in indicator.columns.to_list()]
            data[cols]=indicator.values
    
    except Exception or FutureWarning as e:
        # print(f'{func_name} Didnt work')
        raise Exception
        
    return data

def add_ta_indicators(data,indicators=['RSI','MACD','STOCH',"BBANDS"],verbose=1):
    ta_functions=get_ta_funcs()
    if verbose:
        ta_functions=tqdm(ta_functions)

    for func_name in ta_functions:
        if func_name in indicators:
            try:
                data=add_indicator(func_name,data)

            except:
                print(f'{func_name} Didnt work')
        
    return data

def add_time_funcs(data,time_frame):
    time_funcs=time_features_from_frequency_str(time_frame)

    for i ,t_func  in enumerate(time_funcs):
        # print (i,t_func)
        f_name=t_func.__repr__().split(' ')[1]

        data[f'feature_{f_name}']=t_func(data.index)

    return data
    
def preprocess_data(data,indicators=['RSI','MACD','STOCH',"BBANDS"],time_frame='1h'):

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
    data=data.sort_index()
    data=add_time_funcs(data,time_frame)
    data=add_ta_indicators(data,indicators=indicators)
    
    data = data.replace((np.inf, -np.inf,np.nan), 0)

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

def build_market_image(target_pair='ETH/USDT',time_frame='1h',axis=1,verbose=1,only_target=False,indicators=['RSI','MACD','STOCH',"BBANDS"],data_dir='data'):

    files=glob.glob(f'{data_dir}/**{time_frame}.pkl',recursive=True)

    if only_target:
        files=[f for f in files if target_pair.replace('/','') in f]
    print(files)
    big_data=[]
    if verbose:
        files=tqdm(enumerate(files))
    else:
        files=enumerate(files)
    for i,file in files:
        pair=file.split('-')[1]
        data=pd.read_pickle(file)
        
        data=preprocess_data(data,time_frame=time_frame,indicators=indicators)

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
    if 'symbol_name'in big_data.columns:
        big_data=big_data.drop('symbol_name',axis=1)
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

def simulate_forecasts(model,df,refit_scalers=False):
    model.dataset, model.uids, model.last_dates, model.ds = model._prepare_fit(
                df=df,
                static_df=None,
                sort_df=model.sort_df,
                predict_only=refit_scalers,
                id_col=model.id_col,
                time_col=model.time_col,
                target_col=model.target_col,
            )
    
    pred_df= model.predict_insample(step_size=1)
    pred_cols=pred_df.filter(like='Auto').columns
    pred_df['mean_pred']=pred_df[pred_cols].mean(axis=1)
    return pred_df

def generate_forecasts(model,df):
    
    pred_df= model.predict(df)
    pred_df=pred_df.reset_index()
    pred_cols=pred_df.filter(like='Auto').columns
    pred_df['mean_pred']=pred_df[pred_cols].mean(axis=1)
    pred_df['cutoff']=df['ds'][-1]
    return pred_df

def prepare_forecast_data(model,data,time_frame='1h',real=False,symbol=None):
    if real:
        pred_df=generate_forecasts(model,data)
    else:
        pred_df= simulate_forecasts(model,data)

    if symbol!=None:
        pred_df=pred_df[pred_df['symbol'].str.contains(symbol)]

    horizon=model.h
    flattened_preds=flatten_pred_df(data,pred_df,horizon=horizon,time_frame=time_frame)

    if not real:
        flattened_preds=flattened_preds.dropna(subset=['close'],axis=0)
    else: 
        flattened_preds['close']=flattened_preds['close'].ffill()

    return flattened_preds,pred_df

def flatten_preds(idx,cut_data,horizon=4):
    t_off_pred,symb=idx
    drop_cols=['cutoff','ds','y','symbol']

    t_cut=cut_data.T
    t_cut.columns=[f'H{i}' for i in range(horizon)]
    

    for col in drop_cols:
        if col in t_cut.index:
            t_cut=t_cut.drop(col)

    flat_cols=[f'feature_{model}_{horizon}' for model in t_cut.index for horizon in t_cut.columns]
    pred_values=t_cut.values.flatten()

    flat_df=pd.DataFrame([pred_values],columns=flat_cols)
    id_cols=['ds','symbol']

    flat_df[id_cols]=t_off_pred,symb
    flat_df=  flat_df[id_cols+flat_cols]
    return flat_df

def flatten_pred_df(data,pred_df,horizon=4,time_frame='1h'):
    flattened_preds=pd.concat([flatten_preds(idx,cut_data,horizon=horizon) for idx,cut_data in pred_df.groupby(['cutoff','symbol'])])
    flattened_preds.index=[c for c in flattened_preds['ds'].values]

    flattened_preds=add_time_funcs(flattened_preds,time_frame)
    flattened_preds['ds']=flattened_preds['ds'].dt.round('H')
    data['ds']=data['ds'].dt.round('H')
    flattened_preds=pd.merge(flattened_preds,data[['ds','close']],on='ds',how='left')
    # display(data[['ds','close']])
    # display(flattened_preds[['ds','close','feature_AutoTFT_H0']])
    front=['ds','close','symbol']
    back=[col for col in flattened_preds.columns if col not in front]
    flattened_preds=flattened_preds[front+back]

    return flattened_preds

def plot_insample_forecasts(data):
    fig,axes=plt.subplots(figsize=(10, 5))
    for symb,cut in data.groupby('symbol'):
        fig,axes=plt.subplot(1,1,figsize=(10, 5))
        plt.plot(cut['ds'], cut['y'], label='True')
        for model in cut.filter(like='Auto').columns:
            plt.plot(cut['ds'], cut[model], label=f'{model} Forecast')
        # plt.axvline(cut['ds'].iloc[-12], color='black', linestyle='--', label='Train-Test Split')
        plt.xlabel('Timestamp [t]')
        plt.ylabel(f'{symb} Price')
        plt.grid()
        plt.legend()

def plot_forecasts(data):
    fig,axes=plt.subplots(figsize=(10, 5))
    for symb,cut in data.groupby('symbol'):
        fig,axes=plt.subplot(1,1,figsize=(10, 5))
        plt.plot(cut['ds'], cut['y'], label='True')
        for model in cut.filter(like='Auto').columns:
            plt.plot(cut['ds'], cut[model], label=f'{model} Forecast')
        # plt.axvline(cut['ds'].iloc[-12], color='black', linestyle='--', label='Train-Test Split')
        plt.xlabel('Timestamp [t]')
        plt.ylabel(f'{symb} Price')
        plt.grid()
        plt.legend()

def train_test_split_data(data,n_days=14,round_factor='H'):
    
    split_date=(pd.Timestamp.now()-pd.Timedelta(days=n_days)).round(round_factor)
    end_date=pd.Timestamp.now().round(round_factor)
    
    print(split_date,end_date)
    data.index=data.index.round(round_factor)
    train_data=data.groupby('symbol').apply(lambda x: x[:split_date]).reset_index(drop=True)
    test_data=data.groupby('symbol').apply(lambda x: x[split_date:]).reset_index(drop=True)
    return train_data,test_data

def make_hidden_dims(n_layers, n_units):
    hidden_dims = []
    for i in range(n_layers):
        hidden_dims.append(n_units)

    return hidden_dims