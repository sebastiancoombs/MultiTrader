
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import datasets
from datasets import load_dataset,Dataset,DatasetDict
import datetime
from gluonts.time_feature import time_features_from_frequency_str

symbol_map={'ADABNB': 0,
 'ADABTC': 1,
 'ADAETH': 2,
 'ADAUSDT': 3,
 'BNBETH': 4,
 'BNBUSDT': 5,
 'BTCUSDT': 6,
 'ETHBTC': 7,
 'ETHUSDT': 8,
 'SOLBNB': 9,
 'SOLBTC': 10,
 'SOLETH': 11,
 'SOLUSDT': 12,
 'XRPBNB': 13,
 'XRPBTC': 14,
 'XRPETH': 15,
 'XRPUSDT': 16}

def preprocess_data(data,time_frame='1h'):
    returns=[2,5,10,15,20,25,30]
    time_funcs=time_features_from_frequency_str(time_frame)
    data['close']=data['close'].apply(pd.to_numeric)
    data['open']=data['open'].apply(pd.to_numeric)
    data['high']=data['high'].apply(pd.to_numeric)
    data['low']=data['low'].apply(pd.to_numeric)
    data['volume']=data['volume'].apply(pd.to_numeric)
    data['y']=data['close'].copy()
    data['ds']=data.index.copy()

    for i ,f  in enumerate(time_funcs):
        f_name=f.__repr__().split(' ')[1]

        data[f'feature_{f_name}']=data['ds'].apply(f).copy()
    data=data.drop('ds',axis=1)
    if'date_close' in data.columns:
        data=data.drop('date_close',axis=1)
    data['feature_MA_20'] = data['close'].rolling(window=20).mean()
    data['feature_MA_50'] = data['close'].rolling(window=50).mean()
    data['feature_MA_200'] = data['close'].rolling(window=200).mean()
    
    for r in returns:
        data[f'feature_log_return_{r}'] = np.log(1 + data.close.pct_change(r)+1e-15)
        data[f'feature_log_volume_{r}'] = np.log(1 + data.volume.pct_change(r)+1e-15)

    data = data.replace((np.inf, -np.inf,np.nan), 0)
    data=data-data.min(0)/(data.max(0)-data.min(0))

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

def build_market_image(target_pair='ETH/USDT',time_frame='1h',axis=1):

    files=glob.glob(f'data/**{time_frame}.pkl',recursive=True)
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

def build_dataset(data=None,
                  prediction_window=2,
                    n_samples=24,
                    look_back=30,
                    start_date=datetime.datetime(year= 2024, month= 2, day=1),
                    target_pair='ETH/USDT',
                    time_frame='1h'
                    ):
    if isinstance(data,type(None)):
        data=build_market_image(target_pair=target_pair,
                            time_frame=target_pair)
    
    target_pair=target_pair.replace('/','')
    
    ## get the data splittng dates
    split_date=start_date+datetime.timedelta(days=look_back)
    end_date=split_date+datetime.timedelta(days=look_back)

    train_data=data[start_date:split_date]
    test_data=data[split_date:end_date]
    data=preprocess_data(data
                            )
    
    train_list,val_list=stack_arrays(train_data,
                                     name=target_pair,prediction_window=prediction_window,
                                     n_samples=n_samples,
                                     )
    
    test_list,test_val_list=stack_arrays(test_data,
                                         name=target_pair,prediction_window=prediction_window,
                                         n_samples=n_samples,
                                         )
    crypto=DatasetDict()
    crypto['train']=Dataset.from_list(train_list,)
    crypto['validation']=Dataset.from_list(val_list)
    crypto['test']=Dataset.from_list(test_list)
    crypto['test_validation']=Dataset.from_list(test_val_list)
    return crypto



def build_huggface_data_set(time_frame='1h',
                           
                            prediction_window=2,
                            n_samples=24,
                            look_back=30,
                            start_date=datetime.datetime(year= 2024, month= 2, day=1),
                            ):
    files=glob.glob(f'data/**{time_frame}.pkl',)

    big_train_list=[]
    big_val_list=[]
    big_test_list=[]
    big_test_val_list=[]
    for i,file in enumerate(files):
        pair=file.split('-')[1]
        data=pd.read_pickle(file)
        
        data=preprocess_data(data)
        # display(data.head(1))
   
        ## get the data splittng dates
        split_date=start_date+datetime.timedelta(days=look_back)
        end_date=split_date+datetime.timedelta(days=look_back)

        train_data=data[start_date:split_date]
        test_data=data[split_date:end_date]
        
        train_list,val_list=stack_arrays(train_data,
                                        name=pair,
                                        prediction_window=prediction_window,
                                        n_samples=n_samples,
                                        feature_id=i
                                        )
        
        test_list,test_val_list=stack_arrays(test_data,
                                            name=pair,
                                            prediction_window=prediction_window,
                                            n_samples=n_samples,
                                            feature_id=i
                                            )
        
        [big_train_list.append(x) for x in train_list]
        [big_val_list.append(x) for x in val_list]
        [big_test_list.append(x) for x in test_list]
        [big_test_val_list.append(x) for x in test_val_list]

    crypto=DatasetDict()
    crypto['train']=Dataset.from_list(big_train_list,)
    crypto['validation']=Dataset.from_list(big_val_list)
    crypto['test']=Dataset.from_list(big_test_list)
    crypto['test_validation']=Dataset.from_list(big_test_val_list)
    return crypto


def sharpe_reward(history):
    current_position=history['position',-1]
    portfolio_return=history["portfolio_valuation"]
    portfolio_pct_change=pd.Series(np.array(portfolio_return)).pct_change().fillna(0)
    reward = (portfolio_pct_change.mean() / portfolio_pct_change.std())
    reward = 0 if np.isnan(reward) else reward
    return float(reward)