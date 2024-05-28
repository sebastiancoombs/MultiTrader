import datetime
import glob
from functools import lru_cache, partial
from pprint import pprint
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")


from tqdm.auto import tqdm
from transformers import (
                            EarlyStoppingCallback,
                            PatchTSTConfig,
                            PatchTSTForPrediction,

                        )

def get_dataset_columns(data,id_columns,timestamp_column = "date_open"):
    output_columns=data.filter(like='close').columns.to_list()
    feature_columns = data.filter(like='feature').columns.to_list()
    feature_columns =[c for c in feature_columns if c not in [timestamp_column,'volume']]
    
    drop_columns = [c for c in  data.columns if c not in feature_columns+output_columns+[timestamp_column]+id_columns
                    ]
    return output_columns,feature_columns,drop_columns

def create_ts_preprocessor(data,
                           timestamp_column = "date_open",
                           id_columns = ['symbol'],


                           ):
    
    output_columns,feature_columns,drop_columns= get_dataset_columns(data,id_columns=id_columns,timestamp_column=timestamp_column)
    
    time_series_preprocessor = TimeSeriesPreprocessor(
                                                timestamp_column=timestamp_column,
                                                id_columns=id_columns,
                                                input_columns=feature_columns,
                                                target_columns=output_columns,
                                                scaling=True,
                                            )
    time_series_preprocessor = time_series_preprocessor.train(data)

    return time_series_preprocessor

def create_ts_dataset(data,
                        timestamp_column = "date_open",
                        id_columns = [
                                        'symbol'
                                    ],
                        forecast_horizon = 12,
                        context_length = 48,
                        ):
    output_columns,feature_columns,drop_columns = get_dataset_columns(data,id_columns)

    ts_dataset = ForecastDFDataset(
                                    data,
                                    id_columns=id_columns,
                                    timestamp_column=timestamp_column,
                                    observable_columns=output_columns,
                                    target_columns=output_columns,
                                    context_length=context_length,
                                    prediction_length=forecast_horizon,
                                )
    return ts_dataset

def ts_train_test_split(data, 
                        id_columns,
                        context_length):
    # get split
    sample_id=data[id_columns[0]].unique()[-1]
    sample=data[data[id_columns[0]]==sample_id]

    num_train = int(len(sample) * 0.7)
    num_test = int(len(sample) * 0.2)
    num_valid = len(sample) - num_train - num_test
    border1s = [
        0,
        num_train - context_length,
        len(sample)- num_test - context_length,
    ]
    border2s = [num_train, num_train + num_valid, len(sample)]

    train_start_index = border1s[0]  # None indicates beginning of dataset
    train_end_index = border2s[0]

    # we shift the start of the evaluation period back by context length so that
    # the first evaluation timestamp is immediately following the training data
    valid_start_index = border1s[1]
    valid_end_index = border2s[1]

    test_start_index = border1s[2]
    test_end_index = border2s[2]

    train_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=train_start_index,
        end_index=train_end_index,
    )

    valid_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=valid_start_index,
        end_index=valid_end_index,
    )
    
    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )

    return train_data,valid_data,test_data

def make_ts_train_test_datasets(
                        data,
                        timestamp_column = "date_open",
                        id_columns = ['symbol'],
                        context_length=48,
                        forecast_horizon=12,   
                        **kwargs
                        ):

    train_data,valid_data,test_data=ts_train_test_split(data,
                                                        context_length=context_length,

                                                        id_columns=id_columns)
    
    ts_prepcessor=create_ts_preprocessor(data,
                        timestamp_column = timestamp_column,
                        id_columns = id_columns ,

                        )
    
    train_dataset=create_ts_dataset(ts_prepcessor.preprocess(train_data),
                        timestamp_column = timestamp_column,
                        id_columns = id_columns ,
                        forecast_horizon = forecast_horizon,
                        context_length = context_length,
                        )
    
    valid_dataset=create_ts_dataset(ts_prepcessor.preprocess(valid_data),
                        timestamp_column = timestamp_column,
                        id_columns = id_columns ,
                        forecast_horizon = forecast_horizon,
                        context_length = context_length,
                        )
    
    test_dataset=create_ts_dataset(ts_prepcessor.preprocess(test_data),
                        timestamp_column = timestamp_column,
                        id_columns = id_columns ,
                        forecast_horizon = forecast_horizon,
                        context_length = context_length,
                        )
    
    return train_dataset,valid_dataset,test_dataset

def build_model(
                num_input_channels,
                context_length=48,
                patch_length=2,
                forecast_horizon=12,                                 
                random_mask_ratio=0.4,
                d_model=128,
                num_attention_heads=16,
                num_hidden_layers=3,
                ffn_dim=256,
                dropout=0.2,
                head_dropout=0.2,
                pooling_type=None,
                channel_attention=False,
                scaling="std",
                loss="mse",
                pre_norm=True,
                norm_type="batchnorm",

                        ):

    
    config = PatchTSTConfig(
                                num_input_channels=num_input_channels,
                                context_length=context_length,
                                patch_length=patch_length,
                                prediction_length=forecast_horizon,
                                random_mask_ratio=random_mask_ratio,
                                d_model=d_model,
                                num_attention_heads=num_attention_heads,
                                num_hidden_layers=num_hidden_layers,
                                ffn_dim=ffn_dim,
                                dropout=dropout,
                                head_dropout=head_dropout,
                                pooling_type=pooling_type,
                                channel_attention=channel_attention,
                                scaling=scaling,
                                loss=loss,
                                pre_norm=pre_norm,
                                norm_type= norm_type,
                            )
    

    model=PatchTSTForPrediction(config)
    return model

def build_model_get_data(
                data,
                timestamp_column = "date_open",
                id_columns = ['symbol'],
                context_length=48,
                patch_length=2,
                forecast_horizon=12,                                 
                random_mask_ratio=0.4,
                d_model=128,
                num_attention_heads=16,
                num_hidden_layers=3,
                ffn_dim=256,
                dropout=0.2,
                head_dropout=0.2,
                pooling_type=None,
                channel_attention=False,
                scaling="std",
                loss="mse",
                pre_norm=True,
                norm_type="batchnorm",
                **kwargs):
    output_columns,feature_columns,drop_columns = get_dataset_columns(data,id_columns=id_columns)


    train_dataset,valid_dataset,test_dataset=make_ts_train_test_datasets(
                                                data,
                                                timestamp_column =  timestamp_column,
                                                id_columns = id_columns,
                                                context_length=context_length,
                                                forecast_horizon=forecast_horizon,   
                                                )
    num_input_channels=train_dataset.n_targets
    
    model=build_model(   
                        num_input_channels=num_input_channels,

                        context_length=context_length,
                        patch_length=patch_length,
                        forecast_horizon=forecast_horizon,
                        random_mask_ratio=random_mask_ratio,
                        d_model=d_model,
                        num_attention_heads=num_attention_heads,
                        num_hidden_layers=num_hidden_layers,
                        ffn_dim=ffn_dim,
                        dropout=dropout,
                        head_dropout=head_dropout,
                        pooling_type=pooling_type,
                        channel_attention=pooling_type,
                        scaling=scaling,
                        loss=loss,
                        pre_norm=pre_norm,
                        norm_type= norm_type,
                        )
    
    train_dataset,valid_dataset,test_dataset=make_ts_train_test_datasets(
                        data,
                        timestamp_column =  timestamp_column,
                        id_columns = id_columns,
                        context_length=context_length,
                        forecast_horizon=forecast_horizon,   
                        )
    
    model_data={
        'model':model,
        'train_dataset':train_dataset,
        'valid_dataset':valid_dataset,
        'test_dataset':test_dataset
    }
    
    return model_data
    
    