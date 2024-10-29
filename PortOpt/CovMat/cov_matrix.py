from audioop import avg
from datetime import date, timedelta
from socket import PF_SYSTEM
from turtle import down
import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import numpy as np
from CovMat.JSONReader import read_json

class MPTCovMat():
    def __init__(self, tickers:yf.Tickers,start:date,end:date):
        self.ticker_list = tickers
        self.price_df:pd.DataFrame = tickers.history(start=start,end=end,auto_adjust=True)
        self.log_return_df:pd.DataFrame = self.calc_log_return()
        self.cov_matrix:pd.DataFrame = self.calc_cov_matrix()
        self.col_indices = {idx:col for idx, col in enumerate(self.log_return_df.columns)}

    def calc_log_return(self):
        log_return_df:pd.DataFrame = np.log(self.price_df['Close']/self.price_df['Close'].shift(1))
        log_return_df.dropna(inplace=True)
        return log_return_df
    
    def calc_cov_matrix(self):
        cov_matrix = self.log_return_df.cov(2,1)
        return cov_matrix
    
if __name__ == '__main__':
    from optimizer import WeightOptimizer
    ticker_list =read_json('omxs30.json')
    tickers = yf.Tickers(ticker_list,)
    start = date(2022,1,1)
    end = date(2023,1,1)
    cov_matrix_obj = MPTCovMat(tickers=tickers,start=start,end=end)
    log_return_df = cov_matrix_obj.log_return_df
    cov_matrix = cov_matrix_obj.cov_matrix_tensor
    col_indices = cov_matrix_obj.col_indices
    risk_free_period = timedelta(days=120)
    w_optimizer = WeightOptimizer(500,1e-3,torch.tensor(cov_matrix.values),torch.tensor(log_return_df.values),risk_free=0.0027,risk_free_period=risk_free_period)
    sharpes,returns,stds = w_optimizer.optimize_weights()
    col_names = [col_indices[i] for i in sorted(col_indices.keys())]
    weights_df = pd.DataFrame(w_optimizer.alloc_weights.numpy(force=True), index=col_names,columns=['raw_weights'])

    weights_df['Weights %'] = weights_df['raw_weights']*100
    weights_df.index.name = 'Ticker'
    print('Cov_matrix.py tested')
