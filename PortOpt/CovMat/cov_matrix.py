from datetime import date, timedelta
from socket import PF_SYSTEM
import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import numpy as np
from JSONReader import read_json

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
    
class WeightOptimizer():
    def __init__(self,num_iter:int,lr:float,cov_matrix:torch.Tensor,returns:torch.Tensor,risk_free:float,risk_free_period:timedelta):
        self.num_iter = num_iter
        self.cov_matrix = cov_matrix
        self.returns = returns
        self.risk_free = risk_free
        self.risk_free_period = risk_free_period
        self.num_assets = cov_matrix.shape[0]
        self.weights = nn.Parameter(torch.rand(self.num_assets,requires_grad=True).softmax(0)) #init weights and set to values 0-1
        self.optim = torch.optim.AdamW([self.weights],lr=lr, weight_decay=0.1)

    def _sharpe_loss(self,weights:torch.Tensor,returns:torch.Tensor,cov_matrix,risk_free:float,risk_free_period:timedelta) -> tuple:
        avg_returns = returns.to(dtype=weights.dtype).mean(0)
        cov_matrix = cov_matrix.to(dtype=weights.dtype)
        portfolio_return = torch.dot(weights,avg_returns) #avg daily
        portfolio_var = torch.matmul(weights.T,torch.matmul(cov_matrix,weights))
        portfolio_std = torch.sqrt(portfolio_var)
        
        rr_days = risk_free_period.days
        daily_rr = (1+risk_free) ** (1/rr_days) - 1 
        #acc_rr = (1+daily_rr)**(expected_returns.shape[0]) -1 
        sharpe_ratio = (portfolio_return-daily_rr)/portfolio_std

        return sharpe_ratio, portfolio_return, portfolio_std
    
    def optimize_weights(self):
        sharpes = []
        pf_returns = []
        pf_stds = []
        for i in range(self.num_iter):
            self.optim.zero_grad()
            self.alloc_weights = self.weights.softmax(0)
            sharpe,pf_return,pf_std = self._sharpe_loss(self.alloc_weights,self.returns,self.cov_matrix,risk_free=self.risk_free,risk_free_period=self.risk_free_period)
            neg_sharp = -sharpe
            neg_sharp.backward()
            self.optim.step()
            sharpes.append(sharpe)
            pf_returns.append(pf_return)
            pf_stds.append(pf_std)
        sharpes = torch.vstack(sharpes)
        pf_returns = torch.vstack(pf_returns)
        pf_stds = torch.vstack(pf_stds)
        return sharpes, pf_returns, pf_stds
    
if __name__ == '__main__':
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
