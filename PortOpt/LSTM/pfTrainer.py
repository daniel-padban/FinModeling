from datetime import date
import torch
import yfinance as yf
import pandas as pd
import numpy as np
from torch.optim import AdamW

from CovMat.optimizer import WeightOptimizer

class pfLSTMTrainer():
    def __init__(self, LSTMModel:torch.nn.Module, allocator:WeightOptimizer, tickers:yf.Tickers, train_range:tuple[date, date], test_range:tuple[date, date], lr:float):
        self.model = LSTMModel
        self.allocator = allocator
        self.asset_weights = self.allocator.alloc_weights
        self.tickers = tickers
        
        train_start = train_range[0]
        train_end = train_range[1]
        
        test_start = test_range[0]
        test_end = test_range[1]

        self.train_data = self._get_data(train_start,train_end) #full price data
        self.test_data = self._get_data(test_start,test_end) #full price data
        col_indices = {idx:col for idx, col in enumerate(self.train_data.columns)}
        self.asset_names =  [col_indices[i] for i in sorted(col_indices.keys())]

        self.optimizer = AdamW(self.model.parameters(),lr=lr,maximize=True)

    def _get_data(self,start,end) -> pd.DataFrame:
        '''
        Returns full price data 
        '''
        data = self.tickers.history(start=start,end=end,auto_adjust=True,repair=True)
        return data
    
    def _process_data(self,data) -> torch.Tensor:
        
        return
        
    def optimize_weights(self, alpha:float, beta:float, gamma:float) -> tuple[np.ndarray, pd.DataFrame]:
        '''
        Allocates weights based on train period. 
        Weights are then used to evaluate fututue portfolio performance and train LSTM.
        '''
        losses, _= self.allocator.optimize_weights(alpha,beta,gamma, returns=self.train_data)
        weights = pd.DataFrame(self.asset_weights.numpy(force=True), index=self.asset_names, columns=['Weights'])
        return losses.numpy(force=True), weights

    def _pf_performance(self,comp_df:pd.DataFrame ):
        cu_test_returns = self.test_data['Close'].pct_change(1).dropna()
        cu_pf_returns = torch.dot(cu_test_returns.values, self.asset_weights)
        cu_comp_returns = comp_df['Close'].pct_change(1).dropna()

        date_index = self.test_data.index
        pf_auc = torch.trapezoid(cu_pf_returns,date_index)
        comp_auc = torch.trapezoid(cu_comp_returns.values,date_index)

        auc_diff = pf_auc-comp_auc

        return auc_diff


    def train_lstm(self,comp_df:pd.DataFrame):
        '''
        A single train cycle
        Needs to be in a loop with epochs
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self._pf_performance(comparison_df=comp_df)
        loss.backward()
        self.optimizer.step()

    
if __name__ == '__main__':
    tickers = yf.Tickers(['GOOGL','MSFT'])
    allocator = WeightOptimizer(1000,1e-3,)
    trainer = pfLSTMTrainer()