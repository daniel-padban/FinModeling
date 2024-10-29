from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np


class BackTester():
    def __init__(self,test_weights,start:date,end:date,tickers:yf.Tickers,comp_weights=None,):
        '''
        
        :param comp_weights: weight of assets to compare against
        '''

        self.asset_data:pd.DataFrame = tickers.history(start=start,end=end,auto_adjust=True)['Close']
        daily_returns = self.asset_data.pct_change(1,).dropna()
        daily_returns.dropna(inplace=True)

        if comp_weights is not None:
            self.comparison_portfolio_returns = daily_returns.dot(comp_weights)
        else:
            self.comparison_portfolio_returns = None
        self.test_portfolio_returns = daily_returns.dot(test_weights)
        print('TESTING')


    def period_return(self,returns:pd.DataFrame,intervals_per_period=252):
        total_comp_return = (1+returns).prod()
        num_intervals = returns.shape[0]
        period_return = (total_comp_return ** (intervals_per_period/num_intervals))-1
        return period_return
    
    def cumulative_returns(self,returns:pd.DataFrame):
        cumulative_returns = (returns+1).cumprod(0)-1
        return cumulative_returns
    
    def period_sharpe(self,returns:pd.DataFrame,risk_free_annual:float,intervals_per_period=252):
        avg_return = returns.mean(0)
        std = returns.std(0)

        daily_risk_free = (1+risk_free_annual)**(1/intervals_per_period)-1
        excess_return = avg_return-daily_risk_free

        sharpe = (excess_return*intervals_per_period-daily_risk_free)/std
        return sharpe

