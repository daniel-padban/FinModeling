from faulthandler import cancel_dump_traceback_later
import torch
import torch.nn as nn
from datetime import timedelta
from tensorScaling import TensorMinMaxScaler

class WeightOptimizer():
    def __init__(self,num_iter:int,lr:float,num_assets:int, risk_free:float,risk_free_period:timedelta):
        self.num_iter = num_iter
        self.risk_free = risk_free
        self.risk_free_period = risk_free_period
        self.weights = nn.Parameter(torch.rand(num_assets,requires_grad=True).softmax(0)) #init weights and set to values 0-1
        self.optim = torch.optim.AdamW([self.weights],lr=lr, weight_decay=0.1,maximize=True)
        
        #data to scale optimization targets
        self.calmar_scaler = TensorMinMaxScaler(scaling_range=(0,1),input_range=(0,15))
        self.omega_scaler = TensorMinMaxScaler(scaling_range=(0,1), input_range=(0,5))
        self.sortino_scaler = TensorMinMaxScaler(scaling_range=(0,1),input_range=(-1, 5))

    def _sortino_loss(self,weights:torch.Tensor,returns:torch.Tensor,risk_free:float,risk_free_period:timedelta) -> torch.Tensor:
        returns = returns.to(dtype=weights.dtype)
        avg_returns = returns.mean(0) #avg daily
        portfolio_daily_return = torch.dot(weights,avg_returns) #total portfolio return - scalar
    
        #target in sortino:
        rf_days = risk_free_period.days
        daily_rf = (1+risk_free) ** (1/rf_days) - 1 

        #semi deviation:
        downside_returns = torch.where(returns<0,returns,torch.tensor(0.0))
        downside_cov = torch.cov(downside_returns.T) #semi standard deviation (downside)
        semi_var = torch.matmul(weights.permute(*torch.arange(weights.ndim - 1, -1, -1)),torch.matmul(downside_cov,weights))
        semi_dev = torch.sqrt(semi_var)

        #sortino
        sortino_ratio = (portfolio_daily_return-daily_rf)/semi_dev
        
        return sortino_ratio
    
# ------------------------------ Fix weights ------------------------------
    def _omega_ratio(self,weights:torch.Tensor,returns:torch.Tensor,risk_free:float,risk_free_period:timedelta) -> torch.Tensor:
        returns_T = returns.permute(1,0)
        weighted_returns = torch.matmul(weights,returns_T)
        #target return
        rf_days = risk_free_period.days
        daily_rf = (1+risk_free) ** (1/rf_days) - 1 

        #gains & losses
        excess_returns = weighted_returns-daily_rf
        cumulative_gains = torch.sum(excess_returns[excess_returns>0]) #profit above threshold
        cumulative_losses = -1*torch.sum(excess_returns[excess_returns<0]) #loss below threshold

        #omega
        omega_ratio = cumulative_gains/cumulative_losses        
        return omega_ratio

    def _calmar_ratio(self, weights:torch.Tensor, returns:torch.Tensor,risk_free:float,risk_free_period:timedelta) -> torch.Tensor:
        returns = returns.to(dtype=weights.dtype)
        avg_returns = returns.mean(0) #avg daily
        portfolio_daily_return = torch.dot(weights,avg_returns) #total portfolio return - scalar
        
        #target in calmar:
        rf_days = risk_free_period.days
        daily_rf = (1+risk_free) ** (1/rf_days) - 1 
        
        #drawdowns:
        maximums = torch.cummax(returns+1, dim=0)[0] # calculate cumulative maximums over time
        drawdowns = 1 - ((returns+1)/maximums) # calculate return to maximum ratio (drawdown) over time
        max_drawdown = drawdowns.max() #max drawdown

        #calmar
        calmar_ratio = (portfolio_daily_return-daily_rf)/max_drawdown
        return calmar_ratio
    
    def _cos_criterion(self,returns:torch.Tensor, weights:torch.Tensor, rf:float,rf_period:timedelta, alpha:float,beta:float,gamma:float) -> tuple: # cos - calmar, omega, sortino
        '''
        
         combines calmar ratio, omega ratio and sortino ratio into a loss function
        Returns tuple (cos, weighted avg return)
        :param alpha: weight of calmar ratio 
        :param beta: weight of omega ratio 
        :param gamma: weight of sortino ratio 
        '''
        returns = returns.to(dtype=weights.dtype)
        calmar = self._calmar_ratio(weights,returns,rf,rf_period)
        omega = self._omega_ratio(weights,returns,rf,rf_period)
        sortino = self._sortino_loss(weights,returns,rf,rf_period)

        scaled_calmar = self.calmar_scaler.transform(calmar)
        scaled_omega = self.omega_scaler.transform(omega)
        scaled_sortino = self.sortino_scaler.transform(sortino)
        
        cos = alpha*scaled_calmar + beta*scaled_omega + gamma*scaled_sortino
        
        avg_returns = returns.mean(0)
        portfolio_daily_return = torch.dot(weights,avg_returns)

        return cos, portfolio_daily_return

    
    def optimize_weights(self,alpha,beta,gamma, returns):

        '''
        :param alpha: weight of calmar ratio in optimization goal
        :param beta: weight of omega ratio in optimization goal
        :param gamma: weight of sortino ratio in optimization goal
        '''
        print('Optimizing weights... ')
        cos_losses = []
        pf_returns = []
        for i in range(self.num_iter):
            self.optim.zero_grad()
            self.alloc_weights = self.weights.softmax(0)
            cos_loss, pf_return = self._cos_criterion(returns,self.weights,self.risk_free,self.risk_free_period,alpha,beta,gamma)
            cos_loss.backward()
            self.optim.step()
            cos_losses.append(cos_loss)
            pf_returns.append(pf_return)
        cos_losses_tensor = torch.vstack(cos_losses)
        pf_daily_returns = torch.vstack(pf_returns)
        return cos_losses_tensor, pf_daily_returns