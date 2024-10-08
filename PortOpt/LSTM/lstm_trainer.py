from numpy import float32, ndarray
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
import numpy as np

class LSTMTrainer():
    def __init__(self, model:nn.Module,train_dataloader:DataLoader, test_dataloader:DataLoader, config_dict:dict, train_scaler_dict:dict,test_scaler_dict:dict,report_freq:int,device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config_dict = config_dict
        self.train_scaler_dict = train_scaler_dict
        self.test_scaler_dict = test_scaler_dict
        self.report_freq = report_freq
        self.device = device
        self.epochs:int = config_dict['epochs']
        self.optim = AdamW(params=self.model.parameters(),lr=config_dict['lr'])
        self.loss = self.MSPELoss

    def MSPELoss(self,pred:torch.Tensor, target:torch.Tensor):
        '''
        Mean squared percentage erorr
        '''
        eps=1e-8
        PE = (target-pred)/(target+eps)
        SPE = PE**2
        MSPE = SPE.mean()
        return MSPE
    
    def inverse_standard_scaling(self,input:torch.Tensor,std:ndarray,mean:ndarray,col_idx):
        std = torch.tensor(std)[col_idx]
        mean = torch.tensor(mean)[col_idx]
        x = std * input + mean
        return x

    def _train_loop(self):
        self.model.train(True)
        batch_losses = list()
        for i, (X, y, tickers) in enumerate(self.train_dataloader): 
            running_loss = 0 #used to calculate mean loss 
            self.optim.zero_grad()

            if self.device == torch.device('mps'): #mps can't handle float64
                X = X.to(dtype=torch.float32)
                y = y.to(dtype=torch.float32)
            X:torch.Tensor = X.to(device=self.device)
            y:torch.Tensor = y.to(device=self.device)
            y = y.squeeze(2)[:,-12:]
            pred:torch.Tensor = self.model(X)
 
            #apply scaling based on ticker - uses index of predictions to find corresponding ticker
            descaled_preds_concat_list = list()
            for i in range(len(tickers)):
                scaler:StandardScaler = self.train_scaler_dict[tickers[i]]['scaler']
                current_pred = pred.select(0,i)
                scaler_std = scaler.scale_
                scaler_mean = scaler.mean_
                log_return_idx = self.train_dataloader.dataset.log_return_idx
                descaled_pred = self.inverse_standard_scaling(current_pred,scaler_std,scaler_mean,log_return_idx)
                descaled_preds_concat_list.append(descaled_pred)
            joined_descaled_preds = torch.stack(descaled_preds_concat_list)
            
            #calculate loss
            loss = self.MSPELoss(joined_descaled_preds,y) #mean squared percentage error of batch
            running_loss += loss.item()
            batch_losses.append(loss.item())
            #compute gradients and update params
            loss.backward()
            self.optim.step()

            if i%self.report_freq == 0:
                mean_loss = running_loss/self.report_freq
                print(f"Mean train loss: {mean_loss}")
        return batch_losses #list of all batch losses
        
    def _test_loop(self):
        self.model.eval()
        batch_losses = list()
        with torch.no_grad():
            for X, y, tickers in self.test_dataloader:
                running_loss = 0

                if self.device == torch.device('mps'): #mps can't handle float64
                    X = X.to(dtype=torch.float32)
                    y = y.to(dtype=torch.float32)

                X:torch.Tensor = X.to(device=self.device)
                y:torch.Tensor = y.to(device=self.device)
                y = y.squeeze(2)[:,-12:]

                preds = self.model(X)
                
                #apply scaling based on ticker - uses index of predictions to find corresponding ticker
                descaled_preds_concat_list = list()
                for i in range(len(tickers)):
                    scaler:StandardScaler = self.test_scaler_dict[tickers[i]]['scaler']
                    current_pred = preds.select(0,i)
                    scaler_std = scaler.scale_
                    scaler_mean = scaler.mean_
                    log_return_idx = self.train_dataloader.dataset.log_return_idx
                    descaled_pred = self.inverse_standard_scaling(current_pred,scaler_std,scaler_mean,log_return_idx)
                    descaled_preds_concat_list.append(descaled_pred)
                joined_descaled_preds = torch.stack(descaled_preds_concat_list)
                loss = self.MSPELoss(joined_descaled_preds,y)
                running_loss += loss.item()
                batch_losses.append(loss.item())

                if i%self.report_freq == 0:
                    mean_loss = running_loss/self.report_freq
                    print(f"Mean test loss: {mean_loss}")
        return batch_losses

    def full_epoch_loop(self):
        train_batch_loss_list = []
        test_batch_loss_list = []
        for epoch in range(self.epochs):
            print(f'---------- Full epoch: {epoch +1} ----------')
            train_batch_losses = self._train_loop()
            train_batch_loss_list.append(train_batch_losses)
            test_batch_losses = self._test_loop()
            test_batch_loss_list.append(test_batch_losses)
        train_batch_loss_array = np.array(train_batch_loss_list)
        test_batch_loss_array = np.array(train_batch_loss_list)

        return train_batch_loss_array, test_batch_loss_array
    
    def train_epoch_loop(self):
        batch_loss_list = []
        for epoch in range(self.epochs):
            print(f'---------- Train epoch: {epoch +1} ----------')
            batch_losses = self._train_loop()
            batch_loss_list.append(batch_losses)
        batch_loss_array = np.array(batch_loss_list)

        return batch_loss_array
    
    def test_epoch_loop(self):
        batch_loss_list = []
        for epoch in range(self.epochs):
            print(f'---------- Test epoch: {epoch +1} ----------')
            batch_losses = self._train_loop()
            batch_loss_list.append(batch_losses)
        batch_loss_array = np.array(batch_loss_list)

        return batch_loss_array


            

