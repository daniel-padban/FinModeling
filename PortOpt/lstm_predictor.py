import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_size:int,hidden_size:int,output_size:int,n_layers:int=3,batch_first:bool=True):
        super().__init__()

        self.mainLSTM = nn.Sequential(
            nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=n_layers,batch_first=batch_first),
            nn.SiLU(),
        )
        self.fco = nn.Linear(in_features=hidden_size,out_features=output_size)

    def forward(self,x):
        x = self.mainLSTM()
        output = self.fco(x[:,-1,:]) #x[:,-1,:] = [batches,last_step,hidden_size]
        return output
