import yfinance
from torch.utils.data import Dataset
import torch
import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

class ReturnDataset(Dataset):
    def __init__(self, seq_len:int,prediction_len:int,ticker_list:list,start_date:datetime.date,end_date:datetime.date):
        super().__init__()
        self.n_shift = prediction_len
        self.seq_len = seq_len

        self.scaler_dict = dict()
        self.ticker_indices = list()
        self.concat_feat_list = list()
        self.concat_target_list = list()
        for asset in ticker_list:
            data = yfinance.Ticker(asset).history(start=start_date,end=end_date,auto_adjust=True,actions=False,repair=True)
            processed_feats, targets = self.process_data(data)
            scaled_feats, scaler = self.scale_data(processed_feats)
            sequenced_feats, sequenced_targets  = self.sequence_data(scaled_feats,targets.values)
            self.scaler_dict[asset] = {} #init dictionary for each ticker
            self.scaler_dict[asset]['scaler'] = scaler
            self.concat_feat_list.append(sequenced_feats)
            self.concat_target_list.append(sequenced_targets)
            self.ticker_indices.extend([asset]*sequenced_feats.shape[0])
        self.joined_feats = np.concat(self.concat_feat_list)
        self.joined_targets = np.concat(self.concat_target_list)
        shuffled_feats, shuffled_targets, self.shuffled_tickers = self.shuffle_sequences()
        self.feat_tensor = torch.tensor(shuffled_feats)
        self.target_tensor = torch.tensor(shuffled_targets)  
        self.num_feats = shuffled_feats.shape[2] #get third dim, d3 = num_feats, [num_seqs, seq_len, num_feats]
        
    def process_data(self,data):
        feat_df = pd.DataFrame(data['Close']) #date is index
        feat_df['log_return'] = np.log(feat_df['Close']/feat_df['Close'].shift(1)) # Log return of asset compared to previous day
        feat_df['target_log_return'] = feat_df['log_return']

        #time features
        feat_df.index = pd.to_datetime(feat_df.index)
        feat_df['day'] = feat_df.index.day #0-30 len=31
        feat_df['dayofweek'] = feat_df.index.dayofweek #0-6 len=7
        feat_df['month'] = feat_df.index.month #0-11 len=12

        # sine transformations of time features
        feat_df['sine_day'] = np.sin(2*np.pi*feat_df['day']/31)
        feat_df['sine_dayofweek'] = np.sin(2*np.pi*feat_df['dayofweek']/7)
        feat_df['sine_month'] = np.sin(2*np.pi*feat_df['month']/12)

        feat_df.drop(columns=['day','dayofweek','month'],inplace=True)
        feat_df.dropna(inplace=True)
        feat_df.dropna(inplace=True)

        target_df = pd.DataFrame(feat_df['target_log_return'])
        shifted_targets = target_df.shift(-self.n_shift)
        shifted_targets.dropna(inplace=True)
        feat_df = feat_df[:-self.n_shift] #drop last n rows, because of shifting
        self.log_return_idx = feat_df.columns.get_loc('log_return')

        return feat_df, shifted_targets

    def scale_data(self,data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    def sequence_data(self,feats:np.ndarray,targets:np.ndarray):
        feat_tensor = torch.tensor(feats) #2d
        target_tensor = torch.tensor(targets) #2d
        num_feats = feat_tensor.shape[1]

        if feat_tensor.shape[0] != target_tensor.shape[0]:
            raise ValueError(f'Feat length does not match target length:\n Feat length:{feat_tensor.shape[0]} \n Target length:{target_tensor.shape[0]}')
        
        num_seqs = feat_tensor.shape[0]//self.seq_len
        required_len = (num_seqs * self.seq_len)
        while required_len > feat_tensor.shape[0]: # handle cases where feat_tensor.shape[o] is less than the required len
            num_seqs -= 1 # reduce number of seqs needed
            required_len = (num_seqs * self.seq_len) + self.n_shift 

        rows_to_drop = feat_tensor.shape[0] - required_len
        feat_tensor = feat_tensor[:-rows_to_drop,:]
        target_tensor = target_tensor[:-rows_to_drop,:]
        
        sequenced_feat_tensor = feat_tensor.reshape(num_seqs,self.seq_len,num_feats)
        sequenced_target_tensor = target_tensor.reshape(num_seqs,self.seq_len,1)

        return sequenced_feat_tensor, sequenced_target_tensor #shape: [num_seqs, seq_len, num_feats]
    
    def shuffle_sequences(self):
        shuffled_indices = np.random.permutation(self.joined_feats.shape[0]) # generate random shuffle of indices with the sam lentgh as feats/targets
        shuffled_feats = self.joined_feats[shuffled_indices]
        shuffled_targets = self.joined_targets[shuffled_indices]
        shuffled_tickers = [self.ticker_indices[i] for i in shuffled_indices]

        return shuffled_feats, shuffled_targets, shuffled_tickers

    def __len__(self):
        return self.feat_tensor.shape[0]

    def __getitem__(self, idx):
        X = self.feat_tensor.select(0,idx)
        y = self.target_tensor.select(0,idx)
        ticker = self.shuffled_tickers[idx]
        return X, y, ticker

if __name__ == '__main__':
    tickers = ['AAPL','MSFT']
    start = datetime.date(2022,1,1)
    end = datetime.date(2024,1,1)
    dataset = ReturnDataset(60,12,tickers,start,end)