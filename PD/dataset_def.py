import sklearn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer
import numpy as np

class HMEQ_data(Dataset):
    '''
    data_df overrides csv_path
    '''
    def __init__(self,csv_path:str=None, data_df:pd.DataFrame = None,preprocess:bool=True, tensor:bool=True,):
        super().__init__()
        if data_df is not None:
            data_df = data_df
        elif csv_path is not None:
            data_df = pd.read_csv(csv_path)
        else:
            raise TypeError("Both 'csv_path' and 'data_df' can not be None ")
        
        self.feat_df = data_df.drop(columns='BAD')

        self.target_df = data_df['BAD']
        self.scaler = RobustScaler()
        self.OH_encoder = OneHotEncoder()
        self.knn_imputer = KNNImputer(weights='distance')
        if preprocess:
            feats = self._preprocessing()
        else:
            feats = self.feat_df.values
        if tensor:
            self.feats = torch.tensor(feats)
            self.targets = torch.tensor(self.target_df.values)
        else:
            self.feats = feats
            self.targets = self.target_df.values
        self.num_feats = self.feats.shape[1]

    def _preprocessing(self):
        '''
        Preprocesses self.feat_df inplace
        '''
        feat_df = self.feat_df
        categorical_cols = ['REASON','JOB']
        OH_array = self.OH_encoder.fit_transform(feat_df[categorical_cols]).toarray() #sparse to dense np array
        self.OH_encoder = self.OH_encoder
        feat_df.drop(columns=categorical_cols,inplace=True)
        
        scaled_feats:np.ndarray = self.scaler.fit_transform(feat_df)

        transformed_feats = np.concatenate([scaled_feats,OH_array],axis=1)
        feats = self.knn_imputer.fit_transform(transformed_feats)
        self.imputer = self.knn_imputer
        return feats

    def __len__(self):
        return self.feats.shape[0]
    
    def __getitem__(self, idx) -> tuple:
        X = self.feats.select(0,idx)
        y = self.targets.select(0,idx)
        return X, y


if __name__ == '__main__':
    HMEQ_data(csv_path='PD/pd_data/hmeq.csv')