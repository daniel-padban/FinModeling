import torch.nn as nn
import torch

class PDNN(nn.Module):
    def __init__(self,in_feats:int, h1:int, h2:int, h3:int,s1:int,out:int=2):
        super().__init__()
        self.main_net = nn.Sequential([
            # layer 1
            nn.Linear(in_features=in_feats,out_features=h1),
            nn.SiLU(),
            nn.BatchNorm1d(h1,),
            # layer 2
            nn.Linear(h1,h2,),
            nn.SiLU(),
            nn.BatchNorm1d(h2,),
            # layer 3
            nn.Linear(h2,h3,),
            nn.SiLU(),
            nn.BatchNorm1d(h3,),
        ])
        self.skip = nn.Sequential([
            nn.Linear(in_feats,s1),
            nn.SiLU(),
        ])
        self.fc4 = nn.Linear((h3+s1),out)

    def forward(self, x):
        x = x #main_net input
        skip_x = x #skip connection input

        x = self.main_net(x)
        skip_x = self.skip(skip_x)
        joined_x = torch.cat(x,skip_x,1) #along feat dim (1)
        
        joined_x = nn.functional.silu(joined_x)
        output = self.fc4(joined_x)
        return output
