import torch

class TensorMinMaxScaler():
    def __init__(self, scaling_range:tuple, input_range:tuple=None):    
        #initialize variables for scaling
        if input_range is not None:
            self.input_min = input_range[0]
            self.input_max = input_range[1]
        else:
            self.input_min = None
            self.input_max = None

        self.scale_min = scaling_range[0]
        self.scale_max = scaling_range[1]
    
    def fit(self, input_tensor:torch.Tensor, dim:int=0):
        self.input_min = input_tensor.min(dim)[0]
        self.input_max = input_tensor.max(dim)[0]

        return self

    def transform(self, input_tensor:torch.Tensor):
        numer:torch.Tensor = (input_tensor - self.input_min)*(self.scale_max - self.scale_min)
        denom:torch.Tensor = (self.input_max - self.input_min)
        transformed:torch.Tensor = self.scale_min + (numer/denom)

        return transformed
    
    def fit_transform(self, input_tensor:torch.Tensor, dim:int=0):
        self.input_min = input_tensor.min(dim)[0]
        self.input_max = input_tensor.max(dim)[0]

        numer:torch.Tensor = (input_tensor - self.input_min)*(self.scale_max - self.scale_min)
        denom:torch.Tensor = (self.input_max - self.input_min)
        transformed:torch.Tensor = self.scale_min + (numer/denom)

        return transformed
