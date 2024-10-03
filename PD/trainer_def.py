from numpy import float32
import torch
import torcheval
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torcheval.metrics
import wandb
import wandb.plot
import wandb.sdk
import warnings

class ModelTrainer():
    def __init__(self,run:wandb.sdk.wandb_run.Run,model:torch.nn.Module, train_dataloader:DataLoader, test_dataloader:DataLoader,device,report_freq:int=100):
        self.run = run
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.device = device
        self.report_freq = report_freq
        self.optimizer = AdamW(params=self.model.parameters(),lr=run.config['lr'])
        self.class_weights = torch.tensor(4)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

    def _train_loop(self):
        self.model.train()
        for i, (X, y) in enumerate(self.train_dataloader):
            batch_n = i
            if self.device == torch.device('mps'):
               X = X.to(dtype=torch.float32)
               y = y.to(dtype=torch.float32)
            X = X.to(device=self.device)
            y = y.to(device=self.device)
            self.optimizer.zero_grad()
            pred:torch.Tensor = self.model(X)
            pred = pred.squeeze(1)
            
            #update params
            loss = self.loss_fn(pred,y)
            loss.backward()
            self.optimizer.step()
            
            #get probs for metrics
            probs = pred.softmax(0)
            #probs = probs.long()
            y = y.long()
            most_probable = (probs >= 0.5).long() #most probable

            accuracy = torcheval.metrics.BinaryAccuracy()
            precision = torcheval.metrics.BinaryPrecision()
            recall = torcheval.metrics.BinaryRecall()

            accuracy.update(most_probable,y)
            precision.update(most_probable,y)
            recall.update(most_probable,y)

            if batch_n%self.report_freq == 0:
                print(f"Accuracy: {accuracy.compute()}")
                print(f"Precision: {precision.compute()}")
                print(f"Recall: {recall.compute()}")
                wandb.log({
                    "train_batch":(batch_n*self.report_freq),
                    "train_BCE":loss.item(),
                    "train_accuracy":accuracy.compute(),
                    "train_precision":precision.compute(),
                    "train_recall":recall.compute(),
                    "train_conf_matrix":wandb.plot.confusion_matrix(None,y.tolist(),pred.tolist()),
                })
        
    def _test_loop(self):
        self.model.eval()
        for i, (X, y) in enumerate(self.train_dataloader):
            batch_n = i
            if self.device == torch.device('mps'):
               X = X.to(dtype=torch.float32)
               y = y.to(dtype=torch.float32)
            X = X.to(device=self.device)
            y = y.to(device=self.device)
            pred = self.model(X)
            pred = pred.squeeze(1)
            loss = self.loss_fn(pred,y)

            probs = torch.softmax(pred,0)
            probs = probs.long()
            y = y.long()
            most_probable = (probs >= 0.5).long() #most probable
            accuracy = torcheval.metrics.BinaryAccuracy()
            precision = torcheval.metrics.BinaryPrecision()
            recall = torcheval.metrics.BinaryRecall()

            accuracy.update(most_probable,y)
            precision.update(most_probable,y)
            recall.update(most_probable,y)

            if batch_n%self.report_freq == 0:
                print(f"Accuracy: {accuracy.compute()}")
                print(f"Precision: {precision.compute()}")
                print(f"Recall: {recall.compute()}")

                wandb.log({
                    "test_batch":(batch_n*self.report_freq),
                    "test_BCE":loss.item(),
                    "test_accuracy":accuracy.compute(),
                    "test_precision":precision.compute(),
                    "test_recall":recall.compute(),
                    "test_conf_matrix":wandb.plot.confusion_matrix(None,y.tolist(),probs.tolist(),class_names=['True','False']),
                })

    def full_epoch_loop(self):
        for epoch in range(self.run.config['epochs']):
            print(f'---------- Epoch: {epoch+1} ----------')
            self._train_loop()
            self._test_loop()

            

            
