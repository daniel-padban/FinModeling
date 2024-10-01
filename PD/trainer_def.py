import torch
import torcheval
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torcheval.metrics
import wandb
import wandb.plot
import wandb.sdk

class ModelTrainer():
    def __init__(self,run:wandb.sdk.wandb_run.Run,model:torch.nn.Module, train_dataloader:DataLoader, test_dataloader:DataLoader,device,report_freq:int):
        self.run = run
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.device = device
        self.report_freq = report_freq
        self.optimizer = AdamW(params=self.model.parameters(),lr=run.config['lr'])
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _train_loop(self):
        self.model.train()
        for i, (X, y) in enumerate(self.train_dataloader):
            batch_n = i
            X.to(device=self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.loss_fn(pred,y)
            loss.backward()
            self.optimizer.step()
            accuracy = torcheval.metrics.functional.binary_accuracy(pred,y).item()
            precision = torcheval.metrics.functional.binary_precision(pred,y).item()
            recall = torcheval.metrics.functional.binary_recall(pred,y).item()

            if batch_n%self.report_freq == 0:
                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")

                wandb.log({
                    "train_batch":(batch_n*100),
                    "train_BCE":loss.item(),
                    "train_accuracy":accuracy,
                    "train_precision":precision,
                    "train_recall":recall,
                    "train_conf_matrix":wandb.plot.confusion_matrix(None,y,pred),
                })
        
    def _test_loop(self):
        self.model.eval()
        for i, (X, y) in enumerate(self.train_dataloader):
            batch_n = i
            X.to(device=self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred,y)

            accuracy = torcheval.metrics.functional.binary_accuracy(pred,y).item()
            precision = torcheval.metrics.functional.binary_precision(pred,y).item()
            recall = torcheval.metrics.functional.binary_recall(pred,y).item()

            if batch_n%self.report_freq == 0:
                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")

                wandb.log({
                    "test_batch":(batch_n*100),
                    "test_BCE":loss.item(),
                    "test_accuracy":accuracy,
                    "test_precision":precision,
                    "test_recall":recall,
                    "test_conf_matrix":wandb.plot.confusion_matrix(None,y,pred),
                })

    def full_epoch_loop(self):
        for epoch in self.run['epochs']:
            print(f'---------- Epoch: {epoch+1} ----------')
            self._train_loop()
            self._test_loop()

            

            
