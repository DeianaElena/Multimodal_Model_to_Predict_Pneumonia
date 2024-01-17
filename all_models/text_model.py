################ automate metric code  ###################

import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torchmetrics import Accuracy, Precision, Recall, F1Score
# my imports
from metric import Metric
from path_and_parameters import Defining_Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Text_Model(pl.LightningModule):
    def __init__(self, input_dim, learning_rate):
        super(Text_Model, self).__init__()
        # self.fc = nn.Linear(input_dim, 2)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),  
            nn.LogSoftmax(dim=1)
        )

        self.learning_rate = learning_rate
        weights = torch.FloatTensor([0.07, 0.93])  #full db
        # weights = torch.FloatTensor([0.2, 0.8])  #smaller db
        weights = weights.to(device)
        self.criterion = nn.NLLLoss(weight=weights)
        self.metrics = Metric()


        # Containers for predictions and labels
        self.predictions = []
        self.labels = []
        # Create a DataFrame
        self.df2 = pd.DataFrame(columns=['predictions', 'labels'])


    def forward(self, text):
        if len(text.shape) > 2:
            text = text.view(text.size(0), -1)
        x = self.fc(text)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr = self.learning_rate)
        return optimizer

    def training_step(self, batch):
        text, labels = batch
        preds = self.forward(text)
        loss = self.criterion(preds, labels)
        self.log('train_loss', loss, on_step= False, on_epoch= True, prog_bar=True, logger=True)
        
        # Use training metrics
        metrics = self.metrics.training_metric(preds, labels)
        self.log_dict(metrics, on_step= True, on_epoch= False, prog_bar=True, logger=True)
        return loss

    # def validation_step(self, batch, batch_idx, dataloader_idx=0):
    def validation_step(self, batch, batch_idx):
        text, labels = batch
        preds = self.forward(text)
        loss = self.criterion(preds, labels)
        self.log('val_loss', loss, on_step= False, on_epoch= True, prog_bar=True, logger=True)
        
        # Use validation metrics
        metrics = self.metrics.validation_metric(preds, labels)
        #on_step=True to log the metrics for each batch. Set on_epoch=True to log the epoch metrics
        self.log_dict(metrics, on_step= False, on_epoch= True, prog_bar=True, logger=True)  # log the metrics dictionary here
        return metrics

    
    def test_step(self, batch, batch_idx):
        text, labels = batch
        preds = self.forward(text)
        loss = self.criterion(preds, labels)
        self.log('test_loss', loss, on_step= False, on_epoch= True, prog_bar=True, logger=True)
        
        # Use Metrics for other test metrics
        metrics = self.metrics.test_metric(preds, labels)
        self.log_dict(metrics, on_step= False, on_epoch= True, prog_bar=True, logger=True)  # log the metrics dictionary here
        
        ################--------------#######################
        #Option 2 - w
        # ############### Store the predictions and labels
        # # Get the class predictions
        _, pred_classes = torch.max(preds, dim=1)

        # Convert to Python lists (or you can work with them as tensors)
        pred_classes = pred_classes.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()

        # Append to the DataFrame
        for p, l in zip(pred_classes, labels):
            # self.df2 = self.df2.append({'predictions': p, 'labels': l}, ignore_index=True)
            # df3 = pd.concat([self.df2, pd.DataFrame{'predictions': p, 'labels': l}], ignore_index=True)
            rowitem = {
            "predictions": p,
            "labels": l 
            }

            self.df2.loc[len(self.df2)] = rowitem
        return metrics
    

