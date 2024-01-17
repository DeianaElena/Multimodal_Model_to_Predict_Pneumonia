import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# my imports
from image_model import Image_Model
from image_model_mod_effb4 import Image_Model_effb4
from image_model2 import Image_Model2

from text_model import Text_Model
from metric import Metric
from path_and_parameters import Defining_Parameters


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Fusion_Model(pl.LightningModule):
    def __init__(self, image_input_dim, text_input_dim, learning_rate):
        super(Fusion_Model, self).__init__()
        
        # Final layer to combine the output from both models
        self.fc = nn.Sequential(
            nn.Linear(image_input_dim+text_input_dim, 256),  # if both previous models output 2 classes
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),  
            nn.LogSoftmax(dim=1)
        )

        self.learning_rate = learning_rate
        # weights for the loss function
        weights = torch.FloatTensor([0.07, 0.93])  #full db
        weights = weights.to(device)
        self.criterion = nn.NLLLoss(weight=weights)
        self.metrics = Metric()


        # Containers for predictions and labels
        self.predictions = []
        self.labels = []
        # Create a DataFrame
        self.df2 = pd.DataFrame(columns=['predictions', 'labels'])

    def forward(self, image, text):
        combination = torch.cat((image, text), dim=1)
        x = self.fc(combination)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr = self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        image, text, labels = batch
        preds = self.forward(image, text)
        loss = self.criterion(preds, labels)
        self.log('train_loss', loss)

        # Use training metrics
        metrics = self.metrics.training_metric(preds, labels)
        self.log_dict(metrics)

        #Call calculate_metrics method to update TP, TN, FP, FN in the Metric object
        self.metrics.calculate_metrics(preds, labels)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image, text, labels = batch
        preds = self.forward(image, text)
        loss = self.criterion(preds, labels)
        self.log('val_loss', loss)

        # Use validation metrics
        metrics = self.metrics.validation_metric(preds, labels)
        self.log_dict(metrics)  # log the metrics dictionary here
        
        # Call calculate_metrics method to update TP, TN, FP, FN in the Metric object
        self.metrics.calculate_metrics(preds, labels)      
        return metrics

    def test_step(self, batch, batch_idx):
        image, text, labels = batch
        preds = self.forward(image, text)
        loss = self.criterion(preds, labels)
        self.log('test_loss', loss)

        # Use Metrics for other test metrics
        metrics = self.metrics.test_metric(preds, labels)
        self.log_dict(metrics)  # log the metrics dictionary here
        
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
       
        # ################--------------#######################

        return metrics
    



    



    
