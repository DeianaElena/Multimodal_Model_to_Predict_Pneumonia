import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
# my imports
from metric import Metric
from path_and_parameters import Defining_Parameters

from losses import FocalBCE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Image_Model_bce_focal(pl.LightningModule):
    def __init__(self, input_dim, learning_rate):
        super().__init__()
        #replacing the final layer of the image model by a small set of Sequential layers
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),  
            #remove this for cross-entropy so it directly return the raw logits from the last layer
            nn.Sigmoid()  
        )

        self.learning_rate = learning_rate
        self.weights = [0.07, 0.93]  #full db
        # weights = torch.FloatTensor([0.2, 0.8])  #smaller db 
        self.criterion = FocalBCE()
        # self.val_criterion = nn.BCELoss() 
        self.metrics = Metric()
        
    def forward(self, image):
        if len(image.shape) > 2:
            image = image.view(image.size(0), -1)
        x = self.fc(image)
        # print(x.size())  #torch.size[32,2]
        # raise
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr = self.learning_rate)
        return optimizer
    
    def training_step(self, batch):
        image, labels = batch
        preds = self.forward(image)

        weights = torch.zeros_like(labels)
        weights[labels == 0] = float(self.weights[0])
        weights[labels == 1] = float(self.weights[1])
        # print(preds)
        # print(labels)
        loss = self.criterion(probs=preds, target=labels, weights=weights, gamma=1)
        # print(loss)
        self.log('train_loss', loss)

        # Use training metrics
        metrics = self.metrics.training_metric(preds, labels)
        self.log_dict(metrics)
        return loss

    # def validation_step(self, batch, batch_idx, dataloader_idx=0):
    def validation_step(self, batch, batch_idx):
        image, labels = batch
        preds = self.forward(image)

        weights = torch.zeros_like(labels)
        weights[labels == 0] = float(self.weights[0])
        weights[labels == 1] = float(self.weights[1])

        loss = self.criterion(probs=preds, target=labels, weights=weights, gamma= 2)
        # loss = F.binary_cross_entropy(preds.view(-1), labels.view(-1), weights.view(-1))
        self.log('val_loss', loss)

        # Use validation metrics
        metrics = self.metrics.validation_metric(preds, labels)
        self.log_dict(metrics)  # log the metrics dictionary here
        return metrics

    
    def test_step(self, batch, batch_idx):
        image, labels = batch
        preds = self.forward(image)
        # loss = self.criterion(preds, labels)

        weights = torch.zeros_like(labels)
        weights[labels == 0] = float(self.weights[0])
        weights[labels == 1] = float(self.weights[1])
        loss = self.criterion(probs=preds, target=labels, weights=weights, gamma=2)
        self.log('test_loss', loss)

        # Use Metrics for other test metrics
        metrics = self.metrics.test_metric(preds, labels)
        self.log_dict(metrics)  # log the metrics dictionary here
        return metrics

  
