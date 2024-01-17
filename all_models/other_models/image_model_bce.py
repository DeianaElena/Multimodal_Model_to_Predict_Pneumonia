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
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
# my imports
from metric import Metric
from path_and_parameters import Defining_Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Image_Model_bce(pl.LightningModule):
    def __init__(self, input_dim, learning_rate):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),  
        )

        self.learning_rate = learning_rate
        # weights = torch.FloatTensor([0.07, 0.93]) 
        # weights = weights.to(device)
        # self.criterion = nn.BCEWithLogitsLoss(weights)
        self.criterion = nn.BCEWithLogitsLoss()
        self.metrics = Metric()
        
    def forward(self, image):
        if len(image.shape) > 2:
            image = image.view(image.size(0), -1)
        x = self.fc(image)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr = self.learning_rate)
        return optimizer

    def training_step(self, batch):
        image, labels = batch
        labels = labels.float()
        preds = self.forward(image).squeeze(1)
        loss = self.criterion(preds, labels)
        self.log('train_loss', loss)

        # Use training metrics
        metrics = self.metrics.training_metric(preds, labels)
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        image, labels = batch
        labels = labels.float()
        preds = self.forward(image).squeeze(1)
        loss = self.criterion(preds, labels)
        self.log('val_loss', loss)

        # Use validation metrics
        metrics = self.metrics.validation_metric(preds, labels)
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        image, labels = batch
        labels = labels.float()
        preds = self.forward(image).squeeze(1)
        loss = self.criterion(preds, labels)
        self.log('test_loss', loss)

        # Use Metrics for other test metrics
        metrics = self.metrics.test_metric(preds, labels)
        self.log_dict(metrics)
        return metrics
