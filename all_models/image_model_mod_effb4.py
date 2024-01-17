import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
# my imports
from metric import Metric
from path_and_parameters import Defining_Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Image_Model_effb4(pl.LightningModule):
    def __init__(self, input_dim, learning_rate):
        super().__init__()
        #replacing the final layer of the image model by a small set of Sequential layers
        #Check input dimension #torch.size[16,512,1,1]
     
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_dim, 1792, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1792, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True),
            )
        self.adaptive = nn.AdaptiveAvgPool2d(output_size=(1))
        self.fc = nn.Sequential(
            nn.Linear(1792, 256),
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

        
    def forward(self, image):
        x = self.conv_block(image)
        x= self.adaptive(x)
        # print(x.size()) 
        # raise
        x = x.view(x.size(0), -1)  # flatten the tensor before fc layer
        x = self.fc(x)  
        return x


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr = self.learning_rate)
        return optimizer
    
    def training_step(self, batch):
        image, labels = batch
        preds = self.forward(image)
        loss = self.criterion(preds, labels)
        self.log('train_loss', loss)

        # Use training metrics
        metrics = self.metrics.training_metric(preds, labels)
        self.log_dict(metrics)
        return loss

    # def validation_step(self, batch, batch_idx, dataloader_idx=0):
    def validation_step(self, batch, batch_idx):
        image, labels = batch
        preds = self.forward(image)
        loss = self.criterion(preds, labels)
        self.log('val_loss', loss)

        # Use validation metrics
        metrics = self.metrics.validation_metric(preds, labels)
        self.log_dict(metrics)  # log the metrics dictionary here
        return metrics


    
    def test_step(self, batch, batch_idx):
        image, labels = batch
        preds = self.forward(image)
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

        return metrics

