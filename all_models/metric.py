# ##########################  automate metric #######################3
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity, ConfusionMatrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Metric:
    def __init__(self):
        self.accuracy = Accuracy(task='binary').to(device)
        self.precision = Precision(task='binary', average='weighted', num_classes=2).to(device)
        self.recall = Recall(task='binary', average='weighted', num_classes=2).to(device)
        self.f1 = F1Score(task='binary', average='weighted', num_classes=2).to(device)
        self.specificity = Specificity(task='binary', average='weighted', num_classes=2).to(device)

# ############ changed this for bce loss ##########
#     def calculate_metrics(self, preds, labels):
#         preds_probs = torch.sigmoid(preds)
#         preds_bin = (preds_probs >= 0.5).float()

#         Accuracy = self.accuracy(preds_bin, labels)
#         Precision = self.precision(preds_bin, labels)
#         Recall = self.recall(preds_bin, labels)
#         F1 = self.f1(preds_bin, labels)
#         Specificity = self.specificity(preds_bin, labels)

#         metrics = {
#             'acc': Accuracy,
#             'prec': Precision,
#             'rec': Recall,  #same as sensitivity
#             'f1': F1,
#             'spec': Specificity,
#         }
#         return metrics

############

    def calculate_metrics(self, preds, labels):
        # preds_softmax = torch.softmax(preds, dim=1)
        # preds_argmax = torch.argmax(preds_softmax, dim=1)

        preds_argmax = torch.argmax(preds, dim=1)

        Accuracy = self.accuracy(preds_argmax, labels)
        Precision = self.precision(preds_argmax, labels)
        Recall = self.recall(preds_argmax, labels)
        F1 = self.f1(preds_argmax, labels)
        Specificity = self.specificity(preds_argmax, labels)

 
        metrics = {
            'acc': Accuracy,
            'prec': Precision,
            'rec': Recall,  #same as sensitivity
            'f1': F1,
            'spec': Specificity,
        }
        return metrics

    def training_metric(self, preds, labels):
        metrics = self.calculate_metrics(preds, labels)
        metrics = {'train_' + k: v for k, v in metrics.items()}
        return metrics

    def validation_metric(self, preds, labels):
        metrics = self.calculate_metrics(preds, labels)
        metrics = {'val_' + k: v for k, v in metrics.items()}
        return metrics

    def test_metric(self, preds, labels):
        metrics = self.calculate_metrics(preds, labels)
        metrics = {'test_' + k: v for k, v in metrics.items()}
        return metrics




