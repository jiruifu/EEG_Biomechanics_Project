from datetime import datetime
from torchmetrics import F1Score
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from torchsummary import summary
import os
import sys
from utils.emgUtils import EMG_Loader_3D
from utils.customizedDataset import Custom3dDataset
from model.model_mo_3d import NMI_3D, NMI_3D_Lite

class Trainer3DCNN:
    def __init__(self, device, model, x_data, y_data, criterion, optimizer,
                 classNum: int=4, batch_size:list=[32, 32], splitRat=0.2):
        self.device = device
        self.model = model.to(self.device)
        self.x_train, self.x_val, self.y_train, self.y_val=train_test_split(x_data, y_data,
                                                                            test_size = splitRat, random_state = 42)
        self.trainSet = Custom3dDataset(self.x_train, self.y_train)
        self.valSet = Custom3dDataset(self.x_val, self.y_val)
        self.train_loader = DataLoader(self.trainSet, batch_size = batch_size[0], shuffle = True)
        self.val_loader = DataLoader(self.valSet, batch_size=batch_size[1], shuffle=False)
        self.criterion = criterion
        self.optimizer = optimizer
        self.f1_score  = BinaryF1Score(multidim_average='global').to(device)
        self.f1_score_2 = MulticlassF1Score(average='weighted', num_classes=classNum).to(device)

    def _train_one_epoch(self):
        self.model.train()
        running_loss = []
        for inputs, labels in self.train_loader:
            #move data to the device
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            #zero the gradients
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())
        avg_loss = sum(running_loss) / len(running_loss)
        return avg_loss

    def _validate_one_epoch(self):
        with torch.no_grad():
            self.model.eval()
            f1List = []
            f1List_test = []
            accList = []
            for inputs, labels in self.val_loader:
                x, y = inputs.to(self.device), labels.to(self.device)
                y_pred = self.model(x)
                # Calculate the validation accuracy and then convert the accuracy to a python scalar
                acc = (y_pred.round() == y).float().mean().cpu()
                acc = float(acc)
                accList.append(acc)
                # y_pred = y_pred > 0.5
                # outputs_np = y_pred.cpu().numpy()
                # labels_np = y.bool().cpu().numpy()
                # print(outputs_np)
                # print(labels_np)
                # f1_scoreVal = sklearn.metrics.f1_score(labels_np, outputs_np, average='macro')
                f1_score1 = self.f1_score(y_pred, y).cpu().mean().numpy()
                f1List.append(f1_score1)
                f1_score2 = self.f1_score_2(y_pred, y).cpu().mean().numpy()
                f1List_test.append(f1_score2)
            f1_scoreAvg = sum(f1List) / len(f1List)
            f1_scoreAvg_test = sum(f1List_test) / len(f1List_test)
            accAvg = sum(accList) / len(accList)
        return accAvg, f1_scoreAvg, f1_scoreAvg_test

    def __call__(self, num_epochs):
        model_ret = dict(epoch=[], train_loss=[], val_acc=[], f1_score=[])
        for epoch in range(num_epochs):
            # Training step
            trainLoss = self._train_one_epoch()

            # Validation step
            val_loss, f1_score, f1_weight = self._validate_one_epoch()

            # Print train loss and validation metrics for the epoch
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {trainLoss:.4f}, Accuracy: {val_loss:.4f}, F1 Score: {f1_score:.4f}, F1 Score with Weight: {f1_weight:.4f}')
            model_ret['epoch'].append(epoch)
            model_ret['train_loss'].append(trainLoss)
            model_ret['f1_score'].append(f1_score)
            model_ret['val_acc'].append(val_loss)
        trainedModel = self.model
        return model_ret, trainedModel

