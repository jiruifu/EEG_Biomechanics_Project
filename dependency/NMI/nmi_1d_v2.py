# Trainer for 2D CNN
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import sklearn
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import time
from torchsummary import summary
import os
import sys
from utils import plotter
from utils.emgUtils import EMG_Loader_2D
from utils.customizedDataset import Custom2dDataset
from model.model_1DCNN import MODCNN_1D_MUST
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Trainer1DCNN:
    def __init__(self, device, model, x_data, y_data, optimizer,
                 criterion, test_size=0.2, batch_size:list=[64, 64]):
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        # Define the dataload by using the built-in dataloader function of pytorch
        self.device = device
        self.model = model.to(self.device)
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=test_size, random_state=42)
        train_dataset = Custom2dDataset(x_train, y_train)
        val_dataset = Custom2dDataset(x_val, y_val)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size[0], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size[1], shuffle=False)
        # self.f1_score  = BinaryF1Score(multidim_average='global').to(device)
        # self.f1_score_2 = MulticlassF1Score(average='weighted', num_classes=numMu).to(device)
        print("Number of train batch: {}; train batch size is: {}".format(len(self.train_loader), self.batch_size[0]))
        print("Number of val batch: {}; val batch size is: {}".format(len(self.val_loader), self.batch_size[1]))
        # Check if the file exists, create if it does not

    def _train_one_epoch(self):
        self.model.train()
        running_loss = []
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device),labels.to(self.device).view(-1, 1)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())
        avg_loss = sum(running_loss) / len(running_loss)
        return avg_loss
    
    def _val_one_epoch(self):
        with torch.no_grad():
            self.model.eval()
            f1List = []
            f1List_test = []
            accList = []
            val_losses = []
            all_outputs = []
            all_labels = []

            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).view(-1, 1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                acc = (outputs.round() == labels).float().mean().cpu()
                acc = float(acc)
                accList.append(acc)
                val_losses.append(loss.item())
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                # outputs = outputs > 0.5
                # outputs_np = outputs.cpu().numpy()
                # labels_np = labels.bool().cpu().numpy()
                # f1_scoreVal = sklearn.metrics.f1_score(labels_np, outputs_np, average='macro', zero_division=0)
                # f1_score1 = self.f1_score(outputs, labels).cpu().mean().numpy()
                # f1_score2 = self.f1_score_2(outputs, labels).cpu().mean().numpy()
                # f1List.append(f1_score1)
                # f1List_test.append(f1_score2)
            avg_val_loss = sum(val_losses) / len(val_losses)
            mae = mean_absolute_error(all_labels, all_outputs)
            mse = mean_squared_error(all_labels, all_outputs)
        # f1_scoreAvg = sum(f1List)/len(f1List)
        # f1_scoreAvg_test = sum(f1List_test) / len(f1List_test)
        acc = sum(accList)/len(accList)
        f1_scoreAvg = None
        f1_scoreAvg_test = None
        return avg_val_loss, mae, mse
    
    def __call__(self, epoches=1000):
        num_epochs = epoches
        model_ret = dict(epoch=[], train_loss=[], val_loss=[], mae=[], mse=[])

        for epoch in range(num_epochs):
            # Training step
            trainingLoss = self._train_one_epoch()
            # Validation step
            valLoss, mae, mse = self._val_one_epoch()
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {trainingLoss:.4f}, Validation Loss: {valLoss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}')
            model_ret['epoch'].append(epoch+1)
            model_ret['train_loss'].append(trainingLoss)
            # model_ret['f1_score'].append(f1Score)
            model_ret['val_loss'].append(valLoss)
            model_ret['mae'].append(mae)
            model_ret['mse'].append(mse)
        trainedModel = self.model
        return model_ret, trainedModel
