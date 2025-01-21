import torch
import time
import numpy as np


class Trainer:
    """
    This class is used to train the model
    """
    def __init__(self, model, train_loader, val_loader, \
        test_loader, optimizer, criterion, num_epochs=100,\
             learning_rate=0.001, device="cuda") -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def __call__(self, epochs=100):
        self.num_epochs = epochs

        train_loss_list, epoch_val_loss_list, test_loss, test_accuracy = self._train()

        return train_loss_list, epoch_val_loss_list, test_loss, test_accuracy

    def _train(self):
        def test_with_batch(batch_size):
            self.model.eval()
            with torch.no_grad():
                loss_list = []
                accuracy_list = []
                for eeg, vo2 in self.test_loader:
                    data, target = eeg.to(self.device), vo2.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss_list.append(loss.item())
                    accuracy = (output.round() == target).float().mean()
                    accuracy_list.append(accuracy.item())
            return np.mean(loss_list), np.mean(accuracy_list)
        
        def val_with_batch(batch_size):
            self.model.eval()
            with torch.no_grad():
                loss_list = []
                accuracy_list = []
                for eeg, vo2 in self.val_loader:
                    data, target = eeg.to(self.device), vo2.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss_list.append(loss.item())
                    accuracy = (output.round() == target).float().mean()
                    accuracy_list.append(accuracy.item())
            return np.mean(loss_list), np.mean(accuracy_list)

        self.model.train()
        train_loss_list = []
        epoch_val_loss_list = []
        for epoch in range(self.num_epochs):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss_list.append(loss.item())
            val_loss, val_accuracy = val_with_batch(batch_size=100)
            epoch_val_loss_list.append(val_loss)
            print(f"Epoch {epoch+1} of {self.num_epochs} - Train Loss: {np.mean(train_loss_list):.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
        
        test_loss, test_accuracy = test_with_batch(batch_size=100)
        print(f"Finished Training - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
        return epoch_val_loss_list, train_loss_list,test_loss, test_accuracy
