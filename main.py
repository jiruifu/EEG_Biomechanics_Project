from dependency import extract_vo2_data, read_bdf_file, interpolate_vo2_data
from dependency import EEGVO2Dataset, sliding_window_mean
from utils import setup_experiment_logger
from model import CNN_VO2_1D
from dependency import Trainer
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

shuffle_flag = True
batch_size = 32
eeg_path = r"C:\Users\jirui\Documents\eeg_dataset\Pilot4_data\EEG\Pilot4Oct_OG_A.bdf"
vo2_path = r"C:\Users\jirui\Documents\eeg_dataset\Pilot4_data\Metabolics\pilot4Oct_OG_A.xlsx"
window_size = 64
window_stride = 16
interpolation_method = "two"
train_test_split_ratio = 0.05
train_val_split_ratio = 0.2

step1 = True
step2 = True

if step1:
    # Read the EEG data
    num_eeg_channels, fsamp, time_duration, new_time_duration_returned, raw, raw_numpy, trimmed_raw, trimmed_raw_numpy = read_bdf_file(eeg_path, trim_flag=True, new_time_duration=293)
    print(f"The number of eeg channels is {num_eeg_channels}")
    # Extract the VO2 data
    vo2_df = extract_vo2_data(vo2_path)

    # Interpolate the VO2 data
    vo2_df_interpolated, vo2_array = interpolate_vo2_data(vo2_df, target_length=trimmed_raw_numpy.shape[1], method=interpolation_method)

    # Define the x, and y
    x, y = sliding_window_mean(trimmed_raw_numpy, vo2_array, num_eeg_channels, window_size, window_stride)


    # Create the dataset
    x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(x, y, test_size=train_test_split_ratio, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=train_val_split_ratio, random_state=42)

    print(f"The shape of x_train_all is {x_train_all.shape}")
    print(f"The shape of y_train_all is {y_train_all.shape}")
    print(f"The shape of x_test_all is {x_test_all.shape}")
    print(f"The shape of y_test_all is {y_test_all.shape}")
    print(f"The shape of x_train is {x_train.shape}")
    print(f"The shape of y_train is {y_train.shape}")
    print(f"The shape of x_val is {x_val.shape}")
    print(f"The shape of y_val is {y_val.shape}")

    train_dataset = EEGVO2Dataset(x_train, y_train)
    val_dataset = EEGVO2Dataset(x_val, y_val)
    test_dataset = EEGVO2Dataset(x_test_all, y_test_all)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_flag)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_flag)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_flag)

    print(f"The shape of the train loader is {len(train_loader)}")
    print(f"The shape of the val loader is {len(val_loader)}")
    print(f"The shape of the test loader is {len(test_loader)}")

    if step2:
        model = CNN_VO2_1D(num_eeg_channels, window_size, numNodes=[128, 128, 128, 64, 256])
        learning_rate = 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        num_epochs = 300

        predictor_trainer = Trainer(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs=num_epochs)

        train_loss_list, epoch_val_loss_list, test_loss, test_accuracy = predictor_trainer(num_epochs)