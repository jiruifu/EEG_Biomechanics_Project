import numpy as np
import matplotlib.pyplot as plt

def remove_outliers(data):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return filtered_data

def plot_metrics(model_ret):
    epochs = model_ret['epoch']
    
    # Remove outliers from each metric
    train_loss = remove_outliers(model_ret['train_loss'])
    val_loss = remove_outliers(model_ret['val_loss'])
    mae = remove_outliers(model_ret['mae'])
    mse = remove_outliers(model_ret['mse'])
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot training loss
    axs[0, 0].plot(epochs[:len(train_loss)], train_loss, label='Training Loss')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    # Plot validation loss
    axs[0, 1].plot(epochs[:len(val_loss)], val_loss, label='Validation Loss', color='orange')
    axs[0, 1].set_title('Validation Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    
    # Plot MAE
    axs[1, 0].plot(epochs[:len(mae)], mae, label='Mean Absolute Error', color='green')
    axs[1, 0].set_title('Mean Absolute Error')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('MAE')
    axs[1, 0].legend()
    
    # Plot MSE
    axs[1, 1].plot(epochs[:len(mse)], mse, label='Mean Squared Error', color='red')
    axs[1, 1].set_title('Mean Squared Error')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('MSE')
    axs[1, 1].legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.show()