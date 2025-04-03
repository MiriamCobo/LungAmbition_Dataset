import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_training_metrics(df, path_to_save_figs):
    """
    Plot training and validation metrics from a DataFrame.

    Parameters:
        df (pd.DataFrame): A DataFrame containing metrics with the following columns:
            ['epoch', 'train_loss', 'train_acc', 'train_bal_acc', 'train_mae', 'train_uoc_index',
             'train_correct_att_within1_norm', 'train_correct_att_norm', 'val_loss', 'val_acc',
             'val_bal_acc', 'val_mae', 'val_uoc_index', 'val_correct_att_within1_norm', 'val_correct_att_norm']
        path_to_save_figs (str): Path to save the ourput figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot train_loss and val_loss vs epoch
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='o')
    axes[0, 0].set_title('Loss vs Epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Plot train_acc and val_acc vs epoch
    axes[0, 1].plot(df['epoch'], df['train_acc'], label='Train Accuracy', marker='o')
    axes[0, 1].plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='o')
    axes[0, 1].set_title('Accuracy vs Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # Plot train_bal_acc and val_bal_acc vs epoch
    axes[0, 2].plot(df['epoch'], df['train_bal_acc'], label='Train Balanced Accuracy', marker='o')
    axes[0, 2].plot(df['epoch'], df['val_bal_acc'], label='Validation Balanced Accuracy', marker='o')
    axes[0, 2].set_title('Balanced Accuracy vs Epoch')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Balanced Accuracy')
    axes[0, 2].legend()

    # Plot train_mae and val_mae vs epoch
    axes[1, 0].plot(df['epoch'], df['train_auc'], label='Train AUC', marker='o')
    axes[1, 0].plot(df['epoch'], df['val_auc'], label='Validation AUC', marker='o')
    axes[1, 0].set_title('AUC vs Epoch')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()

    # Plot train_uoc_index and val_uoc_index vs epoch
    axes[1, 1].plot(df['epoch'], df['train_f1_score'], label='Train F1-score', marker='o')
    axes[1, 1].plot(df['epoch'], df['val_f1_score'], label='Validation F1-score', marker='o')
    axes[1, 1].set_title('F1-score vs Epoch')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1-score')
    axes[1, 1].legend()
    # Clear the subplot without removing it from the figure
    axes[1, 2].cla()  # Clear the axis
    axes[1, 2].set_xticks([])  # Remove x-axis ticks
    axes[1, 2].set_yticks([])  # Remove y-axis ticks
    axes[1, 2].set_frame_on(False)  # Remove the border/frame

    plt.tight_layout()
    plt.savefig(path_to_save_figs, dpi=300)
    plt.close()