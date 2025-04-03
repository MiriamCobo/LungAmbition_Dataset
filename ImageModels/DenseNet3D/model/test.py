"""
Testing functions
"""

import os
import torch
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
# from torchmetrics import Dice
from model.metrics import calculate_binary_metrics
from model.loss import total_loss
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn

def val_model(testmodel, data_loader, class_weights, arguments, device):
    """
    Validate model
    :param testmodel: model to be tested
    :param data_loader: data_loader used for testing
    :param arguments: parser arguments
    :param device: device to run test on
    :return: [tested model, mal_accuracy]
    """
    testmodel.eval()
    test_loss_total = 0
    # save predictions and labels for calculating metrics
    all_true_mal = []
    all_pred_probs = []
    all_id_patient = []
    # all_segment_path = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch["image"].to(device, dtype=torch.float)
            # y_mask = batch["seg"].to(device, dtype=torch.float)
            y_mal=batch["mal"].to(device, dtype=torch.int64)
            all_id_patient.extend(batch["id"])
            # all_segment_path.extend(batch["seg_path"])
            pred_mal = testmodel(x)
            val_loss = total_loss(y=y_mal.view(-1, 1), y_pred=pred_mal,
                                        ord_weights=class_weights,
                                        device=device,arguments=arguments)
            test_loss_total += val_loss.item()* x.size(0)
            # pred_mal = pred_mal.squeeze(dim=-1)
            ### append to pred_label and gold_label
            all_true_mal.extend(y_mal.cpu().detach().numpy())
            all_pred_probs.extend(nn.Sigmoid()(pred_mal).squeeze(dim=-1).cpu().detach().numpy())

    test_loss_total /= len(data_loader.dataset)
    all_test_loss = {'total': test_loss_total}
    # Concatenate all predictions and ground truths
    all_true_mal = np.array(all_true_mal)
    all_pred_probs = np.array(all_pred_probs)
    all_ids = np.array(all_id_patient)
    all_ids = np.squeeze(all_ids)  # Convert (276,1) → (276,)
    all_true_mal = np.squeeze(all_true_mal)  # Convert (276,1) → (276,)
    # threshold = 0.5
    all_pred_mal = (all_pred_probs > 0.5).astype(int)
    print("val all_true_mal", all_true_mal)
    print("val all_pred_mal", all_pred_mal)
    print("val all_pred_prob", all_pred_probs)
    # calculate metrics for ordinal malignancy prediction
    usual_metrics_mal = calculate_binary_metrics(y_true=all_true_mal, y_pred=all_pred_mal, y_pred_probs=all_pred_probs)
    
    return all_test_loss, usual_metrics_mal

def test_model(testmodel, data_loader, class_weights, arguments, device, path_to_save_figs, path_to_save_csvs):
    """
    Test model
    :param testmodel: model to be tested
    :param data_loader: data_loader used for testing
    :param arguments: parser arguments
    :param device: device to run test on
    :return: [tested model, mal_accuracy]
    """
    testmodel.eval()
    test_loss_total = 0
    test_loss_pred = 0
    # save predictions and labels for calculating metrics
    all_pred_probs = []
    all_true_mal = []
    all_id_patient = []
    # all_segment_path = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch["image"].to(device, dtype=torch.float)
            # y_mask = batch["seg"].to(device, dtype=torch.float)
            y_mal=batch["mal"].to(device, dtype=torch.int64)
            all_id_patient.extend(batch["id"])
            # all_segment_path.extend(batch["seg_path"])
            pred_mal = testmodel(x)
            if class_weights is not None:
                test_loss = total_loss(y=y_mal.view(-1, 1), y_pred=pred_mal,
                                            ord_weights=class_weights,
                                            device=device,arguments=arguments)
                test_loss_total += test_loss.item()* x.size(0)
            all_pred_mal_prob = nn.Sigmoid()(pred_mal)
            all_pred_probs.extend(all_pred_mal_prob.squeeze(dim=-1).cpu().detach().numpy())
            # pred_mal = pred_mal.squeeze(dim=-1)
            all_true_mal.extend(y_mal.cpu().detach().numpy())

    test_loss_total /= len(data_loader.dataset)
    all_test_loss = {'total': test_loss_total}
    # Concatenate all predictions and ground truths
    all_true_mal = np.array(all_true_mal)
    all_pred_probs = np.array(all_pred_probs)
    all_ids = np.array(all_id_patient)
    all_ids = np.squeeze(all_ids)  # Convert (276,1) → (276,)
    all_true_mal = np.squeeze(all_true_mal)  # Convert (276,1) → (276,)
    # threshold = 0.5
    all_pred_mal = (all_pred_probs > 0.5).astype(int)
    # calculate metrics for ordinal malignancy prediction
    usual_metrics_mal = calculate_binary_metrics(y_true=all_true_mal, y_pred=all_pred_mal, y_pred_probs=all_pred_probs)
    print("test all_true_mal", all_true_mal)
    print("test all_pred_mal", all_pred_mal)
    print("test all_pred_prob", all_pred_probs)
    print("usual metrics mal", usual_metrics_mal)
    # save to csv in path_to_save_csvs
    df = pd.DataFrame({
        "scan_id": all_ids,
        "pred_mal": all_pred_mal,
        "true_mal": all_true_mal
    })
    df.to_csv(path_to_save_csvs+".csv", index=False)

    global_confusion_matrix = confusion_matrix(all_true_mal, all_pred_mal, labels=[0,1])
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(global_confusion_matrix, interpolation='nearest', cmap='Blues')
    plt.colorbar()

    # Add labels and title
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Global Confusion Matrix")

    # Add text annotations for each cell
    threshold = global_confusion_matrix.max() / 2.0
    for i in range(global_confusion_matrix.shape[0]):
        for j in range(global_confusion_matrix.shape[1]):
            color = "white" if global_confusion_matrix[i, j] > threshold else "black"
            plt.text(j, i, format(global_confusion_matrix[i, j], 'd'),
                    ha="center", va="center", color=color)

    # Configure axis ticks
    plt.xticks(ticks=np.arange(len([0, 1])), labels=[0, 1])
    plt.yticks(ticks=np.arange(len([0, 1])), labels=[0, 1])

    # Save the plot
    plt.savefig(path_to_save_figs)
    return all_test_loss, usual_metrics_mal
