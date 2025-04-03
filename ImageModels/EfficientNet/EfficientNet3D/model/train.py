"""
Training functions
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from model.loss import total_loss
import gc
import os
from model.metrics import calculate_binary_metrics
import torch.nn as nn

def train_model(trainmodel, data_loader, arguments, optim, class_weights, device):
    """
    Train model for one epoch.
    :param trainmodel: model to be trained
    :param data_loader: data_loader used for training
    :param arguments: parser arguments
    :param optim: training optimizer
    :param class_weights: class weights for weighted ordinal malignancy loss
    :param device: device to train on
    :return: [trained model, mal_accuracy]
    """
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    trainmodel.train()
    # print(torch.cuda.memory_summary())
    torch.autograd.set_detect_anomaly(True)
    # save predictions and labels for calculating metrics
    all_pred_probs = []
    all_true_mal = []
    train_loss_total = 0

    for batch in data_loader:
        gc.collect()
        torch.cuda.empty_cache()
        x = batch["image"].to(device, dtype=torch.float)
        # y_mask = batch["seg"].to(device, dtype=torch.float)
        y_mal=batch["mal"]
        x, y_mal = x.to(device, dtype=torch.float), y_mal.to(device, dtype=torch.int64)
        # y_mask.to(device, dtype=torch.float), \

        pred_mal = trainmodel(x)
        # pred_mal = torch.squeeze(pred_mal)

        loss = total_loss(y=y_mal.view(-1, 1), y_pred=pred_mal,
                                        ord_weights=class_weights,
                                        device=device,arguments=arguments)

        loss.backward()
        optim.step()
        optim.zero_grad()
        train_loss_total += loss.item()* x.size(0)
        
        all_pred_probs.append(nn.Sigmoid()(pred_mal).cpu().detach().numpy())
        all_true_mal.append(y_mal.cpu().detach().numpy())

    train_loss_total /= len(data_loader.dataset)
    all_train_loss = {'total': train_loss_total}
    # Concatenate all predictions and ground truths
    all_true_mal = np.concatenate(all_true_mal)
    all_pred_probs = np.concatenate(all_pred_probs)
    # threshold all_pred_probs to get all_pred_mal with threshold = 0.5
    all_pred_mal = (all_pred_probs[:, 0] > 0.5).astype(int)
    # print counts of all_pred_mal and all_true_mal per class
    print("Counts of all_true_mal per class: ", np.unique(all_true_mal, return_counts=True))
    print("Counts of all_pred_mal per class: ", np.unique(all_pred_mal, return_counts=True))
    print("Probs 20 first samples:", all_pred_probs[:20])
    # print("train all_pred_probs: ", all_pred_probs)
    # print("train all_pred_mal: ", all_pred_mal)
    # print("train all_true_mal: ", all_true_mal)
    # calculate metrics for ordinal malignancy prediction
    usual_metrics_mal = calculate_binary_metrics(y_true=all_true_mal, y_pred=all_pred_mal, y_pred_probs=all_pred_probs)

    return trainmodel, all_train_loss, usual_metrics_mal