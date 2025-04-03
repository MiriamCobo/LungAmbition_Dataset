"""
Loss function
"""

import torch
from torch import nn
import torch.nn.functional as F
# from dlordinal.losses import BinomialCrossEntropyLoss, BetaCrossEntropyLoss, PoissonCrossEntropyLoss
import numpy as np
# from monai.losses import DiceLoss


def total_loss(y, y_pred, ord_weights, device, arguments):
    # print(f"Debugging: y_pred: {y_pred}")  
    # print(f"Debugging: y: {y}") 
    # print(f"🔎 Before squeeze: y shape = {y.shape}, unique values: {torch.unique(y)}")
    # if y.dim() > 1 and y.shape[-1] == 1:
    #     y = y.squeeze(dim=-1)  # Remove only if last dim is 1
    # y = y.long() 
    # print("y_pred shape: ", y_pred.shape)
    # print("y shape: ", y.shape)
    # print(f"🔎 After squeeze: y shape = {y.shape}, unique values: {torch.unique(y)}")
    L_pred = nn.BCEWithLogitsLoss(pos_weight=ord_weights).to(device)(y_pred, y.float())
        
    return L_pred