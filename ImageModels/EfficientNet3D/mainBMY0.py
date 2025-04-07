"""
Author: Miriam Cobo Cano
"""
import torch.nn as nn
import sys
import torch
import datetime
import numpy as np
from model.train import train_model
from model.test import val_model, test_model
import pandas as pd
from sklearn.utils import class_weight
import logging
from model.plots import plot_training_metrics
sys.path.append('/home/ubuntu/tenerife/data/ZZ_githubRepos/baselinesLungAmbition/ImageModels/LoadData')
from load_LungAmbition3D_test_data_binary import load_lungAmbition
# for paralelization
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
import psutil
import ast
from monai.networks.nets import EfficientNetBN, EfficientNetBNFeatures
torch.cuda.empty_cache()

class EfficientNetB0Regularized(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = EfficientNetBNFeatures(
            model_name="efficientnet-b0",
            spatial_dims=3,
            in_channels=1,
            num_classes=1,  # Not used by this class
            pretrained=False,
            norm=("group", {"num_groups": 8, "eps": 1e-5, "affine": True})
        )
        self.backbone.extract_stacks = (7,)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(320, 1)  # 1280 is fixed for efficientnet-b0

    def forward(self, x):
        x = self.backbone(x)[0]
        # print("Shape after backbone:", x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def print_metrics(binary_metrics):
    """
    Prints the given dictionary of binary classification metrics in a structured format.
    Values are rounded to 3 decimal places.
    
    Args:
        binary_metrics (dict): Dictionary containing metric names as keys and their values.
    """
    for key in ["roc_auc", "accuracy", "balanced_accuracy", "sensitivity", "precision", "f1_score",
                "specificity", "NPV", "ppv"]:
        if key in binary_metrics:
            print(f"{key.replace('_', ' ').capitalize()}: {binary_metrics[key]:.3f}")

if __name__ == "__main__":
    import argparse
    import os
    import json
    from types import SimpleNamespace

    np.set_printoptions(precision=3)
    n_folds=3
    path_to_folds_csv = f'Data_stratified_split/folds-def_{n_folds}folds' # read in your location
    keep_false_positives_as_separate_test = True

    parser = argparse.ArgumentParser(description="DenseNet")
    parser.add_argument('--config', type=str, default="config.json", help="Path to JSON config file")
    # Parse the arguments
    args = parser.parse_args()
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            config_args = SimpleNamespace(**config_dict)
    else:
        raise FileNotFoundError(f"Config file not found at {args.config}")

    if (not config_args.train and not config_args.test):
        raise TypeError(
            "Please specify, whether you want to run the training or testing code by setting the parameter --train=True or --test=True")
    
    # create from config_args.path_to_save_outputs subpaths to save data and results
    for path in [
        config_args.path_to_save_outputs,
        f"{config_args.path_to_save_outputs}/Model",
        f"{config_args.path_to_save_outputs}/Results",
        f"{config_args.path_to_save_outputs}/Figures",
        f"{config_args.path_to_save_outputs}/Test"
    ]:
        os.makedirs(path, exist_ok=True)
    # save output files in sys file
    sys.stdout = open(os.path.join(config_args.path_to_save_outputs, config_args.name_output_file), mode='w')
    original_stdout = sys.stdout

    df_merged = pd.read_csv('df_merged.csv') # read df_merged.csv in your location
    # filter df_merged by GroupUpdated to keep only Lung_Cancer, Benign_Nodules and False_Positive
    df_merged = df_merged[df_merged['GroupUpdated'].isin(['Lung_Cancer', 'Benign_Nodules', 'False_Positive'])]
    df_merged = df_merged[['ID_proteinData', 'Group', 'Stage_category', 'NRRD_File', 'SEG_Files', 'Cancer_Status', 'TimeYears_CT_blood']]
    df_merged['SEG_Files'] = df_merged['SEG_Files'].apply(ast.literal_eval)
    if keep_false_positives_as_separate_test:
        y_false_positives = df_merged[df_merged['Group'] == 'False_Positive']['Group']
        # convert label to 1
        y_false_positives = y_false_positives.replace({'False_Positive': 0})
        ID_false_positives = df_merged[df_merged['Group'] == 'False_Positive']['ID_proteinData']
        # create list to store wrong predicted false positives
        list_ID_wrong_predicted_false_positives = []
        X_false_positives = df_merged[df_merged['Group'] == 'False_Positive'].drop(columns=['ID_proteinData', 'Group'])
        # drop in df_cur rows where Group is False_Positive
        df_merged = df_merged[df_merged['Group'].isin(['Lung_Cancer', 'Benign_Nodules'])]
        print("Number of false positives:", X_false_positives.shape[0])
        # save false_positive_metrics in df
        false_positive_metrics = pd.DataFrame(columns=['Fold', 'AUC', 'Balanced_accuracy', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
    else:
        df_merged = df_merged[df_merged['Group'].isin(['Lung_Cancer', 'Benign_Nodules', 'False_Positive'])]
        # print shape of excel
        print("Dimensions excel after dropping rows where Group is False_Positive:", df_merged.shape)
    # assert that Cancer_Status is 1 for patients where TimeYears_CT_blood is 0 and
    # Cancer_Status is 0 for patients where TimeYears_CT_blood is 5
    assert (df_merged.loc[df_merged["TimeYears_CT_blood"] == 0, "Cancer_Status"] == 1).all(), \
        "There are patients with TimeYears_CT_blood = 0 who do not have Cancer_Status = 1"
    assert (df_merged.loc[df_merged["TimeYears_CT_blood"] == 5, "Cancer_Status"] == 0).all(), \
        "There are patients with TimeYears_CT_blood = 5 who do not have Cancer_Status = 0"
    # define a new malignancy column, if Cancer_Status is 0, then malignancy is 0, else 1 according to proposed method
    df_merged['Malignancy'] = df_merged['Cancer_Status'].apply(lambda x: 0 if x == 0 else 1)
    # save best metrics for each fold
    fold_metrics_df = pd.DataFrame(columns=['Fold', 'AUC', 'Accuracy', 'Bal_accuracy', 'Precision', 'Sensitivity', 'F1-score', 'Specificity', 'NPV'])

    for fold in range(2, n_folds):
        print("=" * 80)
        print(f"Fold {fold}:")
        # read train, test and val indices for each fold
        fold_data = pd.read_csv(os.path.join(path_to_folds_csv, f'id2splitfold_{fold}.csv'))
        train_index = fold_data[fold_data['split'] == 'train']['ID_proteinData']
        test_index = fold_data[fold_data['split'] == 'test']['ID_proteinData']
        val_index = fold_data[fold_data['split'] == 'val']['ID_proteinData']
        # get first train, text, val, then split into X_train, X_test, X_val and y_train, y_test, y_val
        train = df_merged.loc[df_merged['ID_proteinData'].isin(train_index)]
        test = df_merged.loc[df_merged['ID_proteinData'].isin(test_index)]
        val = df_merged.loc[df_merged['ID_proteinData'].isin(val_index)]
        # print number of samples in train, val and test for Malignancy 0 and 1
        print("Train, total benign nodules", train[train['Malignancy'] == 0].shape[0], "lung cancer", train[train['Malignancy'] == 1].shape[0])
        print("Val, total benign nodules", val[val['Malignancy'] == 0].shape[0], "lung cancer", val[val['Malignancy'] == 1].shape[0])
        print("Test, total benign nodules", test[test['Malignancy'] == 0].shape[0], "lung cancer", test[test['Malignancy'] == 1].shape[0])

        print(config_args)# save prints in a txt file
        # Confirm the selected mode
        if getattr(config_args, "train", True):
            print("Running in training mode...")
        elif getattr(config_args, "test", True):
            print("Running in testing mode...")
        # monitor_system()

        if config_args.train:
            train_loader = load_lungAmbition(train, batch_size=config_args.batch_size, spatial_size=[config_args.crop_size, config_args.crop_size, config_args.crop_size], shuffle=False, type_processing = None, augment_prob=0.5)
            val_loader = load_lungAmbition(val, batch_size=config_args.batch_size, spatial_size=[config_args.crop_size, config_args.crop_size, config_args.crop_size], shuffle=False, type_processing = None)
            test_loader = load_lungAmbition(test, batch_size=config_args.batch_size, spatial_size=[config_args.crop_size, config_args.crop_size, config_args.crop_size], shuffle=False, type_processing = None)
            print(f"Loaded LungAmbition data for fold {fold}")
            dist.init_process_group(backend='nccl')
            # Set up device
            rank = dist.get_rank()
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")

            seed = 1
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)

            resize_shape=(config_args.crop_size, config_args.crop_size, config_args.crop_size)
            all_epochs_metrics = pd.DataFrame(columns=['Fold', 'epoch', 'train_loss', 'train_auc', 
                                                       'train_acc', 'train_bal_acc', 'train_sensitivity', 
                                                       'train_precision', 'train_f1_score', 'train_specificity',
                                                        'train_NPV','val_loss', 'val_auc', 'val_acc', 'val_bal_acc',
                                                       'val_sensitivity', 'val_precision', 'val_f1_score', 'val_specificity','val_NPV'])
            # model = EfficientNetBN(
            #             model_name="efficientnet-b0",  # b0, b1, b2, ..., b7 # b3 does not learn in training, b5 is meh
            #             spatial_dims=3,
            #             in_channels=1,
            #             num_classes=1,
            #             pretrained=False  # Set to True if you want ImageNet pretrained weights (only 2D!)
            #         )
            model = EfficientNetB0Regularized(dropout_rate=0.3)
            model.to(device)
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
            print(model)
            # monitor_system()

            optimizer = torch.optim.AdamW(model.parameters(), lr=config_args.lr, weight_decay=0.01, amsgrad=True)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer, mode='min', factor=0.5, patience=5)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                base_lr=1e-4,    # Minimum LR
                max_lr=5e-4,     # Maximum LR
                step_size_up=500,  # Number of iterations to increase LR
                mode='triangular2',  # Scale function: triangular2 reduces amplitude by half each cycle
                cycle_momentum=False # Disable momentum updates (not needed for Adam)
            )

            print("Training samples: " + str(len(train_loader.dataset)))
            print("Val samples: " + str(len(val_loader.dataset)))
            print("Test samples: " + str(len(test_loader.dataset)))
            sys.stdout.flush()
            best_val_bal_acc = 0.0
            best_epoch_val_bal_acc = 0
            datestr = str(datetime.datetime.now())
            print("This run has datestr " + datestr)
            patience = 20 # change to retrain 16 feb 2024
            early_stop = False
            epochs_no_improve = 0
            ## consider defining class weights to acount for class imbalance!!!
           
            classes_array = np.array([int(c) for c in train.Malignancy.values])
            # class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(classes_array), y=classes_array)
            # print(f"{class_weights=}")
            num_positive = np.sum(classes_array == 1)  # Count of positive samples
            num_negative = np.sum(classes_array == 0)  # Count of negative samples
            # Compute pos_weight
            pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32).to(device)
            print(f"{pos_weight=}")
            # class_weights = (
            #     torch.from_numpy(class_weights).float().to(device)
            # )  # Transform to Tensor
            print("Training")
            lrs = []
            for epoch in range(config_args.epochs):
                print("Epoch " + str(epoch) + ' ' + '-' * 70)
                model, all_train_loss, usual_metrics_mal = train_model( ### here Id return epoch metrics
                    model, train_loader, config_args, optim=optimizer,
                    class_weights=pos_weight, 
                    device=device)
                print('Train total loss = %.3f' % (
                    all_train_loss['total']))
                print_metrics(usual_metrics_mal)

                print("Validation")
                val_loss_all, val_usual_metrics_mal = val_model(testmodel=model, 
                                                data_loader=val_loader, 
                                                class_weights=pos_weight,
                                                arguments=config_args, 
                                                device=device)
                print('Val total loss = %.3f' % (
                    val_loss_all['total']))
                # scheduler.step(val_loss_all['total'])
                scheduler.step()
                lrs.append(scheduler.get_last_lr()[0])

                print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
                val_bal_acc = val_usual_metrics_mal['balanced_accuracy']
                print_metrics(val_usual_metrics_mal)
                # monitor_system()
                if val_bal_acc > best_val_bal_acc:
                    best_val_bal_acc = val_bal_acc
                    epochs_no_improve = 0
                    best_epoch_val_bal_acc = epoch
                    print("New best val accuracy. Saving model")
                    torch.save(model, config_args.path_to_save_outputs + "/Model/" + config_args.model_name + "fold" + str(fold) + ".pt")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print("Early stopping")
                        break
                all_epochs_metrics = pd.concat([all_epochs_metrics if not all_epochs_metrics.empty else None, 
                                                pd.DataFrame([[fold, epoch, 
                                                            all_train_loss['total'], 
                                                            usual_metrics_mal['roc_auc'],
                                                            usual_metrics_mal['accuracy'],
                                                            usual_metrics_mal['balanced_accuracy'],
                                                            usual_metrics_mal['sensitivity'],
                                                            usual_metrics_mal['precision'],
                                                            usual_metrics_mal['f1_score'],
                                                            usual_metrics_mal['specificity'],
                                                            usual_metrics_mal['NPV'],
                                                            val_loss_all['total'],
                                                            val_usual_metrics_mal['roc_auc'],
                                                            val_usual_metrics_mal['accuracy'],
                                                            val_usual_metrics_mal['balanced_accuracy'],
                                                            val_usual_metrics_mal['sensitivity'],
                                                            val_usual_metrics_mal['precision'],
                                                            val_usual_metrics_mal['f1_score'],
                                                            val_usual_metrics_mal['specificity'],
                                                            val_usual_metrics_mal['NPV']
                                                            ]], columns=all_epochs_metrics.columns)], ignore_index=True)
                sys.stdout.flush()
            print(f"Training completed, best validation balanced accuracy: {best_val_bal_acc} at epoch {best_epoch_val_bal_acc}")
            # plot training and validation metrics
            path_to_save_figs = config_args.path_to_save_outputs + "/Figures/" + "training_val_metrics_fold" + str(fold) + "_model_" + str(config_args.model_name) + ".png"
            all_epochs_metrics = all_epochs_metrics.apply(lambda col: col.apply(lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else x)) # convert to numpy if there are pytorch tensors
            all_epochs_metrics.to_csv(config_args.path_to_save_outputs + "/Results/" + "training_val_metrics_" + str(fold) + "_model_" + str(config_args.model_name) + ".csv", index=False)
            plot_training_metrics(all_epochs_metrics, path_to_save_figs)
            plt.plot(lrs)
            plt.xlabel("Iterations")
            plt.ylabel("Learning Rate")
            plt.title("Cyclic Learning Rate Progression")
            plt.savefig(config_args.path_to_save_outputs + "/Figures/" + "cyclic_lr_progression_fold" + str(fold) + "_model_" + str(config_args.model_name) + ".png")
            plt.close()

            if config_args.test: 
                # Check and assign values only if they are not already defined
                if 'device' not in globals():
                    str_cuda = config_args.gpu_device
                    device = torch.device(str_cuda if torch.cuda.is_available() else "cpu")
                if 'model' not in globals():
                    if config_args.model_name is None:
                        raise TypeError("Please specify the path to model by setting the parameter --model_name=\"name_model\"")
                    path_model = config_args.path_to_save_outputs + "/Model/" + config_args.model_name + ".pt"
                    model = torch.load(path_model)
                if 'test_loader' not in globals():
                    test_loader = load_lungAmbition(
                        test, 
                        batch_size=config_args.batch_size, 
                        spatial_size=[config_args.crop_size, config_args.crop_size, config_args.crop_size], 
                        shuffle=False, 
                        type_processing=None
                    )
                    print(f"Loaded LungAmbition data for fold {fold}")
                if 'path_to_save_figs' not in globals():
                    path_to_save_figs = config_args.path_to_save_outputs + "/Test/" + "ConfMatrix_Test_" + config_args.model_name
                if 'path_to_save_csvs' not in globals():
                    path_to_save_csvs = config_args.path_to_save_outputs + "/Test/" + "TestResults_" + config_args.model_name + "_fold" + str(fold)
                print("Testing")
                test_loss_all, test_usual_metrics_mal = test_model(testmodel=model, 
                                                    data_loader=test_loader,
                                                    class_weights=pos_weight,
                                                    arguments=config_args, 
                                                    device=device, path_to_save_figs = path_to_save_figs, path_to_save_csvs = path_to_save_csvs)
                # print test_usual_metrics_mal
                print('Test total loss = %.3f' % (
                    test_loss_all['total']))
                print_metrics(test_usual_metrics_mal)
                # save test_usual_metrics_mal in  fold_metrics_df
                fold_metrics_df = pd.concat([fold_metrics_df if not fold_metrics_df.empty else None, 
                                                pd.DataFrame([[fold, test_usual_metrics_mal['roc_auc'], 
                                                               test_usual_metrics_mal['accuracy'], 
                                                               test_usual_metrics_mal['balanced_accuracy'], 
                                                               test_usual_metrics_mal['precision'], 
                                                               test_usual_metrics_mal['sensitivity'], 
                                                               test_usual_metrics_mal['f1_score'], 
                                                               test_usual_metrics_mal['specificity'], 
                                                               test_usual_metrics_mal['NPV']]], columns=fold_metrics_df.columns)], ignore_index=True)
                # save as csv fold_metrics_df
                fold_metrics_df.to_csv(config_args.path_to_save_outputs + "/Results/" + "metrics_df_test_fold" +str(fold)+ "_" + str(config_args.model_name) + ".csv", index=False)
            # Clean up
            dist.barrier()
            dist.destroy_process_group()
            torch.manual_seed(seed)
            torch.cuda.empty_cache()
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.synchronize()
            sys.stdout.flush()

# if fold_metrics_df is not an empty dataframe, print mean and std of metrics
if not fold_metrics_df.empty:
    print("=" * 80)
    # calculate mean and std of metrics for all folds
    mean_auc = fold_metrics_df['AUC'].mean()
    std_auc = fold_metrics_df['AUC'].std()
    mean_accuracy = fold_metrics_df['Accuracy'].mean()
    std_accuracy = fold_metrics_df['Accuracy'].std()
    mean_balanced_accuracy = fold_metrics_df['Bal_accuracy'].mean()
    std_balanced_accuracy = fold_metrics_df['Bal_accuracy'].std()
    mean_precision = fold_metrics_df['Precision'].mean()
    std_precision = fold_metrics_df['Precision'].std()
    mean_recall = fold_metrics_df['Sensitivity'].mean()
    std_recall = fold_metrics_df['Sensitivity'].std()
    mean_f1 = fold_metrics_df['F1-score'].mean()
    std_f1 = fold_metrics_df['F1-score'].std()
    mean_specificity = fold_metrics_df['Specificity'].mean()
    std_specificity = fold_metrics_df['Specificity'].std()
    mean_NPV = fold_metrics_df['NPV'].mean()
    std_NPV = fold_metrics_df['NPV'].std()
    # print metrics
    print("Mean AUC:", mean_auc, "Std AUC:", std_auc)
    print("Mean Accuracy:", mean_accuracy, "Std Accuracy:", std_accuracy)
    print("Mean Balanced accuracy:", mean_balanced_accuracy, "Std Balanced accuracy:", std_balanced_accuracy)
    print("Mean Precision:", mean_precision, "Std Precision:", std_precision)
    print("Mean Recall/Sensitivity:", mean_recall, "Std Recall:", std_recall)
    print("Mean F1-score:", mean_f1, "Std F1-score:", std_f1)
    print("Mean Specificity:", mean_specificity, "Std Specificity:", std_specificity)
    print("Mean NPV:", mean_NPV, "Std NPV:", std_NPV)
sys.stdout.close()