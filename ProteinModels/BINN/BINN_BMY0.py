import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import StratifiedKFold, train_test_split
# add path to binn in sys /home/ubuntu/tenerife/data/ZZ_githubRepos/BINN
sys.path.append('/home/ubuntu/tenerife/data/ZZ_githubRepos/BINN')
from binn import BINN, BINNDataLoader, BINNTrainer, BINNExplainer
from binn.plot.network import visualize_binn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch

np.set_printoptions(precision=3)
path_to_save_results = '/home/ubuntu/tenerife/data/LungAmbition/Proteinas/ResultsBaselines_BMY0/BINN'
path_to_save_plots = os.path.join(path_to_save_results, 'Plots')
n_folds=3
path_to_folds_csv = f'/home/ubuntu/tenerife/data/ZZ_githubRepos/LungAmbition/Data_stratified_split/folds-def_{n_folds}folds'
keep_false_positives_as_separate_test = True
# create path_to_save_plots if it does not exist
if not os.path.exists(path_to_save_plots):
    os.makedirs(path_to_save_plots)
name_file= f"BINN_proteins_14march25_5layers_500ep_drop02_{n_folds}folds"
if keep_false_positives_as_separate_test:
    name_file= name_file + "_keep_FP_separated_test"
sys.stdout=open(os.path.join(path_to_save_results, "run_out_"+name_file+".txt"),'w')
# save prints in a txt file
original_stdout = sys.stdout

df_merged = pd.read_csv('df_merged.csv') # read data in df_merged
# filter df_merged by GroupUpdated to keep only Lung_Cancer, Benign_Nodules and False_Positive
df_merged = df_merged[df_merged['GroupUpdated'].isin(['Lung_Cancer', 'Benign_Nodules', 'False_Positive'])]
columns_to_drop = ['ID_imagingData','Cancer_Status',
                            'TimeYears_CT_blood','TimeMonths_CT_blood',
                            'Age','Sex','Smoking_Category',
                            'Stage_category','NRRD_File','SEG_Files', 'GroupUpdated']

df_merged = df_merged.drop(columns=columns_to_drop)
df1=pd.read_csv('OlinkCode_UniprotID.csv')
for column in df_merged.columns:
    if column == 'ID_proteinData' or column == 'Group':
        continue
    if column in df1['Assay'].values:
        df_merged = df_merged.rename(columns
                            =dict(zip(df1['Assay'], df1['Uniprot ID'])))
    else:
        # print(column)
        # drop column from df_merged
        df_merged = df_merged.drop(columns=column)
feature_name_inverse_mapping = dict(zip(df1['Uniprot ID'], df1['Assay']))
if keep_false_positives_as_separate_test:
    y_false_positives = df_merged[df_merged['Group'] == 'False_Positive']['Group']
    # convert label to 1
    y_false_positives = y_false_positives.replace({'False_Positive': 0})
    ID_false_positives = df_merged[df_merged['Group'] == 'False_Positive']['ID_proteinData']
    # create list to store wrong predicted false positives
    list_ID_wrong_predicted_false_positives = []
    test_false_positives=df_merged[df_merged['Group'] == 'False_Positive']
    X_false_positives = df_merged[df_merged['Group'] == 'False_Positive'].drop(columns=['ID_proteinData', 'Group'])
    # drop in df_cur rows where Group is False_Positive
    df_merged = df_merged[df_merged['Group'].isin(['Lung_Cancer', 'Benign_Nodules'])]
    print("Number of false positives:", X_false_positives.shape[0])
    # save false_positive_metrics in df
    false_positive_metrics = pd.DataFrame(columns=['AUC', 'Balanced_accuracy', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
else:
    df_merged = df_merged[df_merged['Group'].isin(['Lung_Cancer', 'Benign_Nodules', 'False_Positive'])]
    # print shape of excel
    print("Dimensions excel after dropping rows where Group is False_Positive:", df_merged.shape)
# get column to predict Group
y_target = df_merged['Group']
# encode y
y_target = y_target.replace({'Lung_Cancer': 1, 'Benign_Nodules': 0, 'False_Positive': 0})

# save best metrics for each fold
fold_metrics_df = pd.DataFrame(columns=['Fold', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score'])
# save top features for each fold
top_features_df_list = []

for fold in range(0, n_folds):
    print("=" * 80)
    print(f"Fold {fold + 1}:")
    # read train, test and val indices for each fold
    fold_data = pd.read_csv(os.path.join(path_to_folds_csv, f'id2splitfold_{fold}.csv'))
    # get corresponding ID_proteinData for split (train, test, val) in fold_data and read then in df_cur, 
    # loc the rows in df_merged with ID_proteinData in df_cur
    train_index = fold_data[fold_data['split'] == 'train']['ID_proteinData']
    test_index = fold_data[fold_data['split'] == 'test']['ID_proteinData']
    val_index = fold_data[fold_data['split'] == 'val']['ID_proteinData']
    # get first train, text, val, then split into X_train, X_test, X_val and y_train, y_test, y_val
    train = df_merged.loc[df_merged['ID_proteinData'].isin(train_index)]
    test = df_merged.loc[df_merged['ID_proteinData'].isin(test_index)]
    val = df_merged.loc[df_merged['ID_proteinData'].isin(val_index)]
    # join train and val together in train_val
    train_val = pd.concat([train, val], axis=0)
    X_train = train.drop(columns=['ID_proteinData', 'Group'])
    y_train = y_target.loc[y_target.index.isin(X_train.index)]
    X_test = test.drop(columns=['ID_proteinData', 'Group'])
    y_test = y_target.loc[y_target.index.isin(X_test.index)]
    X_val = val.drop(columns=['ID_proteinData', 'Group'])
    y_val = y_target.loc[y_target.index.isin(X_val.index)]
    
    print("Training data:", X_train.shape, y_train.shape, "Benign patients train:", len(y_train[y_train == 0]), "Lung cancer patients train:", len(y_train[y_train == 1]))
    print("Validation data:", X_val.shape, y_val.shape, "Benign patients val:", len(y_val[y_val == 0]), "Lung cancer patients val:", len(y_val[y_val == 1]))
    print("Test data:", X_test.shape, y_test.shape, "Benign patients test:", len(y_test[y_test == 0]), "Lung cancer patients test:", len(y_test[y_test == 1]))
    print("Number of samples per class in y_train")
    print(y_train.value_counts())
    print("Number of samples per class in y_test")
    print(y_test.value_counts())
    print("Number of samples per class in y_val")
    print(y_val.value_counts())
    # join X_val in X_train and y_val in y_train
    X_train = pd.concat([X_train, X_val], axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)
    print("Training data after joining with validation data:", X_train.shape, y_train.shape, "Benign patients train:", len(y_train[y_train == 0]), "Lung cancer patients train:", len(y_train[y_train == 1]))
    # convert to str the columns of X_train, X_test
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    # scale the data using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Create the data matrix with Protein column (Transposed for BINN compatibility)
    train_data_matrix = pd.DataFrame(X_train, columns=train_val.drop(columns=['ID_proteinData', 'Group']).columns, index=train_val['ID_proteinData'])
    train_data_matrix = train_data_matrix.T
    train_data_matrix.insert(0, 'Protein', train_data_matrix.index)
    train_data_matrix.reset_index(drop=True, inplace=True)
    # print("Data Matrix Columns:", train_data_matrix.columns)
    print("Data Matrix Shape:", train_data_matrix.shape)
    # print("Data Matrix Sample:", train_data_matrix.head())
    # test data matrix
    test_data_matrix = pd.DataFrame(X_test, columns=test.drop(columns=['ID_proteinData', 'Group']).columns, index=test['ID_proteinData'])
    test_data_matrix = test_data_matrix.T
    test_data_matrix.insert(0, 'Protein', test_data_matrix.index)
    test_data_matrix.reset_index(drop=True, inplace=True)
    # print("Test Data Matrix Columns:", test_data_matrix.columns)
    print("Test Data Matrix Shape:", test_data_matrix.shape)
    
    # Create the design matrix
    train_design_matrix = pd.DataFrame({'sample': train_val['ID_proteinData'].values, 'group': y_train})
    # print("Design matrix head:", train_design_matrix.head())
    print("Design matrix shape:", train_design_matrix.shape)
    test_design_matrix = pd.DataFrame({'sample': test['ID_proteinData'].values, 'group': y_test})
    # print("Test Design matrix head:", test_design_matrix.head())
    print("Test Design matrix shape:", test_design_matrix.shape)
    
    # Initialize BINN
    binn = BINN(data_matrix=train_data_matrix, network_source="reactome", n_layers=5, dropout=0.2)
    print(binn)

    ## Initialize DataLoader
    binn_dataloader = BINNDataLoader(binn)

    # Create DataLoaders
    train_dataloaders = binn_dataloader.create_dataloaders(
        data_matrix=train_data_matrix,
        design_matrix=train_design_matrix,
        feature_column="Protein",
        group_column="group",
        sample_column="sample",
        batch_size=64,
        validation_split=0.2,
    )
    test_dataloaders = binn_dataloader.create_dataloaders(
        data_matrix=test_data_matrix,
        design_matrix=test_design_matrix,
        feature_column="Protein",
        group_column="group",
        sample_column="sample",
        batch_size=64,
        validation_split=0,
    )
    # Train the model only on training data
    trainer = BINNTrainer(binn)
    trainer.fit(train_dataloaders, num_epochs=500)
    train_dataloaders = binn_dataloader.create_dataloaders(
        data_matrix=train_data_matrix,
        design_matrix=train_design_matrix,
        feature_column="Protein",
        group_column="group",
        sample_column="sample",
        batch_size=64,
        validation_split=0,
    )
    # Predict on all training data
    y_train_pred_logits = trainer.predict(train_dataloaders["train"])
    # print("Logits predicted on training data", y_train_pred_logits)
    y_train_pred_prob = torch.softmax(torch.tensor(y_train_pred_logits), dim=1).numpy()
    # print("Probabilities training data:", y_train_pred_prob)
    y_train_pred = np.argmax(y_train_pred_prob, axis=1)
    # print("Predictions training data:", y_train_pred)
    # calculate metrics   
    auc_train = roc_auc_score(y_train, y_train_pred_prob[:, 1])
    balanced_accuracy_train = balanced_accuracy_score(y_train, y_train_pred)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred, average='binary')
    recall_train = recall_score(y_train, y_train_pred, average='binary')
    f1_train = f1_score(y_train, y_train_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred, labels=[0, 1]).ravel()
    NPV_train=tn/(tn+fn)
    specificity_train=tn/(tn+fp)
    print(f"Metrics in train for fold: {fold+1}")
    print(f"AUC: {auc_train}")
    print(f"Balanced accuracy: {balanced_accuracy_train}")
    print(f"Accuracy: {accuracy_train}")
    print(f"Precision: {precision_train}")
    print(f"Recall: {recall_train}")
    print(f"F1-score: {f1_train}")
    print("Specificity:", specificity_train)
    print("NPV:", NPV_train)
    print("=" * 80)
    # Predict on test data
    y_pred_logits = trainer.predict(test_dataloaders["train"])
    # print("Logits predicted on test data", y_pred_logits)
    y_pred_prob = torch.softmax(torch.tensor(y_pred_logits), dim=1).numpy()
    # print("Probabilities test data:", y_pred_prob)
    y_pred = np.argmax(y_pred_prob, axis=1)
    # print("Predictions test data:", y_pred)
    # calculate metrics
    test_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='binary')
    test_recall = recall_score(y_test, y_pred, average='binary')
    test_f1 = f1_score(y_test, y_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    test_NPV=tn/(tn+fn)
    test_specificity=tn/(tn+fp)
    test_ppv = tp/(tp+fp)
    # save metrics
    fold_metrics_df = pd.concat([fold_metrics_df, pd.DataFrame([{'Fold': fold, 'AUC': test_auc, 
        'Balanced_accuracy': test_balanced_accuracy, 'Accuracy': test_accuracy, 'Specificity': test_specificity, 'NPV': test_NPV, 'Precision': test_precision, 'Recall': test_recall, 'F1-score': test_f1,
        'PPV': test_ppv}])], ignore_index=True)
    # print("Probabilities test:", y_pred_prob)
    print(f"Metrics in test for fold: {fold+1}")
    print(f"AUC: {test_auc}")
    print(f"Balanced accuracy: {test_balanced_accuracy}")
    print(f"Accuracy: {test_accuracy}")
    print(f"Precision: {test_precision}")
    print(f"Recall: {test_recall}")
    print(f"F1-score: {test_f1}")
    print("Specificity:", test_specificity)
    print("NPV:", test_NPV)
    print("PPV:", test_ppv)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    # evaluate separately model on false positives:
    if keep_false_positives_as_separate_test:
        X_false_positives.columns = X_false_positives.columns.astype(str)
        # scale the data using StandardScaler
        X_false_positives_scaled = scaler.transform(X_false_positives)
        # create data matrix for false positives
        false_positives_data_matrix = pd.DataFrame(X_false_positives_scaled, columns=test_false_positives.drop(columns=['ID_proteinData', 'Group']).columns, index=test_false_positives['ID_proteinData'])
        false_positives_data_matrix = false_positives_data_matrix.T
        false_positives_data_matrix.insert(0, 'Protein', false_positives_data_matrix.index)
        false_positives_data_matrix.reset_index(drop=True, inplace=True)
        print("False Positives Data Matrix Columns:", false_positives_data_matrix.columns)
        print("False Positives Data Matrix Shape:", false_positives_data_matrix.shape)
        # Create the design matrix for false positives
        false_positives_design_matrix = pd.DataFrame({'sample': test_false_positives['ID_proteinData'].values, 'group': y_false_positives})
        print("False Positives Design matrix head:", false_positives_design_matrix.head())
        print("False Positives Design matrix shape:", false_positives_design_matrix.shape)
        # Create DataLoader for false positives
        false_positives_dataloaders = binn_dataloader.create_dataloaders(
            data_matrix=false_positives_data_matrix,
            design_matrix=false_positives_design_matrix,
            feature_column="Protein",
            group_column="group",
            sample_column="sample",
            batch_size=64,
            validation_split=0,
        )
        # predict on false positives
        y_false_positives_pred_logits = trainer.predict(false_positives_dataloaders["train"])
        # print("Logits predicted on false positives", y_false_positives_pred_logits)
        y_false_positives_pred_prob = torch.softmax(torch.tensor(y_false_positives_pred_logits), dim=1).numpy()
        # print("Probabilities false positives:", y_false_positives_pred_prob)
        y_false_positives_pred = np.argmax(y_false_positives_pred_prob, axis=1)
        # print("Predictions false positives:", y_false_positives_pred)
        # calculate metrics
        false_positives_accuracy = accuracy_score(y_false_positives, y_false_positives_pred)
        false_positives_precision = precision_score(y_false_positives, y_false_positives_pred, average='binary')
        false_positives_recall = recall_score(y_false_positives, y_false_positives_pred, average='binary')
        false_positives_f1 = f1_score(y_false_positives, y_false_positives_pred, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_false_positives, y_false_positives_pred, labels=[0, 1]).ravel()
        # false_positives_NPV=tn/(tn+fn)
        # false_positives_specificity=tn/(tn+fp)
        print(f"Metrics in False Positives for fold: {fold+1}")
        print(f"Accuracy: {false_positives_accuracy}")
        print(f"Precision: {false_positives_precision}")
        print(f"Recall: {false_positives_recall}")
        print(f"F1-score: {false_positives_f1}")
        # print("Specificity:", false_positives_specificity)
        # print("NPV:", false_positives_NPV)
        print("Confusion matrix:")
        print(confusion_matrix(y_false_positives, y_false_positives_pred))
        # save metrics
        false_positive_metrics = pd.concat([false_positive_metrics, pd.DataFrame([{'Accuracy': false_positives_accuracy, 'Precision': false_positives_precision, 'Recall': false_positives_recall, 'F1-score': false_positives_f1}])], ignore_index=True)
        # recover ID of wrong predicted patients
        ID_false_positives_wrong_predicted = ID_false_positives[y_false_positives != y_false_positives_pred]
        print("Wrong predicted False Positives:", ID_false_positives_wrong_predicted)
        list_ID_wrong_predicted_false_positives.append(ID_false_positives_wrong_predicted)
    # calculate most relevant features for the model
    # Explain model predictions on test data
    explainer = BINNExplainer(binn)
    single_explanations = explainer.explain_single(test_dataloaders, normalization_method="subgraph")
    
    single_explanations_sorted = single_explanations.sort_values(by='importance', ascending=False)
    unique_features = single_explanations_sorted.drop_duplicates(subset=['source_node'])  # Keep only unique features
    top_features = unique_features.head(100)
        
    # Recover original feature names
    top_features['Feature'] = top_features['source_node'].map(feature_name_inverse_mapping).fillna(top_features['source_node'])

    # Save top feature importances for this fold
    fold_feature_importances = pd.DataFrame({
        'Feature': top_features['Feature'].values,
        'Importance': top_features['importance'].values,
        'Rank': range(1, len(top_features) + 1)
    })
    fold_feature_importances['Fold'] = fold

    # Append this fold's DataFrame to the list for further analysis
    top_features_df_list.append(fold_feature_importances)

    # Plot most important features
    plt.figure()
    plt.title("Feature importances fold " + str(fold + 1))
    plt.bar(range(30), top_features['importance'].values[:30])
    plt.xticks(range(30), top_features['source_node'].values[:30], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save_plots, f"Feature_importances_{name_file}_fold_{fold+1}.png"))
    plt.close()

    # Visualize interpreted network
    layer_specific_top_n = {"0": 10, "1": 7, "2": 5, "3": 5, "4": 5}
    plt_network = visualize_binn(single_explanations, top_n=layer_specific_top_n, plot_size=(20, 15), 
                                 sink_node_size=500, node_size_scaling=200, edge_width=1, node_cmap="coolwarm",
                                 pathways_mapping="reactome", input_entity_mapping="uniprot")
    plt_network.title("Interpreted network fold " + str(fold + 1))
    plt_network.savefig(os.path.join(path_to_save_plots, f"Interpreted_network_{name_file}_fold_{fold+1}.png"))
    plt.close()
# print mean and std of each metric
print("=" * 80)
# calculate mean and std of metrics for all folds
mean_auc = fold_metrics_df['AUC'].mean()
std_auc = fold_metrics_df['AUC'].std()
mean_accuracy = fold_metrics_df['Accuracy'].mean()
std_accuracy = fold_metrics_df['Accuracy'].std()
mean_balanced_accuracy = fold_metrics_df['Balanced_accuracy'].mean()
std_balanced_accuracy = fold_metrics_df['Balanced_accuracy'].std()
mean_precision = fold_metrics_df['Precision'].mean()
std_precision = fold_metrics_df['Precision'].std()
mean_recall = fold_metrics_df['Recall'].mean()
std_recall = fold_metrics_df['Recall'].std()
mean_f1 = fold_metrics_df['F1-score'].mean()
std_f1 = fold_metrics_df['F1-score'].std()
mean_specificity = fold_metrics_df['Specificity'].mean()
std_specificity = fold_metrics_df['Specificity'].std()
mean_NPV = fold_metrics_df['NPV'].mean()
std_NPV = fold_metrics_df['NPV'].std()
mean_PPV = fold_metrics_df['PPV'].mean()
std_PPV = fold_metrics_df['PPV'].std()
# print metrics
print("Mean AUC:", mean_auc, "Std AUC:", std_auc)
print("Mean Accuracy:", mean_accuracy, "Std Accuracy:", std_accuracy)
print("Mean Balanced accuracy:", mean_balanced_accuracy, "Std Balanced accuracy:", std_balanced_accuracy)
print("Mean Precision:", mean_precision, "Std Precision:", std_precision)
print("Mean Recall:", mean_recall, "Std Recall:", std_recall)
print("Mean F1-score:", mean_f1, "Std F1-score:", std_f1)
print("Mean Specificity:", mean_specificity, "Std Specificity:", std_specificity)
print("Mean NPV:", mean_NPV, "Std NPV:", std_NPV)
print("Mean PPV:", mean_PPV, "Std PPV:", std_PPV)

# calculate mean and std metrics for false positives
if keep_false_positives_as_separate_test:
    print("=" * 80)
    print("Global metrics for false positives")
    # calculate mean and std of metrics for all folds
    mean_accuracy = false_positive_metrics['Accuracy'].mean()
    std_accuracy = false_positive_metrics['Accuracy'].std()
    mean_precision = false_positive_metrics['Precision'].mean()
    std_precision = false_positive_metrics['Precision'].std()
    mean_recall = false_positive_metrics['Recall'].mean()
    std_recall = false_positive_metrics['Recall'].std()
    mean_f1 = false_positive_metrics['F1-score'].mean()
    std_f1 = false_positive_metrics['F1-score'].std()
    # mean_specificity = false_positive_metrics['Specificity'].mean()
    # std_specificity = false_positive_metrics['Specificity'].std()
    # mean_NPV = false_positive_metrics['NPV'].mean()
    # std_NPV = false_positive_metrics['NPV'].std()
    # print metrics
    print("Mean Accuracy False Positives:", mean_accuracy, "Std Accuracy False Positives:", std_accuracy)
    print("Mean Precision False Positives:", mean_precision, "Std Precision False Positives:", std_precision)
    print("Mean Recall False Positives:", mean_recall, "Std Recall False Positives:", std_recall)
    print("Mean F1-score False Positives:", mean_f1, "Std F1-score False Positives:", std_f1)
    # print("Mean Specificity False Positives:", mean_specificity, "Std Specificity False Positives:", std_specificity)
    # print("Mean NPV False Positives:", mean_NPV, "Std NPV False Positives:", std_NPV)

    # count number of ocurrences of each false positive in list_ID_wrong_predicted_false_positives
    from collections import Counter
    counter = Counter([item for sublist in list_ID_wrong_predicted_false_positives for item in sublist])
    print("Number of times each False Positive was wrongly predicted:")
    print(counter)

# Concatenate all fold DataFrames for easier comparison
all_folds_df = pd.concat(top_features_df_list)

# Count how many times each feature appears across all folds
feature_frequencies = all_folds_df['Feature'].value_counts()

# Find features common to 3, 4, and 5 folds
features_in_3_folds = feature_frequencies[feature_frequencies == 3].index
features_in_4_folds = feature_frequencies[feature_frequencies == 4].index
features_in_5_folds = feature_frequencies[feature_frequencies == 5].index

# Create summary DataFrames for each group
common_in_3_folds = all_folds_df[all_folds_df['Feature'].isin(features_in_3_folds)].sort_values(['Feature', 'Fold'])
common_in_4_folds = all_folds_df[all_folds_df['Feature'].isin(features_in_4_folds)].sort_values(['Feature', 'Fold'])
common_in_5_folds = all_folds_df[all_folds_df['Feature'].isin(features_in_5_folds)].sort_values(['Feature', 'Fold'])

# Print out the summary DataFrames
if n_folds==3:
    print("Features common in exactly 3 folds:\n", common_in_3_folds)
else:
    print("\nFeatures common in exactly 4 folds:\n", common_in_4_folds)
    print("\nFeatures common in all 5 folds:\n", common_in_5_folds)

sys.stdout.close()