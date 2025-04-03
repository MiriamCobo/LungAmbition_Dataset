import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from collections import Counter

np.set_printoptions(precision=3)
path_to_save_results = '/home/ubuntu/tenerife/data/LungAmbition/Proteinas/ResultsBaselines_BMY0/XGB'
path_to_save_plots = os.path.join(path_to_save_results, 'Plots')
n_folds=3
path_to_folds_csv = f'/home/ubuntu/tenerife/data/ZZ_githubRepos/LungAmbition/Data_stratified_split/folds-def_{n_folds}folds'
keep_false_positives_as_separate_test = True
# create path_to_save_plots if it does not exist
if not os.path.exists(path_to_save_plots):
    os.makedirs(path_to_save_plots)
name_file= f"XGB_proteins_14march25_{n_folds}folds"
if keep_false_positives_as_separate_test:
    name_file= name_file + "_keep_false_positives_as_separate_test"
sys.stdout=open(os.path.join(path_to_save_results, "run_out_"+name_file+".txt"),'w')
# save prints in a txt file
original_stdout = sys.stdout

df_merged = pd.read_csv('/home/ubuntu/tenerife/data/LungAmbition/Excels_merged/LungAmbitionMergedAllGroupUpdated3mar2025.csv')
# filter df_merged by GroupUpdated to keep only Lung_Cancer, Benign_Nodules and False_Positive
df_merged = df_merged[df_merged['GroupUpdated'].isin(['Lung_Cancer', 'Benign_Nodules', 'False_Positive'])]
columns_to_drop = ['ID_imagingData','Cancer_Status','TimeYears_blood','TimeMonths_blood',
                            'TimeYears_CT_blood','TimeMonths_CT_blood','Diff_Diag_Blood_TimeYears','LastFollow_upTimeYears',
                            'Age','Sex','Body_mass_index','Smoking_status','Years_smoked','Smoking_pack_years',
                            'Family_history_lung_cancer','Personal_history_cancer','Stage_category','NRRD_File','SEG_Files', 'GroupUpdated']

df_merged = df_merged.drop(columns=columns_to_drop)
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
    false_positive_metrics = pd.DataFrame(columns=['AUC', 'Balanced_accuracy', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'NPV', 'Specificity'])
else:
    df_merged = df_merged[df_merged['Group'].isin(['Lung_Cancer', 'Benign_Nodules', 'False_Positive'])]
    # print shape of excel
    print("Dimensions excel after dropping rows where Group is False_Positive:", df_merged.shape)
# get column to predict Group
y_target = df_merged['Group']
# encode y
y_target = y_target.replace({'Lung_Cancer': 1, 'Benign_Nodules': 0, 'False_Positive': 0})

# save best metrics for each fold
fold_metrics_df = pd.DataFrame(columns=['Fold', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score', 'PPV'])
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
    # scale the data using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # train xgboost classifier, first find the best hyperparameters through a grid search cv,
    # then train the model with the best hyperparameters
    # parameters to search
    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [2, 3],
        'learning_rate': [0.01, 0.005, 0.001],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'gamma': [0.1],
        'alpha': [0.3],
        'lambda': [2],
    }
    class_counts = Counter(y_train)  # y_train should be your target labels
    num_benign = class_counts[0]  # Count of benign cases
    num_malignant = class_counts[1]  # Count of malignant cases
    # Compute scale_pos_weight
    scale_pos_weight = num_benign / num_malignant
    print(f"scale_pos_weight: {scale_pos_weight}")
    # Initialize the XGBClassifier
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=1, use_label_encoder=False)
    # grid search cv
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    # best hyperparameters
    print("Best hyperparameters:")
    print(grid_search.best_params_)
    best_xgb = grid_search.best_estimator_
    # train the model with the best hyperparameters
    best_xgb.fit(X_train, y_train)
    # print metrics in train
    y_train_pred_prob = best_xgb.predict_proba(X_train)
    y_train_pred = best_xgb.predict(X_train)
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
    # predict probabilities
    y_pred_prob = best_xgb.predict_proba(X_test)
    # predict classes
    y_pred = best_xgb.predict(X_test)
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
        if not isinstance(X_false_positives, pd.DataFrame):
            X_false_positives = pd.DataFrame(X_false_positives, columns=df_merged.drop(columns=['ID_proteinData', 'Group']).columns)
        X_false_positives = scaler.transform(X_false_positives)
        y_false_positives_pred_prob = best_xgb.predict_proba(X_false_positives)
        y_false_positives_pred = best_xgb.predict(X_false_positives)
        print("Predicted probabilities false positives: ", y_false_positives_pred_prob)
        false_positives_accuracy = accuracy_score(y_false_positives, y_false_positives_pred)
        false_positives_precision = precision_score(y_false_positives, y_false_positives_pred, average='binary')
        false_positives_recall = recall_score(y_false_positives, y_false_positives_pred, average='binary')
        false_positives_f1 = f1_score(y_false_positives, y_false_positives_pred, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_false_positives, y_false_positives_pred, labels=[0, 1]).ravel()
        false_positives_NPV=tn/(tn+fn)
        false_positives_specificity=tn/(tn+fp)
        print(f"Metrics in False Positives for fold: {fold+1}")
        print(f"Accuracy: {false_positives_accuracy}")
        print(f"Precision: {false_positives_precision}")
        print(f"Recall: {false_positives_recall}")
        print(f"F1-score: {false_positives_f1}")
        print("Specificity:", false_positives_specificity)
        print("NPV:", false_positives_NPV)
        print("Confusion matrix:")
        print(confusion_matrix(y_false_positives, y_false_positives_pred))
        # save metrics
        false_positive_metrics = pd.concat([false_positive_metrics, pd.DataFrame([{'Accuracy': false_positives_accuracy, 'Specificity': false_positives_specificity, 'NPV': false_positives_NPV, 'Precision': false_positives_precision, 'Recall': false_positives_recall, 'F1-score': false_positives_f1}])], ignore_index=True)
        # recover ID of wrong predicted patients
        ID_false_positives_wrong_predicted = ID_false_positives[y_false_positives != y_false_positives_pred]
        print("Wrong predicted False Positives:", ID_false_positives_wrong_predicted)
        list_ID_wrong_predicted_false_positives.append(ID_false_positives_wrong_predicted)
    # Calculate feature importances for the best XGBoost model
    feature_importances = best_xgb.feature_importances_

    # Sort feature indices by importance in descending order
    indices = np.argsort(feature_importances)[::-1]

    # Select the top 50 features and organize them into a DataFrame
    fold_feature_importances = pd.DataFrame({
        'Feature': train.columns[indices][:100],
        'Importance': feature_importances[indices][:100],
        'Rank': range(1, 101)
    })
    fold_feature_importances['Fold'] = fold  # Update this variable appropriately
    top_features_df_list.append(fold_feature_importances)

    # Plot the 30 most important features
    plt.figure()
    plt.title(f"Feature importances fold {fold + 1}")
    plt.bar(range(30), feature_importances[indices][:30])
    plt.xticks(range(30), train.columns[indices][:30], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save_plots, f"Feature_importances_{name_file}_fold_{fold+1}.png"))

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
# print metrics
print("Mean AUC:", mean_auc, "Std AUC:", std_auc)
print("Mean Accuracy:", mean_accuracy, "Std Accuracy:", std_accuracy)
print("Mean Balanced accuracy:", mean_balanced_accuracy, "Std Balanced accuracy:", std_balanced_accuracy)
print("Mean Precision:", mean_precision, "Std Precision:", std_precision)
print("Mean Recall:", mean_recall, "Std Recall:", std_recall)
print("Mean F1-score:", mean_f1, "Std F1-score:", std_f1)
print("Mean Specificity:", mean_specificity, "Std Specificity:", std_specificity)
print("Mean NPV:", mean_NPV, "Std NPV:", std_NPV)

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
    mean_specificity = false_positive_metrics['Specificity'].mean()
    std_specificity = false_positive_metrics['Specificity'].std()
    mean_NPV = false_positive_metrics['NPV'].mean()
    std_NPV = false_positive_metrics['NPV'].std()
    # print metrics
    print("Mean Accuracy False Positives:", mean_accuracy, "Std Accuracy False Positives:", std_accuracy)
    print("Mean Precision False Positives:", mean_precision, "Std Precision False Positives:", std_precision)
    print("Mean Recall False Positives:", mean_recall, "Std Recall False Positives:", std_recall)
    print("Mean F1-score False Positives:", mean_f1, "Std F1-score False Positives:", std_f1)
    print("Mean Specificity False Positives:", mean_specificity, "Std Specificity False Positives:", std_specificity)

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