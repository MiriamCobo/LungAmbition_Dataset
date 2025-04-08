import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


np.set_printoptions(precision=3)
path_to_save_results = '/home/ubuntu/tenerife/data/LungAmbition/Proteinas/ResultsBaselines_BMY0/LASSO'
path_to_save_plots = os.path.join(path_to_save_results, 'Plots')
n_folds=3
path_to_folds_csv = f'/home/ubuntu/tenerife/data/ZZ_githubRepos/LungAmbition/Data_stratified_split/folds-def_{n_folds}folds'
keep_false_positives_as_separate_test = True
# create path_to_save_plots if it does not exist
if not os.path.exists(path_to_save_plots):
    os.makedirs(path_to_save_plots)
name_file= f"LASSO_proteins_14march25_{n_folds}folds"
if keep_false_positives_as_separate_test:
    name_file= name_file + "_keep_false_positives_as_separate_test"
sys.stdout=open(os.path.join(path_to_save_results, "run_out_"+name_file+".txt"),'w')
# save prints in a txt file
original_stdout = sys.stdout

df_merged = pd.read_csv('/home/ubuntu/tenerife/data/LungAmbition/Excels_merged/LungAmbitionMergedAllGroupUpdated3mar2025.csv')
# filter df_merged by GroupUpdated to keep only Lung_Cancer, Benign_Nodules and False_Positive
df_merged = df_merged[df_merged['GroupUpdated'].isin(['Lung_Cancer', 'Benign_Nodules', 'False_Positive'])]
# retrieve IDs
IDs = df_merged['ID_patient']
columns_to_drop = ['ID_patient','Cancer_Status',
                            'TimeYears_CT_blood','TimeMonths_CT_blood',
                            'Age','Sex','Smoking_Category',
                            'Stage_category','NRRD_File','SEG_Files', 'GroupUpdated']

df_merged = df_merged.drop(columns=columns_to_drop)
if keep_false_positives_as_separate_test:
    y_false_positives = df_merged[df_merged['Group'] == 'False_Positive']['Group']
    # convert label to 1
    y_false_positives = y_false_positives.replace({'False_Positive': 0})
    ID_false_positives = df_merged[df_merged['Group'] == 'False_Positive']['ID_patient']
    # create list to store wrong predicted false positives
    list_ID_wrong_predicted_false_positives = []
    X_false_positives = df_merged[df_merged['Group'] == 'False_Positive'].drop(columns=['ID_patient', 'Group'])
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
    # get corresponding ID_patient for split (train, test, val) in fold_data and read then in df_cur, 
    # loc the rows in df_merged with ID_patient in df_cur
    train_index = fold_data[fold_data['split'] == 'train']['ID_patient']
    test_index = fold_data[fold_data['split'] == 'test']['ID_patient']
    val_index = fold_data[fold_data['split'] == 'val']['ID_patient']
    # get first train, text, val, then split into X_train, X_test, X_val and y_train, y_test, y_val
    train = df_merged.loc[df_merged['ID_patient'].isin(train_index)]
    test = df_merged.loc[df_merged['ID_patient'].isin(test_index)]
    val = df_merged.loc[df_merged['ID_patient'].isin(val_index)]
    X_train = train.drop(columns=['ID_patient', 'Group']).copy()
    y_train = y_target.loc[y_target.index.isin(X_train.index)]
    X_test = test.drop(columns=['ID_patient', 'Group']).copy()
    y_test = y_target.loc[y_target.index.isin(X_test.index)]
    X_val = val.drop(columns=['ID_patient', 'Group']).copy()
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
    # retrieve IDs to ensure correctness
    IDs_train = IDs.loc[IDs.index.isin(X_train.index)]
    IDs_test = IDs.loc[IDs.index.isin(X_test.index)]
    IDs_val = IDs.loc[IDs.index.isin(X_val.index)]
    # print IDs in train, test, val
    # print("IDs train:", IDs_train)
    # print("IDs test:", IDs_test)
    # print("IDs val:", IDs_val)

    # join X_val in X_train and y_val in y_train
    X_train = pd.concat([X_train, X_val], axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)
    print("Training data after joining with validation data:", X_train.shape, y_train.shape, "Benign patients train:", len(y_train[y_train == 0]), "Lung cancer patients train:", len(y_train[y_train == 1]))
    # scale the data using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Cross-validation to find the best C (inverse of regularization strength)
    logreg_cv = LogisticRegressionCV(cv=3, penalty='l1', Cs=[75, 80], solver='liblinear', max_iter=5000, class_weight='balanced', random_state=1)
    logreg_cv.fit(X_train, y_train)
    best_C = logreg_cv.C_[0]
    print(f"Best C: {best_C}")

    # Initialize variables to track feature selection frequency
    feature_counts = Counter()

    # Resampling and Logistic Regression with the best C
    for _ in range(500):
        X_resample, _, y_resample, _ = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=np.random.randint(0, 10000))
        logreg = LogisticRegression(C=best_C, penalty='l1', solver='liblinear', max_iter=5000, class_weight='balanced', random_state=1)
        logreg.fit(X_resample, y_resample)
        non_zero_features = np.where(logreg.coef_[0] != 0)[0]
        feature_counts.update(non_zero_features)

    # Determine informative markers (features selected in at least 50% of resamples)
    selected_features = [feature for feature, count in feature_counts.items() if count >= 250]
    print(f"Number of selected features: {len(selected_features)}")
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=train.drop(columns=['ID_patient', 'Group']).columns)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=test.drop(columns=['ID_patient', 'Group']).columns)
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    print("Selected features:", X_train.columns[selected_features])
    print("X_train_selected shape", X_train_selected.shape)
    print("X_test_selected shape", X_test_selected.shape)
    # Train final Logistic Regression model with the best C on selected features
    logreg_final = LogisticRegression(C=best_C, penalty='l1', solver='liblinear', max_iter=5000, class_weight = 'balanced', random_state=1)
    logreg_final.fit(X_train_selected, y_train)
    # l1 regularization for lasso, l2 for ridge

    # Predict probabilities and classes for the train set
    y_train_pred_prob = logreg_final.predict_proba(X_train_selected)[:, 1]
    y_train_pred = (y_train_pred_prob > 0.5).astype(int)

    auc_train = roc_auc_score(y_train, y_train_pred_prob)
    balanced_accuracy_train = balanced_accuracy_score(y_train, y_train_pred)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred, average='binary')
    recall_train = recall_score(y_train, y_train_pred, average='binary')
    f1_train = f1_score(y_train, y_train_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred, labels=[0, 1]).ravel()
    NPV_train=tn/(tn+fn)
    specificity_train=tn/(tn+fp)
    print(f"Metrics in train for fold: {fold+1}")
    # print(f"AUC: {auc_train}")
    print(f"Balanced accuracy: {balanced_accuracy_train}")
    print(f"Accuracy: {accuracy_train}")
    print(f"Precision: {precision_train}")
    print(f"Recall: {recall_train}")
    print(f"F1-score: {f1_train}")
    print("Specificity:", specificity_train)
    print("NPV:", NPV_train)
    print("=" * 80)
    # Predict probabilities and classes for the test set
    y_test_pred_prob = logreg_final.predict_proba(X_test_selected)[:, 1]
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)
    # calculate metrics
    test_auc = roc_auc_score(y_test, y_test_pred_prob)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='binary')
    test_recall = recall_score(y_test, y_test_pred, average='binary')
    test_f1 = f1_score(y_test, y_test_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()
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
    print(confusion_matrix(y_test, y_test_pred))
    # evaluate separately model on false positives:
    if keep_false_positives_as_separate_test:
        X_false_positives = scaler.transform(X_false_positives)
        if not isinstance(X_false_positives, pd.DataFrame):
            X_false_positives = pd.DataFrame(X_false_positives, columns=df_merged.drop(columns=['ID_patient', 'Group']).columns)
        X_false_positives_selected = X_false_positives.iloc[:, selected_features]
        y_false_positives_pred_prob = logreg_final.predict_proba(X_false_positives_selected)[:,1]
        y_false_positives_pred = (y_false_positives_pred_prob > 0.5).astype(int)
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
    
    # Calculate feature importances for LASSO model
    # Plot feature importance
    # Extract feature importance
    feature_importance = np.abs(logreg_final.coef_[0])
    # calculate number of total features in logreg_final
    total_features = len(logreg_final.coef_[0])
    feature_names = X_train.columns[selected_features]
    sorted_idx = np.argsort(feature_importance)[::-1]
    # Create a DataFrame for the top features
    fold_feature_importances = pd.DataFrame({
        'Feature': feature_names[sorted_idx][:total_features],
        'Importance': feature_importance[sorted_idx][:total_features],
        'Rank': range(1, total_features+1)
    })
    fold_feature_importances['Fold'] = fold  # Update this variable appropriately
    top_features_df_list.append(fold_feature_importances)

    plt.figure()
    plt.title(f"Feature importances fold {fold + 1}")
    plt.bar(range(total_features), feature_importance[sorted_idx][:total_features])
    plt.xticks(range(total_features), feature_names[sorted_idx][:total_features], rotation=90)
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
    print("Mean NPV False Positives:", mean_NPV, "Std NPV False Positives:", std_NPV)

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