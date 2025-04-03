import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sksurv.metrics import concordance_index_censored, brier_score, cumulative_dynamic_auc
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import RandomOverSampler
import shap
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.utils.class_weight import compute_sample_weight

seed = 1

# Define custom scoring function for the concordance index
def concordance_scorer(estimator, X, y):
    # Extract event indicator and time from the structured array
    event_indicator, event_time = y["Cancer_Status"], y["TimeYears_CT_blood"]
    
    # Predict risk scores
    risk_scores = estimator.predict(X)
    
    # Calculate concordance index
    concordance = concordance_index_censored(event_indicator, event_time, risk_scores)[0]
    return concordance

def brier_score_at_time(estimator, X, y, time_point=0):
    """Custom Brier Score metric at a specific time point for use in GridSearchCV."""
    surv_funcs = estimator.predict_survival_function(X)  # Get survival functions for the predictions
    test_estimates = np.array([fn(time_point) for fn in surv_funcs])  # Survival prob. at 1 year
    brier_score_1yr = brier_score(y_train, y, test_estimates, [time_point])[1][0]  # Brier Score at 1 year
    return -brier_score_1yr  # Negative for maximizing in GridSearchCV

np.set_printoptions(precision=3)
n_folds=3
path_to_save_results = '/home/ubuntu/tenerife/data/LungAmbition/Proteinas/ResultsBaselines_BM/RSF'
path_to_save_plots = os.path.join(path_to_save_results, 'plots')
path_to_fold_division_folder = f'/home/ubuntu/tenerife/data/ZZ_githubRepos/LungAmbition/Data_stratified_split/folds-def_{n_folds}folds'
keep_false_positives_as_separate_test = True
# create path_to_save_plots if it does not exist
if not os.path.exists(path_to_save_plots):
    os.makedirs(path_to_save_plots)

name_file= "ROS_RSF_proteins_6nov2024_correctedTo5YearsCensorship"
if keep_false_positives_as_separate_test:
    name_file= name_file + "_keep_false_positives_as_separate_test"
sys.stdout=open(os.path.join(path_to_save_results, "run_out_"+name_file+"_5fold.txt"),'w')
# save prints in a txt file
original_stdout = sys.stdout

df_merged = pd.read_csv('/home/ubuntu/tenerife/data/LungAmbition/Excels_merged/LungAmbitionMergedAll7nov2024.csv')
# ignore columns 'ID_imagingData', 'Group', 'Cancer_Status', 'TimeYears_CT_blood',
# 'Diff_Diag_Blood_TimeYears_CT_blood', 'LastFollow_upTimeYears_CT_blood', 'Age', 'Sex', 
# 'Body_mass_index','Smoking_status', 'Years_smoked', 'Smoking_pack_years', 
# 'Family_history_lung_cancer','Personal_history_cancer', 'Stage_category','NRRD_File', 'SEG_Files'
# in df_merged
# for correct handling of TimeYears_CT_blood, if TimeYears_CT_blood is over 5 years, change it to 5 years
# in df_merged column TimeYears_CT_blood change to 5 if its over 5 years for patients that are not in Lung_Cancer group
df_merged.loc[(df_merged['TimeYears_CT_blood'] == 5) & (df_merged['Group'] != 'Lung_Cancer'), 'TimeYears_CT_blood'] = 6
y_target = df_merged[['ID_proteinData', 'Cancer_Status', 'TimeYears_CT_blood']]
columns_to_drop=['ID_imagingData', 'Group', 'Cancer_Status', 
                'TimeYears_blood', 'TimeMonths_blood', 'TimeYears_CT_blood', 'TimeMonths_CT_blood',
                'Diff_Diag_Blood_TimeYears', 'LastFollow_upTimeYears', 
                'Age', 'Sex', 'Body_mass_index','Smoking_status', 'Years_smoked', 'Smoking_pack_years',
                'Family_history_lung_cancer','Personal_history_cancer', 'Stage_category',
                'NRRD_File', 'SEG_Files']
# for correct handling of TimeYears_CT_blood, if TimeYears_CT_blood is over 5 years, change it to 5 years
# in df_merged column TimeYears_CT_blood change to 5 if its over 5 years for patients that are not in Lung_Cancer group

X = df_merged.drop(columns=columns_to_drop)
if keep_false_positives_as_separate_test:
    # create list to store wrong predicted false positives
    list_ID_wrong_predicted_false_positives = []
    # save false_positive_metrics in df

# save best metrics for each fold
fold_metrics_df = pd.DataFrame(columns=['Fold', 'Concordance_Index', 'Integrated_Brier_Score',
                                        'AUC_at_0_Years', 'Accuracy_at_0_Years', 'Balanced_Accuracy_at_0_Years', 'Precision_at_0_Years', 'Recall_at_0_Years', 'F1_Score_at_0_Years', 'Specificity_at_0_Years', 'NPV_at_0_Years',
                                        'AUC_at_1_Years', 'Accuracy_at_1_Years', 'Balanced_Accuracy_at_1_Years', 'Precision_at_1_Years', 'Recall_at_1_Years', 'F1_Score_at_1_Years', 'Specificity_at_1_Years', 'NPV_at_1_Years',
                                        'AUC_at_2_Years', 'Accuracy_at_2_Years', 'Balanced_Accuracy_at_2_Years', 'Precision_at_2_Years', 'Recall_at_2_Years', 'F1_Score_at_2_Years', 'Specificity_at_2_Years', 'NPV_at_2_Years',
                                        'AUC_at_3_Years', 'Accuracy_at_3_Years', 'Balanced_Accuracy_at_3_Years', 'Precision_at_3_Years', 'Recall_at_3_Years', 'F1_Score_at_3_Years', 'Specificity_at_3_Years', 'NPV_at_3_Years',
                                        'AUC_at_4_Years', 'Accuracy_at_4_Years', 'Balanced_Accuracy_at_4_Years', 'Precision_at_4_Years', 'Recall_at_4_Years', 'F1_Score_at_4_Years', 'Specificity_at_4_Years', 'NPV_at_4_Years',
                                        'AUC_at_5_Years', 'Accuracy_at_5_Years', 'Balanced_Accuracy_at_5_Years', 'Precision_at_5_Years', 'Recall_at_5_Years', 'F1_Score_at_5_Years', 'Specificity_at_5_Years', 'NPV_at_5_Years'])
# save top features for each fold
top_features_df_list = []

# iterate over each file in path_to_fold_division_folder
# Use glob to find all CSV files in the folder
for file_path in sorted(glob.glob(os.path.join(path_to_fold_division_folder, 'id2splitfold_*.csv'))):
    print("Reading file:", file_path)
    # Extract the fold number from the filename
    fold = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
    # Load the CSV file into a DataFrame
    df_fold = pd.read_csv(file_path)
    # rename GroupUpdated column into Group in df_fold
    df_fold.rename(columns={'GroupUpdated': 'Group'}, inplace=True)
    # drop columns where Group == Control
    df_fold = df_fold[df_fold['Group'] != 'Control'] ### uncomment this for BC vs LC
    # rename in Group column IndeterminatePreLungCancer to Lung_Cancer in df_fold
    df_fold.loc[df_fold['Group'] == 'IndeterminatePreLungCancer', 'Group'] = 'Lung_Cancer'
    print(df_fold[df_fold['Group']=='Lung_Cancer'].TimeYears_CT_blood.value_counts())
    # next line is not necfcessary, but we keep here for completion
    df_fold.loc[(df_fold['TimeYears_CT_blood'] == 5) & (df_fold['Group'] != 'Lung_Cancer'), 'TimeYears_CT_blood'] = 6
    print("=" * 80)
    print(f"Fold {fold + 1}:")
    # based on df_fold.split (train, test, val, test_false_positive) asign in df_merged the corresponding fold
    # Get the ID_proteinData values for each split in df_fold
    train_ids = df_fold[df_fold['split'] == 'train']['ID_proteinData'].tolist()
    val_ids = df_fold[df_fold['split'] == 'val']['ID_proteinData'].tolist()
    test_ids = df_fold[df_fold['split'] == 'test']['ID_proteinData'].tolist()
    if keep_false_positives_as_separate_test:
        false_positive_ids = df_fold[df_fold['split'] == 'test_false_positive']['ID_proteinData'].tolist()
        # Get the corresponding rows from df_merged
        X_test_false_positives = df_merged[df_merged['ID_proteinData'].isin(false_positive_ids)]
        y_test_false_positives = df_merged[df_merged['ID_proteinData'].isin(false_positive_ids)][['Cancer_Status', 'TimeYears_CT_blood']]
        ID_false_positives = X_test_false_positives['ID_proteinData'].values
        X_test_false_positives = X_test_false_positives.drop(columns=columns_to_drop)
        X_test_false_positives = X_test_false_positives.drop(columns=['ID_proteinData'])
        print("False positives data:", X_test_false_positives.shape, y_test_false_positives.shape, 
              "Test false positives:", len(y_test_false_positives[y_test_false_positives.Cancer_Status == 0]))
    else:
        # include false positives in test
        test_ids = test_ids + df_fold[df_fold['split'] == 'test_false_positive']['ID_proteinData'].tolist()

    # Filter X and y_target based on these lists of IDs
    X_train = X[X['ID_proteinData'].isin(train_ids)].drop(columns=['ID_proteinData'])
    X_val = X[X['ID_proteinData'].isin(val_ids)].drop(columns=['ID_proteinData'])
    X_test = X[X['ID_proteinData'].isin(test_ids)].drop(columns=['ID_proteinData'])
    # ver que hacer con false_positives, eliminar ID_proteinData
    y_train = y_target[y_target['ID_proteinData'].isin(train_ids)].drop(columns=['ID_proteinData'])
    y_val = y_target[y_target['ID_proteinData'].isin(val_ids)].drop(columns=['ID_proteinData'])
    y_test = y_target[y_target['ID_proteinData'].isin(test_ids)].drop(columns=['ID_proteinData'])

    print("Training data:", X_train.shape, y_train.shape, 
          "Benigns train:", len(y_train[y_train.Cancer_Status == 0]), "Lung cancer patients train:", len(y_train[y_train.Cancer_Status == 1]))
    print("Validation data:", X_val.shape, y_val.shape, 
          "Benigns val:", len(y_val[y_val.Cancer_Status == 0]), "Lung cancer patients val:", len(y_val[y_val.Cancer_Status == 1]))
    print("Test data:", X_test.shape, y_test.shape, 
          "Benigns test:", len(y_test[y_test.Cancer_Status == 0]), "Lung cancer patients test:", len(y_test[y_test.Cancer_Status == 1]))
    print("Number of samples per class in y_train")
    print(y_train.Cancer_Status.value_counts())
    print("Number of samples per class in y_test")
    print(y_test.Cancer_Status.value_counts())
    print("Number of samples per class in y_val")
    print(y_val.Cancer_Status.value_counts())
    # join X_val in X_train and y_val in y_train ### OJO esto revisar!!
    X_train = pd.concat([X_train, X_val], axis=0)
    y_train = pd.concat((y_train, y_val), axis=0)
    # print max TimeYears_CT_blood in y_train
    print("Max TimeYears_CT_blood in y_train:", y_train['TimeYears_CT_blood'].max())
    print("Max TimeYears_CT_blood in y_test:", y_test['TimeYears_CT_blood'].max())
    # print training data after joining with validation data
    print("Training data after joining with validation data:", X_train.shape, y_train.shape, "Benign patients train:", len(y_train[y_train['Cancer_Status'] == False]), "Lung cancer patients train:", len(y_train[y_train['Cancer_Status'] == True]))
    print("Lung cancer patients (TimeYears_CT_blood = 0, before duplicating):", len(y_train[(y_train['Cancer_Status'] == 1) & (y_train['TimeYears_CT_blood'] == 0)]))
    print("Lung cancer patients (TimeYears_CT_blood > 0, before duplicating):", len(y_train[(y_train['Cancer_Status'] == 1) & (y_train['TimeYears_CT_blood'] > 0)]))
    print("Benign participants:", len(y_train[y_train['Cancer_Status'] == 0]))
    # combine X_train and y_train to apply oversampling and oversample only the lung cancer class
    X_y_train_combined = pd.concat([X_train, y_train], axis=1)
    # drop column Cancer_Status in X_y_train_combined
    y_train_CancerStatus = X_y_train_combined['Cancer_Status'].values
    # duplicate all LC patients to improve learning
    # in X_y_train_combined_TimeYears_CT_blood repeat twice rows where LungCancer == 1 and TimeYears_CT_blood == 0
    filtered_rows = X_y_train_combined[
        (X_y_train_combined['Cancer_Status'] == 1)]

    # Repeat these rows twice
    resampled_rows = pd.concat([filtered_rows]*2, ignore_index=True)

    # Append to the original DataFrame
    X_y_train_resampled_TimeYears_CT_blood = pd.concat(
        [X_y_train_combined, resampled_rows], ignore_index=True
    )
    X_train_resampled = X_y_train_resampled_TimeYears_CT_blood.drop(columns=['TimeYears_CT_blood', 'Cancer_Status'])
    y_train_resampled = X_y_train_resampled_TimeYears_CT_blood[['Cancer_Status', 'TimeYears_CT_blood']]
    print("Training data after duplicating LC patients:", X_train_resampled.shape, y_train_resampled.shape)
    # rename X_train_resampled to X_train and y_train_resampled to y_train
    X_train = X_train_resampled
    y_train = y_train_resampled
    print("Non-lung cancer patients (All years):", len(y_train[y_train == 0]))
    print("Lung cancer patients (TimeYears_CT_blood = 0):", len(y_train[(y_train['Cancer_Status'] == 1) & (y_train['TimeYears_CT_blood'] == 0)]))
    print("Lung cancer patients (TimeYears_CT_blood > 0):", len(y_train[(y_train['Cancer_Status'] == 1) & (y_train['TimeYears_CT_blood'] > 0)]))
    print("Benign participants:", len(y_train[y_train['Cancer_Status'] == 0]))

    # # print("y_train_CancerStatus after conversion:", y_train_CancerStatus)
    # X_y_train_combined_TimeYears_CT_blood = X_y_train_combined.drop(columns=['Cancer_Status'])

    # # Define RandomOverSampler to target the minority class in `Cancer_Status`
    # ros = RandomOverSampler(sampling_strategy='minority', random_state=seed)

    # # Apply oversampling on the combined DataFrame
    # X_y_train_resampled_TimeYears_CT_blood, y_train_resampled_CancerStatus = ros.fit_resample(X_y_train_combined_TimeYears_CT_blood, y_train_CancerStatus)
    # # print("y_train_resampledCancerStatus after conversion:", y_train_resampled_CancerStatus)
    # # Split the resampled data into X and y parts
    # y_train_resampled = pd.DataFrame(y_train_resampled_CancerStatus, columns=['Cancer_Status'])
    # # include in y_train_resampled the TimeYears_CT_blood
    # y_train_resampled['TimeYears_CT_blood'] = X_y_train_resampled_TimeYears_CT_blood['TimeYears_CT_blood']
    # X_train_resampled = X_y_train_resampled_TimeYears_CT_blood.drop(columns=['TimeYears_CT_blood'])  # All columns except 'TimeYears_CT_blood'

    # print("Training data after oversampling:", X_train_resampled.shape, y_train_resampled.shape)
    # print("Non lung cancer patients train:", len(y_train_resampled[y_train_resampled['Cancer_Status'] == False]))
    # print("Lung cancer patients train:", len(y_train_resampled[y_train_resampled['Cancer_Status'] == True]))

    # # rename X_train_resampled to X_train and y_train_resampled to y_train
    # X_train = X_train_resampled
    # y_train = y_train_resampled
    # oversample lung cancer at year 0 class
    # Identify patients where `TimeYears_CT_blood == 0`
        
    event = y_train["Cancer_Status"]  # Structured array access
    time = y_train["TimeYears_CT_blood"]  # Structured array access
    
    # convert y to structured array with the first field being a binary class event indicator and the second field the time of the event/censoring
    y_train = np.array([(status, time) for status, time in y_train.values], dtype=[('Cancer_Status', 'bool'), ('TimeYears_CT_blood', 'float')])
    y_test = np.array([(status, time) for status, time in y_test.values], dtype=[('Cancer_Status', 'bool'), ('TimeYears_CT_blood', 'float')])
    
    # scale the data using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # print shapes of X_train and X_test
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)
    # train RSF, first find the best hyperparameters through a grid search cv,
    # then train the model with the best hyperparameters
    # parameters to search
    param_grid = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [5, 6, 7],
        "min_samples_leaf": [4, 5],
        "max_features": ['sqrt', 150, 200, 250, 'log2'],
    }
    # random forest classifier
    rsf = RandomSurvivalForest(random_state=seed)
    # grid search cv
    # Set up GridSearchCV with validation
    grid_search = GridSearchCV(
        estimator=rsf,
        param_grid=param_grid,
        cv=KFold(n_splits=3, shuffle=True, random_state=seed),
        n_jobs=-1,
        verbose=0,
        scoring={'concordance': concordance_scorer, 'brier_0yr': brier_score_at_time},
        refit='brier_0yr'  # You can refit on the metric that prioritizes your objective
    )
    # sample_weights = compute_sample_weight(class_weight='balanced', y=(event.astype(bool)) & (time == 0))
    # Fit the grid search on training data
    grid_search.fit(X_train, y_train)#, sample_weight=sample_weights)

    # Print the best hyperparameters
    print("Best hyperparameters:")
    print(grid_search.best_params_)
    best_score = grid_search.best_score_
    print(f"Best Brier Score at 0 year: {-best_score}")
    # print("Concordance Index scores for each parameter set:\n", grid_search.cv_results_['mean_test_concordance'])
    # Use the best estimator to train the model
    best_rsf = grid_search.best_estimator_
    best_rsf.fit(X_train, y_train)#, sample_weight=sample_weights)
    
    # Predict risk scores for train and validation sets
    train_risk_scores = best_rsf.predict(X_train)
    # Concordance Index
    train_concordance = concordance_index_censored(y_train["Cancer_Status"], y_train["TimeYears_CT_blood"], train_risk_scores)[0]
    # Print metrics for train and validation sets
    print("Metrics in train:")
    print(f"Concordance Index: {train_concordance}")
    print("=" * 80)

    # predict in test
    y_pred = best_rsf.predict(X_test)

    # Calculate the concordance index
    concordance = concordance_index_censored(y_test["Cancer_Status"], y_test["TimeYears_CT_blood"], y_pred)[0]

    # Predict the survival functions for each sample in the training set
    surv_funcs = best_rsf.predict_survival_function(X_test)
    # output is a is a list of survival functions, one for each sample in X_test. 
    # Each survival function provides the probability of survival.
    # These probabilities range from 1 (100% survival) at the start of the study period 
    # down to lower values as time progresses, representing the likelihood that the event 
    # has not yet occurred by a given time.

    # Compute censoring survival function
    time_censoring, survival_censoring = kaplan_meier_estimator(event= y_train["Cancer_Status"], time_exit=y_train["TimeYears_CT_blood"])

    # Plot censoring survival function
    plt.step(time_censoring, survival_censoring, where="post", label="Censoring Survival Function")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Kaplan-Meier Censoring Survival Curve")
    plt.savefig(os.path.join(path_to_save_plots, f"KM_Censoring_Surv_fold_{fold+1}.png"))
    plt.close()

    # print("surv_funcs:", surv_funcs)
    # time_points = [0, 1, 2, 3, 4, 5]  # Specify the time points at which to calculate the Brier score
    time_points = [float(t) for t in [0, 1, 2, 3, 4, 5]]
    print(f"Min time: {y_test['TimeYears_CT_blood'].min()}, Max time: {y_test['TimeYears_CT_blood'].max()}")
    print("y_pred", y_pred)
    # Compute time-dependent AUC
    auc_scores = cumulative_dynamic_auc(y_train, y_test, y_pred, time_points)
    # Calculate survival probabilities at each specified time point for each sample
    test_estimates = [np.array([fn(t) for t in time_points]) for fn in surv_funcs]
    # print("test_estimates:", test_estimates)
    # # Compute Brier scores at the specified time points
    # # Ensure y_train is in the required structured format (e.g., dtype=[('Cancer_Status', bool), ('TimeYears_CT_blood', float)])
    # # print max time in y_train, y_test
    # print("Max TimeYears_CT_blood in y_train:", y_train['TimeYears_CT_blood'].max())
    # print("Min TimeYears_CT_blood in y_train:", y_train['TimeYears_CT_blood'].min())
    # print("y_train dtype:", y_train.dtype)
    # print("y_test dtype:", y_test.dtype)
    # print("Max TimeYears_CT_blood in y_test:", y_test['TimeYears_CT_blood'].max())
    # print("Min TimeYears_CT_blood in y_test:", y_test['TimeYears_CT_blood'].min())
    # print("time_points:", time_points)
    test_brier_scores = brier_score(survival_train=y_train, survival_test=y_test, estimate=test_estimates, times=time_points)
    
    # Integrated Brier Score (IBS) is the mean of Brier scores across all time points
    test_ibs = np.mean(test_brier_scores[1])  # train_brier_scores[1] contains the Brier scores at each time point
    print(f"Metrics in test for fold: {fold + 1}")
    print(f"Concordance Index: {concordance}")
    print(f"Integrated Brier Score (IBS): {test_ibs}")
    print("=" * 80)

    threshold = 0.6

    # Calculate survival probabilities at each specified time point for each sample
    # Survival probability predictions at 1, 3, and 5 years
    surv_probs = {t: np.array([fn(t) for fn in surv_funcs]) for t in time_points}

    # Initialize lists to store metric values for each time point
    performance_metrics = []

    # Iterate over each time point (1, 3, 5 years)
    for t in time_points:
        # Binarize the predictions based on the threshold (0.5)
        risk_scores=surv_probs[t]
        y_pred_binary = surv_probs[t] >= threshold
        y_true_binary = y_test["TimeYears_CT_blood"] > t  # True if the patient survived beyond the time point
        print("y_true_binary for time",t,"is", y_true_binary)
        print("y_pred_binary for time",t,"is", y_pred_binary)
        print("surv_probs for time",t,"is", surv_probs[t])
        print(f"AUC at time {t} (calculated using sksurv): {auc_scores[0][int(t)]:.3f}")

        # Calculate metrics
        auc_roc = roc_auc_score(y_true_binary, risk_scores)
        acc = accuracy_score(y_true_binary, y_pred_binary)
        balanced_acc = balanced_accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, average='weighted')  # Handles cases with no positive predictions
        recall = recall_score(y_true_binary, y_pred_binary, average='weighted')
        f1_sc = f1_score(y_true_binary, y_pred_binary, average='weighted')
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
        test_NPV=tn/(tn+fn)
        test_specificity=tn/(tn+fp)

        # Append metrics for this time point to the results list
        performance_metrics.append({
            "Time (years)": t,
            "AUC": auc_roc,
            "Accuracy": acc,
            "Balanced Accuracy": balanced_acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_sc,
            "Specificity": test_specificity,
            "NPV": test_NPV
        })

        # Calculate and print the number of patients who developed cancer by the time point
        num_actual_cancer_patients = sum(y_test[y_test['Cancer_Status']==1]["TimeYears_CT_blood"] <= t)
        num_predicted_cancer_patients = sum(~y_pred_binary)  # Predicted as having cancer within the time point

        # Print metrics
        print(f"Performance at {t} years:")
        print(f"AUC: {auc_roc}")
        print(f"Accuracy: {acc}")
        print(f"Balanced Accuracy: {balanced_acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1_sc}")
        print(f"Specificity: {test_specificity}")
        print(f"NPV: {test_NPV}")
        print(f"Number of patients who developed cancer by {t} years: {num_actual_cancer_patients}/{len(y_test)}")
        print(f"Number of predicted patients with cancer by {t} years: {num_predicted_cancer_patients}/{len(y_test)}")
        print("=" * 80)
        print("=" * 80)


    # Save metrics to the DataFrame
    fold_metrics_df = pd.concat([fold_metrics_df, pd.DataFrame([{
        'Fold': fold,
        'Concordance_Index': concordance,
        'Integrated_Brier_Score': test_ibs,
        'AUC_at_0_Years': performance_metrics[0]["AUC"],
        'Accuracy_at_0_Years': performance_metrics[0]["Accuracy"],
        'Balanced_Accuracy_at_0_Years': performance_metrics[0]["Balanced Accuracy"],
        'Precision_at_0_Years': performance_metrics[0]["Precision"],
        'Recall_at_0_Years': performance_metrics[0]["Recall"],
        'F1_Score_at_0_Years': performance_metrics[0]["F1 Score"],
        'Specificity_at_0_Years': performance_metrics[0]["Specificity"],
        'NPV_at_0_Years': performance_metrics[0]["NPV"],
        'AUC_at_1_Years': performance_metrics[1]["AUC"],
        'Accuracy_at_1_Years': performance_metrics[1]["Accuracy"],
        'Balanced_Accuracy_at_1_Years': performance_metrics[1]["Balanced Accuracy"],
        'Precision_at_1_Years': performance_metrics[1]["Precision"],
        'Recall_at_1_Years': performance_metrics[1]["Recall"],
        'F1_Score_at_1_Years': performance_metrics[1]["F1 Score"],
        'Specificity_at_1_Years': performance_metrics[1]["Specificity"],
        'NPV_at_1_Years': performance_metrics[1]["NPV"],
        'AUC_at_2_Years': performance_metrics[2]["AUC"],
        'Accuracy_at_2_Years': performance_metrics[2]["Accuracy"],
        'Balanced_Accuracy_at_2_Years': performance_metrics[2]["Balanced Accuracy"],
        'Precision_at_2_Years': performance_metrics[2]["Precision"],
        'Recall_at_2_Years': performance_metrics[2]["Recall"],
        'F1_Score_at_2_Years': performance_metrics[2]["F1 Score"],
        'Specificity_at_2_Years': performance_metrics[2]["Specificity"],
        'NPV_at_2_Years': performance_metrics[2]["NPV"],
        'AUC_at_3_Years': performance_metrics[3]["AUC"],
        'Accuracy_at_3_Years': performance_metrics[3]["Accuracy"],
        'Balanced_Accuracy_at_3_Years': performance_metrics[3]["Balanced Accuracy"],
        'Precision_at_3_Years': performance_metrics[3]["Precision"],
        'Recall_at_3_Years': performance_metrics[3]["Recall"],
        'F1_Score_at_3_Years': performance_metrics[3]["F1 Score"],
        'Specificity_at_3_Years': performance_metrics[3]["Specificity"],
        'NPV_at_3_Years': performance_metrics[3]["NPV"],
        'AUC_at_4_Years': performance_metrics[4]["AUC"],
        'Accuracy_at_4_Years': performance_metrics[4]["Accuracy"],
        'Balanced_Accuracy_at_4_Years': performance_metrics[4]["Balanced Accuracy"],
        'Precision_at_4_Years': performance_metrics[4]["Precision"],
        'Recall_at_4_Years': performance_metrics[4]["Recall"],
        'F1_Score_at_4_Years': performance_metrics[4]["F1 Score"],
        'Specificity_at_4_Years': performance_metrics[4]["Specificity"],
        'NPV_at_4_Years': performance_metrics[4]["NPV"],
        'AUC_at_5_Years': performance_metrics[5]["AUC"],
        'Accuracy_at_5_Years': performance_metrics[5]["Accuracy"],
        'Balanced_Accuracy_at_5_Years': performance_metrics[5]["Balanced Accuracy"],
        'Precision_at_5_Years': performance_metrics[5]["Precision"],
        'Recall_at_5_Years': performance_metrics[5]["Recall"],
        'F1_Score_at_5_Years': performance_metrics[5]["F1 Score"],
        'Specificity_at_5_Years': performance_metrics[5]["Specificity"],
        'NPV_at_5_Years': performance_metrics[5]["NPV"]
    }])], ignore_index=True)
    # evaluate separately model on false positives:
    if keep_false_positives_as_separate_test:
        print("Evaluate model on false positives")
        # convert y to structured array with the first field being a binary class event indicator and the second field the time of the event/censoring
        y_test_false_positives = np.array([(status, time) for status, time in y_test_false_positives.values], dtype=[('Cancer_Status', 'bool'), ('TimeYears_CT_blood', 'float')])
        # scale the data using StandardScaler
        X_test_false_positives = scaler.transform(X_test_false_positives)
        y_false_positives_pred = best_rsf.predict(X_test_false_positives)
        surv_funcs_fp = best_rsf.predict_survival_function(X_test_false_positives)
        # Define the time point and probability threshold for survival (e.g., survival probability at 5 years)
        time_threshold = 5 # in years

        # Calculate survival probabilities at 5 years for each sample
        surv_probs_at_5_years = np.array([fn(time_threshold) for fn in surv_funcs_fp])

        # Determine which patients are correctly predicted as survivors
        # A correct prediction means the predicted survival probability is above the threshold (e.g., > 0.5)
        predicted_survivors = surv_probs_at_5_years >= threshold

        # Evaluate how many false positives (all survivors) are correctly predicted
        # Since all patients in `y_test_false_positives` are survivors, we expect all `predicted_survivors` to be True.
        correct_predictions = predicted_survivors.sum()
        incorrect_predictions = len(predicted_survivors) - correct_predictions

        # Output the result
        print(f"Number of correctly predicted survivors: {correct_predictions}")
        print(f"Number of incorrectly predicted survivors: {incorrect_predictions}")

        # Save IDs of incorrect predictions
        ID_false_positives_wrong_predicted = ID_false_positives[~predicted_survivors]  # IDs of incorrectly predicted survivors
        print("Wrongly predicted False Positives:", ID_false_positives_wrong_predicted)
        list_ID_wrong_predicted_false_positives.append(ID_false_positives_wrong_predicted)
    
    # calculate most relevant features for the model
    feature_importances = permutation_importance(best_rsf, X_test, y_test, n_repeats=100, random_state=seed)
    # get only the 30 most relevant features and plot them
    indices = np.argsort(feature_importances.importances_mean)[::-1]
    # Select top 100 features and save them for this fold
    fold_feature_importances = pd.DataFrame({
        'Feature': X.columns[indices][:100],
        'Importance_mean': feature_importances.importances_mean[indices][:100],
        'Importance_SD': feature_importances.importances_std[indices][:100],
        'Rank': range(1, 101)
    })
    fold_feature_importances['Fold'] = fold
    # Append this fold's DataFrame to the list for further analysis
    top_features_df_list.append(fold_feature_importances)
    # plot most important features
    plt.figure()
    plt.title("Permutation feature importances fold "+str(fold+1))
    plt.bar(range(30), feature_importances.importances_mean[indices][:30])
    plt.xticks(range(30), X.columns[indices][:30], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save_plots, f"Permutation_feature_importances_fold_{fold+1}.png"))
    plt.close()
    print("*" * 80)

# print mean and std of each metric
print("=" * 80)
mean_concordance = fold_metrics_df['Concordance_Index'].mean()
std_concordance = fold_metrics_df['Concordance_Index'].std()

mean_ibs = fold_metrics_df['Integrated_Brier_Score'].mean()
std_ibs = fold_metrics_df['Integrated_Brier_Score'].std()

# Print the results
print("Mean and SD of Metrics Across Folds:")
print(f"Concordance Index - Mean: {mean_concordance:.3f}, SD: {std_concordance:.3f}")
print(f"Integrated Brier Score (IBS) - Mean: {mean_ibs:.3f}, SD: {std_ibs:.3f}")

# Print metrics at time points 1, 3, 5 years
print("Mean and SD of Metrics at 1, 3, 5 years:")
for t in [0, 1, 2, 3, 4, 5]:
    mean_auc = fold_metrics_df[f"AUC_at_{t}_Years"].mean()
    std_auc = fold_metrics_df[f"AUC_at_{t}_Years"].std()

    mean_acc = fold_metrics_df[f"Accuracy_at_{t}_Years"].mean()
    std_acc = fold_metrics_df[f"Accuracy_at_{t}_Years"].std()

    mean_balanced_acc = fold_metrics_df[f"Balanced_Accuracy_at_{t}_Years"].mean()
    std_balanced_acc = fold_metrics_df[f"Balanced_Accuracy_at_{t}_Years"].std()

    mean_precision = fold_metrics_df[f"Precision_at_{t}_Years"].mean()
    std_precision = fold_metrics_df[f"Precision_at_{t}_Years"].std()

    mean_recall = fold_metrics_df[f"Recall_at_{t}_Years"].mean()
    std_recall = fold_metrics_df[f"Recall_at_{t}_Years"].std()

    mean_f1_score = fold_metrics_df[f"F1_Score_at_{t}_Years"].mean()
    std_f1_score = fold_metrics_df[f"F1_Score_at_{t}_Years"].std()

    mean_specificity = fold_metrics_df[f"Specificity_at_{t}_Years"].mean()
    std_specificity = fold_metrics_df[f"Specificity_at_{t}_Years"].std()

    mean_NPV = fold_metrics_df[f"NPV_at_{t}_Years"].mean()
    std_NPV = fold_metrics_df[f"NPV_at_{t}_Years"].std()

    print(f"Metrics at {t} years:")
    print(f"AUC - Mean: {mean_auc:.3f}, SD: {std_auc:.3f}")
    print(f"Accuracy - Mean: {mean_acc:.3f}, SD: {std_acc:.3f}")
    print(f"Balanced Accuracy - Mean: {mean_balanced_acc:.3f}, SD: {std_balanced_acc:.3f}")
    print(f"Precision - Mean: {mean_precision:.3f}, SD: {std_precision:.3f}")
    print(f"Recall - Mean: {mean_recall:.3f}, SD: {std_recall:.3f}")
    print(f"F1 Score - Mean: {mean_f1_score:.3f}, SD: {std_f1_score:.3f}")
    print(f"Specificity - Mean: {mean_specificity:.3f}, SD: {std_specificity:.3f}")
    print(f"NPV - Mean: {mean_NPV:.3f}, SD: {std_NPV:.3f}")
    print("-" * 40)


# calculate mean and std metrics for false positives
if keep_false_positives_as_separate_test:
    print("=" * 80)
    # count number of ocurrences of each false positive in list_ID_wrong_predicted_false_positives
    from collections import Counter
    counter = Counter([item for sublist in list_ID_wrong_predicted_false_positives for item in sublist])
    print("Number of times each False Positive was wrongly predicted:")
    print(counter)

# Concatenate all fold DataFrames for easier comparison
all_folds_df = pd.concat(top_features_df_list)
# save to_features_df_list to csv
all_folds_df.to_csv(os.path.join(path_to_save_results, "top_features_folds.csv"), index=False)

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
# print("Features common in exactly 3 folds:\n", common_in_3_folds)
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    # Print out the summary DataFrames
    if n_folds==3:
        print("Features common in exactly 3 folds:\n", common_in_3_folds)
    else:
        print("\nFeatures common in exactly 4 folds:\n", common_in_4_folds)
        print("\nFeatures common in all 5 folds:\n", common_in_5_folds)

sys.stdout.close()