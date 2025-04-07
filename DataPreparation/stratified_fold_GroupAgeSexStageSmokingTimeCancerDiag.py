import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
from matplotlib.font_manager import FontProperties

def create_stratified_folds(df_cur, bins, labels, n_splits=5, random_state=1, val_split=0.10):
    # Update the Group variable to Group_stage variable to include subgroups for Lung_Cancer patients
    df_cur['Group_stage'] = df_cur['GroupUpdated']
    df_cur.loc[df_cur['Group_stage'] == 'Lung_Cancer', 'Group_stage'] = df_cur.apply(
        lambda x: f'Lung_Cancer_{x["Stage_category"]}' if pd.notna(x['Stage_category']) else 'Lung_Cancer', axis=1
    )

    # Create Age groups and combine with other variables
    df_cur['Age_Group'] = pd.cut(df_cur['Age'], bins=bins, labels=labels, right=False)
    df_cur['Grouping_Variable'] = df_cur['Sex'].astype(str) + '_' + df_cur['Age_Group'].astype(str)

    df_cur['Fine_Grained_Group'] = (df_cur['Group_stage'] + "_" + df_cur['Sex'].astype(str) + 
                                    "_" + df_cur['Age_Group'].astype(str) +
                                    "_" + df_cur['Smoking_Category'])

    sgkf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_data = {}

    for fold_number, (train_index, test_index) in enumerate(sgkf.split(df_cur, df_cur['Fine_Grained_Group'])):
        train_data = df_cur.iloc[train_index]

        # Further split the training data into training and validation sets, stratified by GroupUpdated variable
        train_idx, val_idx = train_test_split(
            train_index, test_size=val_split, random_state=random_state, stratify=train_data['GroupUpdated']
        )

        fold_data[fold_number] = {
            'train_index': train_idx,
            'val_index': val_idx,
            'test_index': test_index
        }

    return fold_data

def plot_distributions(df_cur, fold_data, bins, save_plots=False, output_folder=None):
    for fold_number, indices in fold_data.items():
        train_data = df_cur.iloc[indices['train_index']]
        val_data = df_cur.iloc[indices['val_index']]
        test_data = df_cur.iloc[indices['test_index']]

        label_fontsize = 30
        tick_fontsize = 26
        legend_fontsize = 26

        bold_font = FontProperties()
        bold_font.set_size(legend_fontsize)
        bold_font.set_weight('bold')

        # Rename Group categories
        rename_categories = {
            "Benign_Nodules": "Benign",
            "IndeterminatePreLungCancer": "Pre-LC",
            "Lung_Cancer_I-II": "LC I-II",
            "Lung_Cancer_III-IV": "LC III-IV"
        }
        
        for df in [train_data, val_data, test_data]:
            if "Group_stage" in df.columns:
                df["Group_stage"] = df["Group_stage"].replace(rename_categories)

        # Define a 2x2 subplot layout (4 plots total)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=False)
        fig.subplots_adjust(hspace=0.5, wspace=0.25)
        # fig.suptitle(f'Distribution for Fold {fold_number + 1}', fontsize=22, fontweight='bold')

        colors = ['#6baed6',  # Light blue
                '#74c476',  # Light green
                '#fb6a4a']  # Light red/salmon  # Train, Validation, Test colors

        # --- Age Distribution ---
        age_bins = pd.cut(df_cur['Age'], bins=bins)
        age_dist = pd.DataFrame({
            'Train': train_data['Age'].groupby(age_bins).size(),
            'Validation': val_data['Age'].groupby(age_bins).size(),
            'Test': test_data['Age'].groupby(age_bins).size()
        }).fillna(0)
        age_dist.plot(kind='bar', stacked=True, ax=axes[0, 0], color=colors, edgecolor='black', legend=False)

        # axes[0, 0].set_title('Age Distribution', fontsize=18, fontweight='bold')
        axes[0, 0].set_xlabel('Age', fontsize=label_fontsize, fontweight='bold')
        axes[0, 0].set_ylabel('Count', fontsize=label_fontsize, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
        axes[0, 0].tick_params(axis='y', labelsize=tick_fontsize)
        # axes[0, 0].legend(fontsize=label_fontsize, frameon=False)  # Remove legend box

        # --- Sex Distribution ---
        sex_dist = pd.DataFrame({
            'Train': train_data['Sex'].value_counts(),
            'Validation': val_data['Sex'].value_counts(),
            'Test': test_data['Sex'].value_counts()
        }).fillna(0)
        sex_dist.plot(kind='bar', stacked=True, ax=axes[0, 1], color=colors, edgecolor='black', legend=False)

        # axes[0, 1].set_title('Sex', fontsize=18, fontweight='bold')
        axes[0, 1].set_xlabel('Sex', fontsize=label_fontsize, fontweight='bold')
        axes[0, 1].set_ylabel('Count', fontsize=label_fontsize, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=0, labelsize=tick_fontsize)
        axes[0, 1].tick_params(axis='y', labelsize=tick_fontsize)
        # axes[0, 1].legend(fontsize=label_fontsize, frameon=False)  # Remove legend box

        # --- Group Distribution (Renamed Categories Applied) ---
        group_dist = pd.DataFrame({
            'Train': train_data['Group_stage'].value_counts(),
            'Validation': val_data['Group_stage'].value_counts(),
            'Test': test_data['Group_stage'].value_counts()
        }).fillna(0)
        custom_order = ["Control", "Benign", "Pre-LC", "LC I-II", "LC III-IV"]
        group_dist = group_dist.reindex(custom_order)
        group_dist.plot(kind='bar', stacked=True, ax=axes[1, 0], color=colors, edgecolor='black', legend=False)

        # axes[1, 0].set_title('Group Distribution', fontsize=18, fontweight='bold')
        axes[1, 0].set_xlabel('Group', fontsize=label_fontsize, fontweight='bold')
        axes[1, 0].set_ylabel('Count', fontsize=label_fontsize, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
        axes[1, 0].tick_params(axis='y', labelsize=tick_fontsize)
        # axes[1, 0].legend(fontsize=label_fontsize, frameon=False)  # Remove legend box

        # --- Smoking Distribution ---
        smoking_dist = pd.DataFrame({
            'Train': train_data['Smoking_Category'].value_counts(),
            'Validation': val_data['Smoking_Category'].value_counts(),
            'Test': test_data['Smoking_Category'].value_counts()
        }).fillna(0)
        smoking_dist.plot(kind='bar', stacked=True, ax=axes[1, 1], color=colors, edgecolor='black', legend=False)

        # axes[1, 1].set_title('Smoking Distribution', fontsize=18, fontweight='bold')
        axes[1, 1].set_xlabel('Smoking', fontsize=label_fontsize, fontweight='bold')
        axes[1, 1].set_ylabel('Count', fontsize=label_fontsize, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
        axes[1, 1].tick_params(axis='y', labelsize=tick_fontsize)
        # axes[1, 1].legend(fontsize=label_fontsize, frameon=False)  # Remove legend box
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, prop=bold_font, fontsize=legend_fontsize, frameon=False)
        # Save the plot if requested
        if save_plots and output_folder:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plot_path = os.path.join(output_folder, f'fold_{fold_number + 1}_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            # save in pdf too
            plot_path = os.path.join(output_folder, f'fold_{fold_number + 1}_distribution.pdf')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {plot_path}")

        plt.show()
        plt.close()

