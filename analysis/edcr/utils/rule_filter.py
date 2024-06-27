import os
import pandas as pd
from . import reader
from sklearn.metrics import classification_report

def should_exclude(filename, exclude_models):
    """ 
    Ablation Helper function
    """
    try:
        model_type = filename.split('Rule')[1].strip().split('_')[0]
        # print(f'File: {filename}, Model: {model_type}, Exclude: {exclude_models}')
    except IndexError:
        return False
    
    return model_type in exclude_models

def npy_to_top_n_f1_bowpy(base_model_file_path, rule_result_dir, top_n, exclude_models=None):
    """
    Top N selection algorithm, and includes functionality to exclude models for ablation
    """
    ablation_filter = 0
    if exclude_models is None:
        exclude_models = []

    # Read the base model predictions
    bowpy_dataframe = pd.read_csv(base_model_file_path)
    bowpy_dataframe.rename(columns={"Predicted": "pred", "True": "corr"}, inplace=True)
    bowpy_dataframe['true_positive'] = bowpy_dataframe.apply(lambda x: 1 if x['pred'] == 1 and x['corr'] == 1 else 0, axis=1)
    bowpy_dataframe['false_positive'] = bowpy_dataframe.apply(lambda x: 1 if x['pred'] == 1 and x['corr'] == 0 else 0, axis=1)

    base_model_name = os.path.basename(base_model_file_path).replace("_predictions.csv", "")
    mapped_base_model_name = reader.map_base_model_to_rule_name(base_model_name)

    # Collect f1 scores and rule model details
    rank = []
    for model_file in os.listdir(rule_result_dir):
        if model_file.endswith(".csv") and not "Rule all" in model_file and mapped_base_model_name in model_file:
            if should_exclude(model_file, exclude_models):
                # print(f'Excluded: {model_file} due to {exclude_models} exclusion')
                ablation_filter += 1
                continue

            model_predictions = pd.read_csv(os.path.join(rule_result_dir, model_file))
            if len(model_predictions['Predicted']) == len(bowpy_dataframe['pred']):
                f1_score = classification_report(model_predictions['True'], model_predictions['Predicted'], output_dict=True)['1']['f1-score']
                rank.append((model_file, model_predictions['Predicted'], f1_score))

    # Sort rules by F1 score and apply top N filtering
    sorted_rank = sorted(rank, key=lambda x: x[2], reverse=True)
    top_n = min(top_n, len(sorted_rank))  # Ensure we do not exceed the number of available rules

    # Add top N rules to the DataFrame
    for i in range(top_n):
        model_file, model_predictions, _ = sorted_rank[i]
        bowpy_dataframe[f"rule{i}"] = model_predictions

    print (f'Excluded {ablation_filter} rules due to model filtering')

    return bowpy_dataframe

def npy_to_threshold_f1_bowpy(base_model_file_path, rule_result_dir, threshold, exclude_models=None):
    """
    Select models based on F1 threshold, and includes functionality to exclude models for ablation
    """
    if exclude_models is None:
        exclude_models = []    
        
    # Read the base model predictions
    bowpy_dataframe = pd.read_csv(base_model_file_path)
    bowpy_dataframe.rename(columns={"Predicted": "pred", "True": "corr"}, inplace=True)
    bowpy_dataframe['true_positive'] = bowpy_dataframe.apply(lambda x: 1 if x['pred'] == 1 and x['corr'] == 1 else 0, axis=1)
    bowpy_dataframe['false_positive'] = bowpy_dataframe.apply(lambda x: 1 if x['pred'] == 1 and x['corr'] == 0 else 0, axis=1)

    base_model_name = os.path.basename(base_model_file_path).replace("_predictions.csv", "")
    mapped_base_model_name = reader.map_base_model_to_rule_name(base_model_name)
    # print(mapped_base_model_name)

    # Filtering rules and applying F1 threshold
    rule_index = 0
    ablation_filter = 0
    f1_filter = 0
    scuffed_filter = 0
    for model_file in os.listdir(rule_result_dir):
        if model_file.endswith(".csv") and "Rule all" not in model_file and mapped_base_model_name in model_file:
            if should_exclude(model_file, exclude_models):
                print(f'Excluded: {model_file}')
                ablation_filter += 1
                continue

            model_predictions = pd.read_csv(os.path.join(rule_result_dir, model_file))

            if len(model_predictions['Predicted']) == len(bowpy_dataframe['pred']):
                f1_score = classification_report(model_predictions['True'], model_predictions['Predicted'], output_dict=True)['1']['f1-score']

                if f1_score >= threshold:
                    bowpy_dataframe[f"rule{rule_index}"] = model_predictions['Predicted']
                    rule_index += 1
                else: 
                    f1_filter += 1
            else:
                scuffed_filter += 1

    print (f'Excluded {f1_filter} rules due to F1 filtering')
    print (f'Excluded {ablation_filter} rules due to model filtering')
    print (f'Excluded {scuffed_filter} rules due to problematic formatting')
    return bowpy_dataframe

