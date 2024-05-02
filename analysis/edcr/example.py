#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd
from src.rule_correction import *
import os

# load data
def run_edcr():
    with open(f'data/test.npy', 'rb') as f:
        data = np.load(f, allow_pickle=True)

    # results = []
    # epsilon = [0.001 * i for i in range(1, 100, 1)]
    epsilon = [0.001 * i for i in range(5, 300, 1)]
    # for ep in epsilon:
    #     #result = PosNegRuleLearn(all_charts, epsilon)
    #     result = ruleForNegativeCorrection(data, ep)
    #     results.append([ep] + result)
    #     print(f"ep:{ep}\n{result}")
    # col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
    # df = pd.DataFrame(results, columns = ['epsilon'] + col )
    # df.to_csv(f"rule_for_Negativecorrection.csv")

    # results = []
    # for ep in epsilon:
    #     #result = PosNegRuleLearn(all_charts, epsilon)
    #     result = ruleForPNCorrection(data, ep)
    #     results.append([ep] + result)
    #     print(f"ep:{ep}\n{result}")
    # df = pd.DataFrame(results, columns = ['epsilon'] + col )
    # df.to_csv( f"rule_for_PNcorrection.csv")

    results = []
    for ep in epsilon:
        #result = PosNegRuleLearn(all_charts, epsilon)
        result = ruleForNPCorrection(data, ep)
        results.append([ep] + result)
        print(f"ep:{ep}\n{result}")
    col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
    #df = pd.DataFrame(results, columns = ['epsilon'] + col + ['acc', 'macro-F1', 'micro-F1'])
    df = pd.DataFrame(results, columns = ['epsilon'] + col )
    df.to_csv( f"rule_for_NPcorrection.csv")
    if os.path.exists('Results.xlsx'):
        os.remove('Results.xlsx')
    df.to_excel('Results.xlsx', sheet_name='EDCR Results', index=False)

    return df

def format_model_name(row):
    name = row['Name']
    params = row['Params']
    
    if 'RNN' in name:
        units = params.split(' ')[0]
        return f"{name}_{units}_units_predictions"
    if 'LSTM' in name:
        layers = params.split(' ')[0]
        return f"{name}_{layers}_layers_predictions"
    elif 'CNN' in name:
        filters, kernel = params.split(', ')
        filter_num = filters.split(' ')[0]
        kernel_size = kernel.split(' ')[2]
        if 'Attention' in name:
            return f"CNN_Attention_{filter_num}_filters_{kernel_size}_kernels_predictions"
        else:
            return f"CNN_{filter_num}_filters_{kernel_size}_kernels_predictions"
    else:
        return f"Unknown_model_type_{name}"


def model_select(csv_filepath):
    df = pd.read_csv(csv_filepath, nrows=32)

    # Remove rows where any of the precision or recall for 0 or 1 is 0 or 1
    filtered_df = df[(df[['Precision (0)', 'Recall (0)', 'Precision (1)', 'Recall (1)']] != 0).all(axis=1) &
                     (df[['Precision (0)', 'Recall (0)', 'Precision (1)', 'Recall (1)']] != 1).all(axis=1)]

    if filtered_df.empty:
        print("No models meet the criteria.")
        return []
    
    def filter_by_precision_recall_diff(sub_df, threshold=0.6):
        return sub_df[(sub_df['Precision (1)'] - sub_df['Recall (1)']).abs() <= threshold]

    # Identify models based on the criteria
    highest_f1 = filtered_df.loc[filtered_df['F1 (1)'].idxmax()]
    highest_recall = filtered_df.loc[filtered_df['Recall (1)'].idxmax()]
    highest_precision = filtered_df.loc[filtered_df['Precision (1)'].idxmax()]
    lowest_f1 = filtered_df.loc[filtered_df['F1 (1)'].idxmin()]

    models = {
        'Highest F1': {
            'model': format_model_name(highest_f1),
            'metrics': highest_f1[['Precision (1)', 'Recall (1)', 'F1 (1)']].to_dict()
        },
        'Highest Recall': {
            'model': format_model_name(highest_recall),
            'metrics': highest_recall[['Precision (1)', 'Recall (1)', 'F1 (1)']].to_dict()
        },
        'Highest Precision': {
            'model': format_model_name(highest_precision),
            'metrics': highest_precision[['Precision (1)', 'Recall (1)', 'F1 (1)']].to_dict()
        },
        'Lowest F1': {
            'model': format_model_name(lowest_f1),
            'metrics': lowest_f1[['Precision (1)', 'Recall (1)', 'F1 (1)']].to_dict()
        }
    }

    # Print the selected metric values for verification
    for key, value in models.items():
        print(f"{key}: {value['metrics']} at model - {value['model']}")

    return models

# def model_select(csv_filepath):
#     df = pd.read_csv(csv_filepath, nrows=32)

#     # Remove rows where any of the precision or recall for 0 or 1 is 0 or 1
#     filtered_df = df[(df[['Precision (0)', 'Recall (0)', 'Precision (1)', 'Recall (1)']] != 0).all(axis=1) &
#                      (df[['Precision (0)', 'Recall (0)', 'Precision (1)', 'Recall (1)']] != 1).all(axis=1)]

#     if filtered_df.empty:
#         print("No models meet the criteria.")
#         return []

#     # Filter by precision-recall difference
#     filtered_df = filtered_df[(filtered_df['Precision (1)'] - filtered_df['Recall (1)']).abs() <= 0.5]
    
#     # Initialize a dictionary to store model data and a set to track used models
#     models = {}
#     used_models = set()

#     # Define metric priorities and criteria
#     metric_criteria = [
#         ('F1 (1)', False, 'Highest F1'),
#         ('Recall (1)', False, 'Highest Recall'),
#         ('Precision (1)', False, 'Highest Precision'),
#         ('F1 (1)', True, 'Lowest F1')
#     ]

#     for metric, ascending, label in metric_criteria:
#         # Sort and filter for the best or worst model based on the metric criteria
#         best_model = filtered_df.sort_values(by=metric, ascending=ascending).drop(index=used_models).iloc[0] if not filtered_df.empty else None
        
#         if best_model is not None:
#             model_id = best_model.name  # Assuming the index is the model ID
#             models[label] = {
#                 'model': format_model_name(best_model),
#                 'metrics': best_model[['Precision (1)', 'Recall (1)', 'F1 (1)']].to_dict()
#             }
#             used_models.add(model_id)
#         else:
#             print(f"No suitable model found for {label} after precision-recall filtering.")

#     # Print the selected metric values for verification
#     for key, value in models.items():
#         print(f"{key}: {value['metrics']} at model - {value['model']}")

#     return models


# calculate metrics
def calculate_metrics(df, rule_result_column):
    TP = ((df[rule_result_column] == 1) & (df['corr'] == 1)).sum()
    FP = ((df[rule_result_column] == 1) & (df['corr'] == 0)).sum()
    FN = ((df[rule_result_column] == 0) & (df['corr'] == 1)).sum()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, recall, precision