import sys
import os
analysis_dir_path = '../'
sys.path.append(analysis_dir_path)

import models
import pandas as pd
import warnings
import example as ep
warnings.filterwarnings('ignore')

model_check = ep.model_select(f'../cobalt_shift_new_20/test/results_test.csv')
print(model_check)

# %%capture

# COMMODITY = 'nickel_no_val_20'
# MODEL = CNNA
# RULE_NUM = 10
# confidence_levels = [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95]

def exclude_zero(series):
    temp = series.copy()
    temp = temp[temp != 0]
    temp = temp[temp != 1]
    return temp

# A function for providing useful labels for the results
def label(results, row):
    labels = []
    max_base_model_precision = exclude_zero(results["Precision (Base Model)"]).max()
    max_base_model_recall = exclude_zero(results["Recall (Base Model)"]).max()
    max_base_model_f1 = exclude_zero(results["F1 (Base Model)"]).max()
    min_base_model_f1 = exclude_zero(results["F1 (Base Model)"]).min()

    if max_base_model_precision == row["Precision (Base Model)"]: labels.append("Best Precision")
    if max_base_model_recall == row["Recall (Base Model)"]: labels.append("Best Recall")
    if max_base_model_f1 == row["F1 (Base Model)"]: labels.append("Best F1")
    if min_base_model_f1 == row["F1 (Base Model)"]: labels.append("Worst F1")
    return ', '.join(labels)

def evaluate_df(df, properties={}):
    # Base model metrics are now passed directly in properties
    precision = properties.get('Base Precision', 0)
    recall = properties.get('Base Recall', 0)
    f1 = properties.get('Base F1', 0)
    prior = df["corr"].sum() / len(df) if len(df) else 0

    df.to_numpy().dump('data/test.npy')
    df = ep.run_edcr()

    new_precision = df.iloc[50]["pre"] if len(df) > 50 else 0
    new_recall = df.iloc[50]["recall"] if len(df) > 50 else 0
    new_f1 = df.iloc[50]["F1"] if len(df) > 50 else 0

    percent_precision = ((new_precision - precision) / precision) if precision != 0 else 0
    percent_recall = ((new_recall - recall) / recall) if recall != 0 else 0
    percent_f1 = ((new_f1 - f1) / f1) if f1 != 0 else 0

    return {
        **properties,
        "Precision (Base Model)": precision,
        "Recall (Base Model)": recall,
        "F1 (Base Model)": f1,
        "Prior": prior,
        "Precision (EDCR)": new_precision,
        "Recall (EDCR)": new_recall,
        "F1 (EDCR)": new_f1,
        "Precision Improvement": new_precision - precision,
        "Recall Improvement": new_recall - recall,
        "F1 Improvement": new_f1 - f1,
        "Precision Improvement (%)": percent_precision,
        "Recall Improvement (%)": percent_recall,
        "F1 Improvement (%)": percent_f1
    }

for COMMODITY in [
    # 'cobalt_shift_20', 
    # 'copper_shift_20', 'magnesium_shift_20', 'nickel_shift_20',
    'cobalt_shift_new_20', 
    # 'copper_shift_new_20', 'magnesium_shift_new_20', 'nickel_shift_new_20',
]:
    results = []
    model_metrics = ep.model_select(f'../{COMMODITY}/test/results_test.csv')
    print(f'({COMMODITY}): {model_metrics}')

    for model_key, model_info in model_metrics.items():
        model_name = model_info['model']
        base_metrics = model_info['metrics']

        for ALGO in ['correction', 'detection_correction']:
            # for THRESHOLD in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            for THRESHOLD in [0.1]:
                df_path = f'../{COMMODITY}/test/predictions/test/{model_name}.csv'
                df = models.npy_to_threshold_f1_bowpy(df_path, f'../{COMMODITY}/test/predictions/test_{ALGO}', THRESHOLD, exclude_models=[])

                # Pass the model's base metrics along with other properties to evaluate_df
                properties = {
                    "Model": model_name,
                    "Algorithm": ALGO,
                    "Threshold": THRESHOLD,
                    "Base Precision": base_metrics['Precision (1)'],
                    "Base Recall": base_metrics['Recall (1)'],
                    "Base F1": base_metrics['F1 (1)'],
                }
                result = evaluate_df(df, properties)
                results.append(result)

    results_df = pd.DataFrame(results)
    results_df['Label'] = results_df.apply(lambda x: label(results_df, x), axis=1)
    results_df.to_excel(f'out/threshold/{COMMODITY}_test_results.xlsx', index=False)
