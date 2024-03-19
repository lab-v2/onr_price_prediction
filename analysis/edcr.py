import numpy as np
import pandas as pd
import models
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

DUMB_MODELS = ['dumb_spikes', 'dumb_non_spikes']

# Support (s): fraction of samples in O where the body is true.
def calculate_support(y_pred, class_of_interest=1):
    return np.mean(y_pred == class_of_interest)

# Confidence (c): the number of times the body and head are true together divided by the number of times the body is true.
def calculate_confidence(y_pred, y_test, class_of_interest=1):
    true_positives = np.logical_and(y_pred == class_of_interest, y_test == class_of_interest).sum()
    predicted_positives = (y_pred == class_of_interest).sum()
    confidence = true_positives / predicted_positives if predicted_positives > 0 else 0
    return confidence

def evaluate_edcr(name, rule_model_pred, base_model_pred, y_test):
    print(rule_model_pred.shape,type(rule_model_pred))
    print(base_model_pred.shape,type(base_model_pred))

    y_pred = np.squeeze(rule_model_pred) | np.squeeze(base_model_pred)
    output = models.make_output_dict("EDCR", name, classification_report(y_test, y_pred, output_dict=True), models.prior(y_test))
    return y_pred, output

def evaluate_edcr_detection(name, rule_model_pred, base_model_pred, y_test):
    print(rule_model_pred.shape,type(rule_model_pred))
    print(base_model_pred.shape,type(base_model_pred))

    rule_model_pred = np.squeeze(rule_model_pred)
    base_model_pred = np.squeeze(base_model_pred)

    # Use detection to flag incorrect predictions
    error_flags = base_model_pred != rule_model_pred


    y_pred = np.copy(base_model_pred)
    # y_pred[error_flags] = 1 - y_pred[error_flags] # flip prediction is rule says its wrong
    y_pred[error_flags] = rule_model_pred[error_flags] | y_pred[error_flags]    # Keep class 1 predictions, even if rule says its wrong.

    # y_pred = np.squeeze(rule_model_pred) | np.squeeze(base_model_pred)
    output = models.make_output_dict("EDCR", name, classification_report(y_test, y_pred, output_dict=True), models.prior(y_test))

    # Generate classification report
    return y_pred, output

def edcr_evaluation_method(method, name, rule_pred, base_pred, y_test, pred_file_path, metric_name):
    if method == 'correction':
        y_pred, output_dict = evaluate_edcr(name, rule_pred, base_pred, y_test)
    elif method == 'detection_correction':
        y_pred, output_dict = evaluate_edcr_detection(name, rule_pred, base_pred, y_test)
    else:
        raise ValueError("Unknown method specified")

    # Save the prediction and evaluation results
    models.save_predictions_to_file(name, y_pred, y_test, f"{pred_file_path}_{metric_name}_{method}")
    return output_dict

def apply_edcr(rules, y_test, pred_file_path):
    output_dicts = []

    for base_idx in range(len(rules)):  # Loop through all models as potential base models
        base_model = rules[base_idx]
        base_model_name = base_model[2]  
        print(f"Base model: {base_model_name}")

        # Exclude the current base model from the selection for F1, Recall, and Precision
        f1_df = pd.DataFrame([x[4] for idx, x in enumerate(rules) if idx != base_idx])
        sorted_f1 = list(f1_df.sort_values(by=0, ascending=False).head(5).index)

        rec_df = pd.DataFrame([x[5] for idx, x in enumerate(rules) if idx != base_idx])
        sorted_rec = list(rec_df.sort_values(by=0, ascending=False).head(5).index)

        prec_df = pd.DataFrame([x[6] for idx, x in enumerate(rules) if idx != base_idx])
        sorted_prec = list(prec_df.sort_values(by=0, ascending=False).head(5).index)

        rules_index = {
            'F1': sorted_f1,
            'Recall': sorted_rec,
            'Precision': sorted_prec,
        }

        # Just to clear up confusion
        # base_model = rules[best_acc_index]
        # base_model_pred = rules[best_acc_index][0]
        # base_model_pred_confidence = rules[best_acc_index][1]
        # base_model_name = rules[best_acc_index][2]
        
        for metric_name, metric_rule_index in rules_index.items():
            print(f"\nEvaluating rules based on {metric_name}:")

            for confident in [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95]:
                y_pred_all = []

                # Use a model (with high accuracy) as baseline and use high [OPTIMAL_METRIC] models to improve it
                for ri in metric_rule_index:

                    # Just to clear up confusion
                    # rule_model = rules[ri]
                    # rule_model_pred = rule_model[0]
                    # rule_model_pred_confidence = rule_model[1]
                    # rule_model_name = rule_model[2]

                    name = "Confident " + str(confident) + "Rule " + rules[ri][2] + "for " + rules[base_idx][2]
                    try:
                        y_pred1 = (rules[ri][1] > confident).astype(int)
                        if(len(y_pred_all)):
                            y_pred_all = np.squeeze(y_pred_all) | np.squeeze(y_pred1)
                        else:
                            y_pred_all = y_pred1

                        # Evaluate using correction rule
                        output_dict_direct = edcr_evaluation_method('correction', name, y_pred1, rules[base_idx][0], y_test, pred_file_path, metric_name)
                        output_dicts.append(output_dict_direct)

                        # Evaluate using detection + correction rule
                        output_dict_detection = edcr_evaluation_method('detection_correction', name, y_pred1, rules[base_idx][0], y_test, pred_file_path, metric_name)
                        output_dicts.append(output_dict_detection)

                    except Exception as e:
                        print(f"Failed to evaluate {name}: {e}")
                
                # Use a model as baseline with ensemble to improve it
                # Perhaps scrap this?
                name = "Confident " + str(confident) + "Rule all" + "for " + rules[base_idx][2]
                try:
                    # Evaluate using correction rule
                    output_dict_direct = edcr_evaluation_method('correction', name, y_pred_all, rules[base_idx][0], y_test, pred_file_path, metric_name)
                    output_dicts.append(output_dict_direct)

                    # Evaluate using detection + correction rule
                    output_dict_detection = edcr_evaluation_method('detection_correction', name, y_pred_all, rules[base_idx][0], y_test, pred_file_path, metric_name)
                    output_dicts.append(output_dict_detection)
                except Exception as e:
                    print(f"Failed to evaluate {name}: {e}")  

                # Use dumb model as baseline with ensemble to improve it
                for dumb in DUMB_MODELS:
                    name = "Confident " + str(confident) + "Rule all" + "for " + dumb
                    try:
                        pred_value = 1 if dumb == 'dumb_spikes' else 0
                        
                        y_pred_dumb = pd.Series([pred_value] * len(y_test))

                        # Evaluate using correction rule
                        output_dict_direct = edcr_evaluation_method('correction', name, y_pred_all, y_pred_dumb, y_test, pred_file_path, metric_name)
                        output_dicts.append(output_dict_direct)

                        # Evaluate using detection + correction rule
                        output_dict_detection = edcr_evaluation_method('detection_correction', name, y_pred_all, y_pred_dumb, y_test, pred_file_path, metric_name)
                        output_dicts.append(output_dict_detection)
                    except Exception as e:
                        print(f"Failed to evaluate {name}: {e}")

    return output_dicts



