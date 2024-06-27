import numpy as np
import pandas as pd
from . import eval
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

DUMB_MODELS = ['dumb_spikes', 'dumb_non_spikes']

# Flip base model's prediction of '0' to '1', if the rule model's pred confidence exceeds threshold
def evaluate_rule(name, rule_model_pred, base_model_pred, y_test):
    print(rule_model_pred.shape,type(rule_model_pred))
    print(base_model_pred.shape,type(base_model_pred))

    y_pred = np.squeeze(rule_model_pred) | np.squeeze(base_model_pred)
    output = eval.make_output_dict("EDCR", name, classification_report(y_test, y_pred, output_dict=True), eval.prior(y_test))
    return y_pred, output

# A more nuanced version of "evaluate_rule". The idea is that the base model's prediction of 0 may not always be worth flipping
def evaluate_rule_detection(name, rule_model_pred, base_model_pred, rule_model_pred_conf, base_model_pred_conf, y_test):
    print(rule_model_pred.shape,type(rule_model_pred))
    print(base_model_pred.shape,type(base_model_pred))

    rule_model_pred = np.squeeze(rule_model_pred)
    base_model_pred = np.squeeze(base_model_pred)
    rule_model_pred_conf = np.squeeze(rule_model_pred_conf)
    base_model_pred_conf = np.squeeze(base_model_pred_conf)

    # for change condition 2
    confidence_difference_threshold = 0.3

    # Only flip 0 -> 1, if even base model "incorrectly" predicts 1, we keep it
    change_condition_0 = base_model_pred != rule_model_pred

    # Only flip 0 -> 1, if rule says its 1 AND base model's confidence is low
    change_condition_1 = (rule_model_pred == 1) & (base_model_pred == 0) & (base_model_pred_conf > 0.25)

    # Only flip 0 -> 1, if rule says its 1 AND if rule model is significantly more confident than base model
    change_condition_2 = (rule_model_pred == 1) & (base_model_pred == 0) & (rule_model_pred - base_model_pred > confidence_difference_threshold)

    y_pred = np.copy(base_model_pred)
    # y_pred[change_condition_0] = 1 - y_pred[change_condition_0] # flip prediction is rule says its wrong

    # In our experiments we ended up using condition 1 only.
    y_pred[change_condition_1] = 1
    output = eval.make_output_dict("EDCR", name, classification_report(y_test, y_pred, output_dict=True), eval.prior(y_test))

    # Generate classification report
    return y_pred, output

# Alternate version of function above where we dont filter and we dont produce ensembles
def rule_evaluation_method(method, name, rule_pred, base_pred, rule_model_pred_conf, base_model_pred_conf, y_test, pred_file_path):
    if method == 'correction':
        y_pred, output_dict = evaluate_rule(name, rule_pred, base_pred, y_test)
    elif method == 'detection_correction':
        # y_pred, output_dict = evaluate_rule_detection(name, rule_pred, base_pred, y_test)
        y_pred, output_dict = evaluate_rule_detection(name, rule_pred, base_pred, rule_model_pred_conf, base_model_pred_conf, y_test)
    else:
        raise ValueError("Unknown method specified")

    # Save the prediction and evaluation results
    eval.save_predictions_to_file(name, y_pred, y_test, f"{pred_file_path}_{method}")
    return output_dict



def make_rules(rules, y_test, pred_file_path):
    output_dicts = []

    for base_idx, base_model in enumerate(rules):  # Loop through all models as potential base models
        base_model_pred = base_model[0]
        base_model_pred_conf = base_model[1]
        base_model_name = base_model[2]
        print(f"Base model: {base_model_name}")

        for rule_idx, rule_model in enumerate(rules):
            if rule_idx == base_idx:
                continue  # Skip the rule application for the base model itself

            rule_model_pred_conf = rule_model[1]
            rule_model_name = rule_model[2]

            for confident in [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95]:
                name = f"Confident {confident} Rule {rule_model_name} for {base_model_name}"
                try:
                    # Determine rule-based predictions based on confidence threshold
                    y_pred_rule_based = (rule_model_pred_conf > confident).astype(int)

                    # Evaluate using correction algorithm
                    output_dict_direct = rule_evaluation_method('correction', name, y_pred_rule_based, base_model_pred, rule_model_pred_conf, base_model_pred_conf, y_test, pred_file_path)
                    output_dicts.append(output_dict_direct)

                    # Evaluate using detection + correction algorithm
                    output_dict_detection = rule_evaluation_method('detection_correction', name, y_pred_rule_based, base_model_pred, rule_model_pred_conf, base_model_pred_conf, y_test, pred_file_path)
                    output_dicts.append(output_dict_detection)

                except Exception as e:
                    print(f"Failed to evaluate {name}: {e}")

    return output_dicts

