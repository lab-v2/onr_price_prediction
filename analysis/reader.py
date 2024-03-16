import os

def map_base_model_to_rule_name(base_model_file_name):
    """
    Maps the base model file name to the naming convention used in the rule result files.
    """
    base_model_file_name = os.path.basename(base_model_file_name).replace("_predictions.csv", "")
    if 'LSTM' in base_model_file_name:
        layers = base_model_file_name.split('_')[1]
        return f"LSTM_{layers}"
    elif 'CNN_Attention' in base_model_file_name:
        parts = base_model_file_name.split('_')
        filters = parts[2]
        kernels = parts[4]
        return f"CNNA_{filters}_{kernels}"
    elif 'CNN' in base_model_file_name and 'Attention' not in base_model_file_name:
        parts = base_model_file_name.split('_')
        filters = parts[1]
        kernels = parts[3]
        return f"CNN_{filters}_{kernels}"
    elif 'RNN' in base_model_file_name:
        units = base_model_file_name.split('_')[1]
        return f"RNN_{units}"
    else:
        return "Unknown"
    
def find_matching_rule_file(base_model_file_path, rule_result_dir):
    """
    Finds a file that matches the "Rule all" criteria for the base model.
    """
    base_model_file_name = os.path.basename(base_model_file_path).replace("_predictions.csv", "")
    mapped_base_model_name = map_base_model_to_rule_name(base_model_file_name)

    for file in os.listdir(rule_result_dir):
        if file.endswith(".csv") and "Rule all" in file and mapped_base_model_name in file:
            return os.path.join(rule_result_dir, file)
    return None

def find_matching_rule_file(base_model_file_name, rule_result_dir):
    """
    Finds a file that matches the "Rule all" criteria for the base model.
    """
    mapped_base_model_name = map_base_model_to_rule_name(base_model_file_name)

    for file in os.listdir(rule_result_dir):
        if file.endswith(".csv") and "Rule all" in file and mapped_base_model_name in file:
            return os.path.join(rule_result_dir, file)
    return None