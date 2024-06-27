import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Dropout, LayerNormalization, MultiHeadAttention, Input, LSTM, SimpleRNN
from tensorflow.keras.layers import Attention, Reshape
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam
import re
import os
import reader
import edcr
import models2
import models1

DUMB_MODELS = ['dumb_spikes', 'dumb_non_spikes']
learning_rate = 0.0001
auc_count = 0
auc = tf.keras.metrics.AUC()

def save_predictions_to_file(model_name, y_pred, y_test, directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)

        # Check and adjust shapes. Flatten if 2D with one column, otherwise respect multi-dimensionality for multi-class
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        if y_test.ndim > 1 and y_test.shape[1] == 1:
            y_test = y_test.flatten()

        # Handling for multi-class classification where y_pred is not 1-dimensional
        if y_pred.ndim > 1:
            # For multi-dimensional y_pred, create a DataFrame with a column for each class/label
            predictions_df = pd.DataFrame(y_pred, columns=[f'Predicted_{i}' for i in range(y_pred.shape[1])])
            predictions_df['True'] = y_test  
        else:
            # For 1-dimensional y_pred, proceed as before
            predictions_df = pd.DataFrame({'Predicted': y_pred, 'True': y_test})

        csv_filename = f"{directory_path}/{model_name}_predictions.csv"
        predictions_df.to_csv(csv_filename, index=False)

        # Optionally, save y_pred with original shape to NPY if it's crucial to preserve multi-dimensionality
        # npy_filename = f"{directory_path}/{model_name}_predictions.npy"
        # np.save(npy_filename, y_pred)

        print(f"Predictions saved to CSV file: {csv_filename}")
        # print(f"Predictions saved to NPY file: {npy_filename}")
    except Exception as e:
        print(f"2. Failed to save predictions for {model_name}: {e}")

def make_output_dict(name, params, classification_report, prior):
    return {
        "Name": name,
        "Params": params,
        "Accuracy": classification_report["accuracy"],
        "Precision (0)": classification_report["0"]["precision"],
        "Recall (0)": classification_report["0"]["recall"],
        "F1 (0)": classification_report["0"]["f1-score"],
        "Precision (1)": classification_report["1"]["precision"],
        "Recall (1)": classification_report["1"]["recall"],
        "F1 (1)": classification_report["1"]["f1-score"],
        "Prior": f"{prior:.2f}"
    }

def prior(y_test):
    total_spikes_in_test = np.sum(y_test == 1)
    total_data_points_in_test = y_test.shape[0]
    spike_percentage_in_test = (total_spikes_in_test / total_data_points_in_test) 
    return spike_percentage_in_test

def npy_to_based_bowpy(base_model_file_path, rule_result_dir, confidence_levels):
    """
    Merges base model predictions with the corresponding rule results for specified confidence levels.
    """
    # Read the base model predictions
    bowpy_dataframe = pd.read_csv(base_model_file_path)
    bowpy_dataframe.rename(columns={"Predicted": "pred", "True": "corr"}, inplace=True)
    bowpy_dataframe['true_positive'] = bowpy_dataframe.apply(lambda x: 1 if x['pred'] == 1 and x['corr'] == 1 else 0, axis=1)
    bowpy_dataframe['false_positive'] = bowpy_dataframe.apply(lambda x: 1 if x['pred'] == 1 and x['corr'] == 0 else 0, axis=1)

    base_model_name = os.path.basename(base_model_file_path).replace("_predictions.csv", "")
    mapped_base_model_name = reader.map_base_model_to_rule_name(base_model_name)

    # Iterate through the directory and add each matching rule's predictions as a new column
    index = 0
    for model_file in os.listdir(rule_result_dir):
        if model_file.endswith(".csv") and not "Rule all" in model_file and mapped_base_model_name in model_file:
            
            model_predictions = pd.read_csv(os.path.join(rule_result_dir, model_file))
            bowpy_dataframe[f"rule{index}"] = model_predictions['Predicted']
            index += 1

    return bowpy_dataframe

def npy_to_bowpy(base_model_file_path, base_dir, confidence_levels, algo):
    """
    Merges base model predictions with the corresponding rule results for specified confidence levels across multiple directories.
    """
    metrics = ['F1', 'Recall', 'Precision']
    # algos = ['correction', 'detection_correction']

    # Read the base model predictions
    bowpy_dataframe = pd.read_csv(base_model_file_path)
    bowpy_dataframe.rename(columns={"Predicted": "pred", "True": "corr"}, inplace=True)
    bowpy_dataframe['true_positive'] = bowpy_dataframe.apply(lambda x: 1 if x['pred'] == 1 and x['corr'] == 1 else 0, axis=1)
    bowpy_dataframe['false_positive'] = bowpy_dataframe.apply(lambda x: 1 if x['pred'] == 1 and x['corr'] == 0 else 0, axis=1)

    base_model_name = os.path.basename(base_model_file_path).replace("_predictions.csv", "")
    mapped_base_model_name = reader.map_base_model_to_rule_name(base_model_name)

    # Loop through each combination of metric and algo
    for metric in metrics:
        # for algo in algos:
        dir_name = f'test_{metric}_{algo}'
        rule_result_dir = os.path.join(base_dir, dir_name)
        if os.path.isdir(rule_result_dir):
            # Iterate through files in the directory and add matching rule's predictions as a new column
            for model_file in os.listdir(rule_result_dir):
                # if model_file.endswith(".csv") and "Rule all" in model_file and mapped_base_model_name in model_file:
                if model_file.endswith(".csv") and "all" in model_file and mapped_base_model_name in model_file:
                    confidence_column = reader.extract_rule_confidence(model_file, confidence_levels)
                    if confidence_column:
                        full_path = os.path.join(rule_result_dir, model_file)
                        model_predictions = pd.read_csv(full_path)
                        column_name = f"{dir_name}_{confidence_column}"
                        bowpy_dataframe[column_name] = model_predictions['Predicted']

    return bowpy_dataframe

def npy_to_top_n_f1_bowpy(base_model_file_path, rule_result_dir, top_n):
    """
    Merges base model predictions with the corresponding rule results for specified confidence levels.
    """
    # Read the base model predictions
    bowpy_dataframe = pd.read_csv(base_model_file_path)
    bowpy_dataframe.rename(columns={"Predicted": "pred", "True": "corr"}, inplace=True)
    bowpy_dataframe['true_positive'] = bowpy_dataframe.apply(lambda x: 1 if x['pred'] == 1 and x['corr'] == 1 else 0, axis=1)
    bowpy_dataframe['false_positive'] = bowpy_dataframe.apply(lambda x: 1 if x['pred'] == 1 and x['corr'] == 0 else 0, axis=1)

    base_model_name = os.path.basename(base_model_file_path).replace("_predictions.csv", "")
    mapped_base_model_name = reader.map_base_model_to_rule_name(base_model_name)

    # Iterate through the directory and add each matching rule's predictions as a new column
    index = 0
    for model_file in os.listdir(rule_result_dir):
        if model_file.endswith(".csv") and not "Rule all" in model_file and mapped_base_model_name in model_file:
            index += 1

    rank = []
    index = 0
    for model_file in os.listdir(rule_result_dir):
        if model_file.endswith(".csv") and not "Rule all" in model_file and mapped_base_model_name in model_file:
            # print(model_file)
            model_predictions = pd.read_csv(os.path.join(rule_result_dir, model_file))
            if len(model_predictions['Predicted']) == len(bowpy_dataframe['pred']):
                rank += [(model_file, model_predictions['Predicted'], classification_report(model_predictions['True'], model_predictions['Predicted'], output_dict=True)['1']['f1-score'])]
                index += 1
    sorted_rank = sorted(rank, key=lambda x: x[2], reverse=True)
    
    top_n = min(top_n, index)
    for i in range(top_n):
        model_file = sorted_rank[i][0]
        model_predictions = sorted_rank[i][1]
        bowpy_dataframe[f"rule{i}"] = model_predictions

    return bowpy_dataframe

# Select models based on F1 threshold, and includes functionality to exclude models
def npy_to_threshold_f1_bowpy(base_model_file_path, rule_result_dir, threshold, exclude_models=[]):
    
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
            model_details = model_file.split('Rule')[1].split('for')[0].strip()
            # print(model_details)
            model_predictions = pd.read_csv(os.path.join(rule_result_dir, model_file))
            # print('1',len(model_predictions['Predicted']))
            # print('2', len(bowpy_dataframe['pred']))
            if len(model_predictions['Predicted']) == len(bowpy_dataframe['pred']):
                f1_score = classification_report(model_predictions['True'], model_predictions['Predicted'], output_dict=True)['1']['f1-score']
                if any(excl in model_details for excl in exclude_models):
                    ablation_filter += 1
                    # print(f"Excluded model: {model_details}, F1 Score: {f1_score}")
                    continue
                if f1_score >= threshold:
                    bowpy_dataframe[f"rule{rule_index}"] = model_predictions['Predicted']
                    rule_index += 1
                else: 
                    f1_filter += 1
            else:
                scuffed_filter += 1
    if rule_index == 0: return pd.DataFrame()
    print (f'Excluded {f1_filter} rules due to F1 filtering')
    print (f'Excluded {ablation_filter} rules due to model filtering')
    print (f'Excluded {scuffed_filter} rules due to problematic formatting')
    return bowpy_dataframe

# Function to parse model descriptor and return appropriate model architecture
# def get_model_from_descriptor(descriptor, input_shape):
#     adam_optimizer = Adam(learning_rate=learning_rate)
#     # LSTM Model
#     if "LSTM" in descriptor:
#         num_layers = int(re.search(r"LSTM_(\d+)_layers", descriptor).group(1))
#         model = Sequential()
#         model.add(LSTM(num_layers, input_shape=input_shape, activation='relu'))
#         model.add(Dense(num_layers // 2, activation='relu'))
#         model.add(Dense(num_layers // 2, activation='relu'))
#         model.add(Dense(1, activation='sigmoid'))
    
#     # RNN Model
#     elif "RNN" in descriptor:
#         num_units = int(re.search(r"RNN_(\d+)_units", descriptor).group(1))
#         model = Sequential()
#         model.add(SimpleRNN(num_units, input_shape=input_shape, activation='relu'))
#         model.add(Dense(num_units // 2, activation='relu'))
#         model.add(Dense(1, activation='sigmoid'))
    
#     # CNN Model
#     elif "CNN" in descriptor and "Attention" not in descriptor:
#         num_filters, kernel_size = map(int, re.search(r"CNN_(\d+)_filters_(\d+)_kernels", descriptor).groups())
#         model = Sequential()
#         model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
#         model.add(MaxPooling1D(pool_size=2))
#         model.add(Flatten())
#         model.add(Dense(num_filters // 2, activation='relu'))
#         model.add(Dense(1, activation='sigmoid'))

#     # CNN with Attention Model
#     elif "CNN_Attention" in descriptor:
#         num_filters, kernel_size = map(int, re.search(r"CNN_Attention_(\d+)_filters_(\d+)_kernels", descriptor).groups())
#         model = create_acnn_model2(input_shape, 1, num_filters, kernel_size)
    
#     else:
#         raise ValueError("Model descriptor not recognized.")

#     model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# Retrain the best performing model
# def retrain_best_model(saved_model_path, X_train, y_train, X_val, y_val, X_test, y_test):
#     # Extract model descriptor from file path
#     descriptor = saved_model_path.split('/')[-1].replace('.h5', '')
#     input_shape = (X_train.shape[1], X_train.shape[2])
    
#     # Rebuild model based on the descriptor
#     model = get_model_from_descriptor(descriptor, input_shape)
    
#     # Retrain the model
#     early_stopping = EarlyStopping(monitor='val_auc', patience=50, restore_best_weights=True)
#     model.fit(X_train, y_train, epochs=2000, batch_size=32, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
#     # Evaluate the retrained model
#     y_pred = (model.predict(X_test) > 0.5).astype(int)
#     output = make_output_dict("Retrained Model", descriptor, classification_report(y_test, y_pred, output_dict=True), prior(y_test))
#     output_dicts = pd.DataFrame([output])
    
#     return output_dicts

def evaluate_dumb_model(y_test, model_type='non_spikes'):
    y_pred = np.ones(len(y_test), dtype=int) if model_type == 'spikes' else np.zeros(len(y_test), dtype=int)
    output_dict = make_output_dict("Dumb Model", model_type, classification_report(y_test, y_pred, output_dict=True), prior(y_test))
    
    return y_pred, output_dict

# Added for modularity
# def evaluate_model(model_type, params, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, saved_model_path, pred_file_path, confidence_path, val=True):
#     model_descriptor = f"{model_type}_{params}"
#     model_path = f'{saved_model_path}/{model_descriptor}.h5'
#     try:
#         if val:
#             y_pred, output_dict, y_pred_conf, model = models1.evaluate(model_type, params, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path)
#         else:
#             y_pred, output_dict, y_pred_conf, model = models2.evaluate(model_type, params, X_train, y_train, X_test, y_test, pretrain, model_path)
#         save_predictions_to_file(model_descriptor, y_pred, y_test, pred_file_path)
#         save_predictions_to_file(model_descriptor, y_pred_conf, y_test, confidence_path)
#         model.save(model_path)
#         print(f"{model_descriptor} evaluation successful:", output_dict)
#         return [y_pred, y_pred_conf, model_descriptor, output_dict['Accuracy'], output_dict['F1 (1)'], output_dict['Recall (1)'], output_dict['Precision (1)']], output_dict
#     except Exception as e:
#         print(f"Failed to evaluate {model_descriptor}: {e}")
#         return None, None

# Evaluate all models
def evaluate_all(X_train, y_train, X_val, y_val, X_test, y_test, output_file_path, pred_file_path, saved_model_path, pretrain, edcra=True, val=True):
    global auc_count
    output_dicts = []
    
    # FNN
    rules = []
    results = []
    auc_count = 0

    pred_confidence_path = f"{pred_file_path}_confidence"


    # LSTM
    for layers in [256, 128, 64, 32]:
        try: 
            model_descriptor = f"LSTM_{layers}_layers"
            model_path = f'{saved_model_path}/{model_descriptor}.h5'
            if val:
                y_pred, output_dict, y_pred_lstm, model = models1.evaluate_lstm(layers, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path)
            else:
                y_pred, output_dict, y_pred_lstm, model = models2.evaluate_lstm(layers, X_train, y_train, X_test, y_test, pretrain, model_path)
            rules.append([y_pred, y_pred_lstm, f"LSTM_{layers}", output_dict['Accuracy'], output_dict['F1 (1)'], output_dict['Recall (1)'], output_dict['Precision (1)']]) 
            output_dicts.append(output_dict)
            save_predictions_to_file(model_descriptor, y_pred, y_test, pred_file_path)
            save_predictions_to_file(model_descriptor, y_pred_lstm, y_test, pred_confidence_path)

            # Check if this model has the best accuracy so far
            print(output_dict)
            model.save(model_path)
        except Exception as e:
            print(f"Failed to evaluate LSTM with {layers} layers: {e}")

    # CNN w/ Attention
    for filter in [32, 64, 128, 256]:
        for kernel in [7,5,3]:
            try:
                model_descriptor = f"CNN_Attention_{filter}_filters_{kernel}_kernels"
                model_path = f'{saved_model_path}/{model_descriptor}.h5'
                if val:
                    y_pred, output_dict, y_pred_cnna, model  = models1.evaluate_attention_cnn2(filter, kernel, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path) # Switch this later?
                else:
                    y_pred, output_dict, y_pred_cnna, model  = models2.evaluate_attention_cnn2(filter, kernel, X_train, y_train, X_test, y_test, pretrain, model_path) # Switch this later?
                #rules.append([y_pred_cnna, f"CNNA_{filter}_{kernel}"]) 
                rules.append([y_pred, y_pred_cnna, f"CNNA_{filter}_{kernel}", output_dict['Accuracy'], output_dict['F1 (1)'], output_dict['Recall (1)'], output_dict['Precision (1)']]) 
                output_dicts.append(output_dict)
                save_predictions_to_file(model_descriptor, y_pred, y_test, pred_file_path)
                save_predictions_to_file(model_descriptor, y_pred_cnna, y_test, pred_confidence_path)

                # Check if this model has the best accuracy so far
                print(f"CNN_Attention_{filter}_filters_{kernel}_kernels",output_dict)
                model.save(model_path)
            except Exception as e:
                print(f"Failed to evaluate CNN with Attention {filter} filters and {kernel} kernel size: {e}")
    # RNN 
    for units in [256, 128, 64, 32]:
        try: 
            model_descriptor = f"RNN_{units}_units"
            model_path = f'{saved_model_path}/{model_descriptor}.h5'
            if val:
                y_pred, output_dict, y_pred_rnn, model  = models1.evaluate_rnn(units, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path)
            else:
                y_pred, output_dict, y_pred_rnn, model  = models2.evaluate_rnn(units, X_train, y_train, X_test, y_test, pretrain, model_path)
            #rules.append([y_pred_rnn, f"RNN_{units}"]) 
            rules.append([y_pred, y_pred_rnn, f"RNN_{units}", output_dict['Accuracy'], output_dict['F1 (1)'], output_dict['Recall (1)'], output_dict['Precision (1)']]) 
            output_dicts.append(output_dict)
            save_predictions_to_file(model_descriptor, y_pred, y_test, pred_file_path)
            save_predictions_to_file(model_descriptor, y_pred_rnn, y_test, pred_confidence_path)
        
            # Check if this model has the best accuracy so far
            print(f"RNN_{units}_units:",output_dict)
            model.save(model_path)
        except Exception as e:
            print(f"Failed to evaluate RNN with {units} units: {e}")

    # CNN
    for filter in [32, 64, 128, 256]:
        for kernel in [7,5,3]:
            try:
                model_descriptor = f"CNN_{filter}_filters_{kernel}_kernels"
                model_path = f'{saved_model_path}/{model_descriptor}.h5'
                if val:
                    y_pred, output_dict, y_pred_cnn, model  = models1.evaluate_cnn(filter, kernel, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path)
                else:
                    y_pred, output_dict, y_pred_cnn, model  = models2.evaluate_cnn(filter, kernel, X_train, y_train, X_test, y_test, pretrain, model_path)
                #rules.append([y_pred_cnn, f"CNN_{filter}_{kernel}"]) 
                rules.append([y_pred, y_pred_cnn, f"CNN_{filter}_{kernel}", output_dict['Accuracy'], output_dict['F1 (1)'], output_dict['Recall (1)'], output_dict['Precision (1)']]) 
                output_dicts.append(output_dict)
                save_predictions_to_file(model_descriptor, y_pred, y_test, pred_file_path)
                save_predictions_to_file(model_descriptor, y_pred_cnn, y_test, pred_confidence_path)


                # Check if this model has the best accuracy so far
                print(f"CNN_{filter}_filters_{kernel}_kernels",output_dict)
                model.save(model_path)
            except Exception as e:
                print(f"Failed to evaluate CNN with {filter} filters and {kernel} kernel size: {e}")

    # Dumb models
    for dm in ['spikes', 'non_spikes']:
        y_pred, output_dict = evaluate_dumb_model(y_test, model_type=dm)
        output_dicts.append(output_dict)
        save_predictions_to_file(f'Dumb_Model_{dm}', y_pred, y_test, pred_file_path)

    # EDCR        
    if edcra:
        edcr_results = edcr.apply_edcr_v2(rules, y_test, pred_file_path)
        output_dicts.extend(edcr_results)
      
    output_dicts = pd.DataFrame(output_dicts)
    output_dicts.to_csv(output_file_path)
    return output_dicts
    