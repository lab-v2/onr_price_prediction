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
        npy_filename = f"{directory_path}/{model_name}_predictions.npy"
        np.save(npy_filename, y_pred)

        print(f"Predictions saved to CSV file: {csv_filename}")
        print(f"Predictions saved to NPY file: {npy_filename}")
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

# def npy_to_bowpy(base_model_file_path, rule_result_dir, confidence_levels):
#     """
#     Merges base model predictions with the corresponding rule results for specified confidence levels.
#     """
#     # Read the base model predictions
#     bowpy_dataframe = pd.read_csv(base_model_file_path)
#     bowpy_dataframe.rename(columns={"Predicted": "pred", "True": "corr"}, inplace=True)
#     bowpy_dataframe['true_positive'] = bowpy_dataframe.apply(lambda x: 1 if x['pred'] == 1 and x['corr'] == 1 else 0, axis=1)
#     bowpy_dataframe['false_positive'] = bowpy_dataframe.apply(lambda x: 1 if x['pred'] == 1 and x['corr'] == 0 else 0, axis=1)

#     base_model_name = os.path.basename(base_model_file_path).replace("_predictions.csv", "")
#     mapped_base_model_name = reader.map_base_model_to_rule_name(base_model_name)

#     # Iterate through the directory and add each matching rule's predictions as a new column
#     for model_file in os.listdir(rule_result_dir):
#         if model_file.endswith(".csv") and "Rule all" in model_file and mapped_base_model_name in model_file:
#             column_name = reader.extract_rule_confidence(model_file, confidence_levels)
#             if column_name:
#                 model_predictions = pd.read_csv(os.path.join(rule_result_dir, model_file))
#                 bowpy_dataframe[column_name] = model_predictions['Predicted']

#     return bowpy_dataframe

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

# Function to parse model descriptor and return appropriate model architecture
def get_model_from_descriptor(descriptor, input_shape):
    adam_optimizer = Adam(learning_rate=learning_rate)
    # LSTM Model
    if "LSTM" in descriptor:
        num_layers = int(re.search(r"LSTM_(\d+)_layers", descriptor).group(1))
        model = Sequential()
        model.add(LSTM(num_layers, input_shape=input_shape, activation='relu'))
        model.add(Dense(num_layers // 2, activation='relu'))
        model.add(Dense(num_layers // 2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    
    # RNN Model
    elif "RNN" in descriptor:
        num_units = int(re.search(r"RNN_(\d+)_units", descriptor).group(1))
        model = Sequential()
        model.add(SimpleRNN(num_units, input_shape=input_shape, activation='relu'))
        model.add(Dense(num_units // 2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    
    # CNN Model
    elif "CNN" in descriptor and "Attention" not in descriptor:
        num_filters, kernel_size = map(int, re.search(r"CNN_(\d+)_filters_(\d+)_kernels", descriptor).groups())
        model = Sequential()
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(num_filters // 2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    # CNN with Attention Model
    elif "CNN_Attention" in descriptor:
        num_filters, kernel_size = map(int, re.search(r"CNN_Attention_(\d+)_filters_(\d+)_kernels", descriptor).groups())
        model = create_acnn_model2(input_shape, 1, num_filters, kernel_size)
    
    else:
        raise ValueError("Model descriptor not recognized.")

    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Retrain the best performing model
def retrain_best_model(saved_model_path, X_train, y_train, X_val, y_val, X_test, y_test):
    # Extract model descriptor from file path
    descriptor = saved_model_path.split('/')[-1].replace('.h5', '')
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Rebuild model based on the descriptor
    model = get_model_from_descriptor(descriptor, input_shape)
    
    # Retrain the model
    early_stopping = EarlyStopping(monitor='val_auc', patience=50, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=2000, batch_size=32, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    # Evaluate the retrained model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    output = make_output_dict("Retrained Model", descriptor, classification_report(y_test, y_pred, output_dict=True), prior(y_test))
    output_dicts = pd.DataFrame([output])
    
    return output_dicts

# Transformer
def positional_encoding(length, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(length)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    
    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
      
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_encoder(inputs, num_heads, ff_dim, dropout=0.1):
    # Multi-head self-attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed-forward layer
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    return LayerNormalization(epsilon=1e-6)(attention_output + ff_output)


def build_transformer(num_encoder_layers, length, d_model):
    inputs = Input(shape=(length, d_model))
    x = inputs + positional_encoding(length, d_model)

    for _ in range(num_encoder_layers):
        x = transformer_encoder(x, num_heads=4, ff_dim=128)

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Evaluate Transformer model
def evaluate_transformer(num_encoder_layers,length,d_model,X_train, y_train, X_test, y_test):
    model = build_transformer(num_encoder_layers, length, d_model)
    model.fit(X_train, y_train, epochs=2000, batch_size=32, verbose=False)

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_pred_ori = model.predict(X_test) 
    
    report = classification_report(y_test, y_pred, output_dict=True)
    output = make_output_dict(f"Transformer", f"{num_encoder_layers} encoder layers", report)
    return y_pred, output, y_pred_ori


# #SVM Model
# def evaluate_svm(param_grid, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path):
#     X_train = X_train.reshape(X_train.shape[0], -1)
#     y_train = y_train.reshape(y_train.shape[0], -1)

#     # Initialize the SVM model
#     model = SVC(probability=True)
#     grid_search = GridSearchCV(model, param_grid, scoring='precision', cv=5, verbose=0)
#     try:
#         grid_search.fit(X_train, y_train)
#     except Exception as e:
#         print(f'flop: {e}')

#     best_model = grid_search.best_estimator_

    
#     # Predictions with the best model
#     y_pred_ori = best_model.predict_proba(X_test)[:, 1]
#     y_pred = (y_pred_ori > 0.5).astype(int)

#     best_params_str = ', '.join(f"{key}: {val}" for key, val in grid_search.best_params_.items())

#     output = make_output_dict("SVM",best_params_str, classification_report(y_test, y_pred, output_dict=True), prior(y_test))
#     return y_pred, output, y_pred_ori, output['Recall (1)'], best_model

def evaluate_dumb_model(y_test, model_type='non_spikes'):
    y_pred = np.ones(len(y_test), dtype=int) if model_type == 'spikes' else np.zeros(len(y_test), dtype=int)
    output_dict = make_output_dict("Dumb Model", model_type, classification_report(y_test, y_pred, output_dict=True), prior(y_test))
    
    return y_pred, output_dict

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
        edcr_results = edcr.apply_edcr(rules, y_test, pred_file_path)
        output_dicts.extend(edcr_results)
      
    output_dicts = pd.DataFrame(output_dicts)
    output_dicts.to_csv(output_file_path)
    return output_dicts
    
