from . import models1
from . import models2
from . import rules_generation as rg
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report

# Store predictions in a .csv or .npy file
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

# Create dictionary for metrics
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

# Prior definition - number of class '1' predictions in the test results
def prior(y_test):
    total_spikes_in_test = np.sum(y_test == 1)
    total_data_points_in_test = y_test.shape[0]
    spike_percentage_in_test = (total_spikes_in_test / total_data_points_in_test) 
    return spike_percentage_in_test

# Evaluate dumb model
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
        edcr_results = rg.make_rules(rules, y_test, pred_file_path)
        output_dicts.extend(edcr_results)
      
    output_dicts = pd.DataFrame(output_dicts)
    output_dicts.to_csv(output_file_path)
    return output_dicts
