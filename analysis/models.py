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
import re
import os

OPTIMAL_METRIC = 'F1 (1)'
DUMB_MODELS = ['dumb_spikes', 'dumb_non_spikes']

auc_count = 0
auc = tf.keras.metrics.AUC()
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

def save_predictions_to_file(model_name, y_pred, y_true=None, file_format="npy", commodity=''):
    if file_format == "npy":
        # Save as NumPy binary file
        np.save(f"results/{commodity}/pred/{model_name}_predictions.npy", y_pred)
        if y_true is not None:
            np.save(f"results/{commodity}/true/{model_name}_true_labels.npy", y_true)
    elif file_format == "csv":
        # Save as CSV for easier readability
        df = pd.DataFrame({"Predictions": y_pred.flatten()})
        if y_true is not None:
            df["True Labels"] = y_true
        df.to_csv(f"results/{commodity}/.csv/{model_name}_predictions.csv", index=False)
    else:
        raise ValueError("Unsupported file format. Use 'npy' or 'csv'.")


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


#LSTM Model
def evaluate_lstm(num_layers: int, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path):
    global auc_count
    # Build the LSTM model
    if pretrain:
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(LSTM(num_layers, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
        #model.add(LSTM(num_layers, activation='relu'))
        model.add(Dense(num_layers // 2, activation='relu'))
        model.add(Dense(num_layers // 2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])

        # EarlyStopping callback
    if auc_count == 0:
        mv = f"val_auc"
    else:
        mv = f"val_auc{auc_count}"
    early_stopping = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    auc_count += 1

    # Train the model with validation data
    model.fit(X_train, y_train, epochs=2000, batch_size=32, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_pred_ori = model.predict(X_test)

    # Assuming make_output_dict and prior are defined elsewhere in your code
    output = make_output_dict("LSTM", f"{num_layers} layers", classification_report(y_test, y_pred, output_dict=True), prior(y_test))
    return y_pred, output, y_pred_ori, output['Recall (1)'], model


# Build RNN
def evaluate_rnn(num_units: int, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path):
    global auc_count
    # Build the RNN model
    if pretrain:
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(SimpleRNN(num_units, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
        #model.add(SimpleRNN(num_units, activation='relu'))
        model.add(Dense(int(num_units/2), activation='relu'))  # Ensure num_units/2 is cast to int for layer units
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])

    # EarlyStopping callback
    if auc_count == 0:
        mv = f"val_auc"
    else:
        mv = f"val_auc{auc_count}"
    early_stopping = EarlyStopping(monitor= 'val_loss', patience=50, restore_best_weights=True)
    auc_count += 1


    # Train the model with validation data
    model.fit(X_train, y_train, epochs=2000, batch_size=32, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_pred_ori = model.predict(X_test)

    # Assuming make_output_dict and prior are defined elsewhere in your code
    output = make_output_dict("RNN", f"{num_units} units", classification_report(y_test, y_pred, output_dict=True), prior(y_test))
    return y_pred, output, y_pred_ori, output['Recall (1)'], model

# Build the CNN model
def evaluate_cnn(num_filters: int, kernel_size: int, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path):
    global auc_count

    if pretrain:
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        #model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(num_filters/2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])

    # EarlyStopping callback
    if auc_count == 0:
        mv = f"val_auc"
    else:
        mv = f"val_auc{auc_count}"
    early_stopping = EarlyStopping(monitor= 'val_loss', patience=50, restore_best_weights=True)
    auc_count += 1

    # Train the model with validation data
    model.fit(X_train, y_train, epochs=2000, batch_size=32, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_pred_ori = model.predict(X_test)

    output = make_output_dict("CNN", f"{num_filters} filters, kernel size {kernel_size}", classification_report(y_test, y_pred, output_dict=True), prior(y_test))

    # Generate classification report
    return y_pred, output, y_pred_ori, output['Recall (1)'], model

# Attention CNN
def create_acnn_model(input_shape, num_classes, filters, kernel_size):
    inputs = tf.keras.Input(shape=input_shape)

    # CNN layers
    conv1 = Conv1D(filters, kernel_size=kernel_size, activation='relu')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    # Calculate the new shape for dynamic reshaping
    conv_shape = conv1.shape.as_list()
    new_shape = (-1, conv_shape[1] * conv_shape[2] // 2)

    # Reshape for attention
    reshape = Reshape(new_shape)(pool1)

    # Attention mechanism
    attention = Attention()([reshape, reshape])

    # Flatten for fully connected layers
    flatten = Flatten()(attention)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dropout = Dropout(0.5)(dense1)
    outputs = Dense(num_classes, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def evaluate_attention_cnn(filters, kernel_size, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path):
    if pretrain:
        model = load_model(model_path)
    else:
        model = create_acnn_model(X_train.shape[1:], 2, filters, kernel_size)
        #model = create_acnn_model(X_train.shape, 2, filters, kernel_size)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[auc])
    # EarlyStopping callback
    early_stopping = EarlyStopping(monitor=f'val_loss', patience=50, restore_best_weights=True)

    # Train the model with validation data
    model.fit(X_train, y_train, epochs=2000, batch_size=32, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stopping])
    y_pred = model.predict(X_test)
    y_pred_ori = model.predict(X_test)

    output = make_output_dict("CNN with Attention", f"{filters} filters, kernel size {kernel_size}", classification_report(y_test, y_pred.argmax(axis=1), output_dict=True), prior(y_test))
    return y_pred, output, y_pred_ori, output['Recall (1)'], model

# Trying out a different implementation of ACNN
def create_acnn_model2(input_shape, num_classes, filters, kernel_size):
    inputs = Input(shape=input_shape)

    # CNN layers
    conv1 = Conv1D(filters, kernel_size=kernel_size, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling1D(pool_size=2, padding='same')(conv1)
    conv1 = Conv1D(filters*2, kernel_size=kernel_size, activation='relu', padding='same')(inputs)

    # Attention mechanism applied directly on the pooled output, preserving the temporal dimension
    attention_output = Attention()([pool1, pool1])

    # Flatten for fully connected layers
    flatten = Flatten()(attention_output)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dropout = Dropout(0.5)(dense1)
    outputs = Dense(num_classes, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def evaluate_attention_cnn2(filters, kernel_size, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path):
    #auc = tf.keras.metrics.AUC()
    if pretrain:
        model = load_model(model_path)
    else:
        model = create_acnn_model(X_train.shape[1:], 2, filters, kernel_size)
        #model = create_acnn_model(X_train.shape, 2, filters, kernel_size)
        #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[auc])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        
    # EarlyStopping callback
    #early_stopping = EarlyStopping(monitor=f'val_{auc.name}', patience=10, restore_best_weights=True)
    early_stopping = EarlyStopping(monitor=f'val_loss', patience=50, restore_best_weights=True)

    # Train the model with validation data
    model.fit(X_train, y_train, epochs=2000, batch_size=32, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stopping])
    y_pred = model.predict(X_test)
    
    output = make_output_dict("CNN with Attention", f"{filters} filters, kernel size {kernel_size}", classification_report(y_test, y_pred.argmax(axis=1), output_dict=True), prior(y_test))
    y_pred_ori = y_pred[:,1]
    y_pred = y_pred.argmax(axis=1)
    #y_pred_ori = y_pred
    return y_pred, output, y_pred_ori, output['Recall (1)'], model

def evaluate_dumb_model(y_test, model_type='non_spikes'):
    y_pred = np.ones(len(y_test), dtype=int) if model_type == 'spikes' else np.zeros(len(y_test), dtype=int)
    output_dict = make_output_dict("Dumb Model", model_type, classification_report(y_test, y_pred, output_dict=True), prior(y_test))
    
    return y_pred, output_dict

def evaluate_edcr(name, y_pred1, y_pred2, y_test):
    print(y_pred1.shape,type(y_pred1))
    print(y_pred2.shape,type(y_pred2))

    y_pred = np.squeeze(y_pred1) | np.squeeze(y_pred2)
    output = make_output_dict("EDCR", name, classification_report(y_test, y_pred, output_dict=True), prior(y_test))

    # Generate classification report
    return y_pred, output

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
            predictions_df['True'] = y_test  # Assuming y_test is correctly shaped or is a single class label
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

# Evaluate all models
def evaluate_all(X_train, y_train, X_val, y_val, X_test, y_test, output_file_path, pred_file_path, saved_model_path, pretrain):
    global auc_count
    output_dicts = []

    # Picking best Model
    
    # FNN
    rules = []
    results = []
    auc_count = 0

    # LSTM
    for layers in [256, 128, 64, 32]:
        try: 
            model_descriptor = f"LSTM_{layers}_layers"
            model_path = f'{saved_model_path}/{model_descriptor}.h5'
            y_pred, output_dict, y_pred_lstm, acc, model = evaluate_lstm(layers, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path)
            rules.append([y_pred, y_pred_lstm, f"LSTM_{layers}", output_dict['Accuracy'], output_dict[OPTIMAL_METRIC]]) 
            output_dicts.append(output_dict)
            save_predictions_to_file(f"LSTM_{layers}_layers", y_pred, y_test, pred_file_path)

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
                y_pred, output_dict, y_pred_cnna, acc, model  = evaluate_attention_cnn2(filter, kernel, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path) # Switch this later?
                #rules.append([y_pred_cnna, f"CNNA_{filter}_{kernel}"]) 
                rules.append([y_pred, y_pred_cnna, f"CNNA_{filter}_{kernel}", output_dict['Accuracy'], output_dict[OPTIMAL_METRIC]]) 
                output_dicts.append(output_dict)
                save_predictions_to_file(f"CNN_Attention_{filter}_filters_{kernel}_kernels", y_pred, y_test, pred_file_path)

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
            y_pred, output_dict, y_pred_rnn, acc, model  = evaluate_rnn(units, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path)
            #rules.append([y_pred_rnn, f"RNN_{units}"]) 
            rules.append([y_pred, y_pred_rnn, f"RNN_{units}", output_dict['Accuracy'], output_dict[OPTIMAL_METRIC]]) 
            output_dicts.append(output_dict)
            save_predictions_to_file(f"RNN_{units}_units", y_pred, y_test, pred_file_path)

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
                y_pred, output_dict, y_pred_cnn, acc, model  = evaluate_cnn(filter, kernel, X_train, y_train, X_val, y_val, X_test, y_test, pretrain, model_path)
                #rules.append([y_pred_cnn, f"CNN_{filter}_{kernel}"]) 
                rules.append([y_pred, y_pred_cnn, f"CNN_{filter}_{kernel}", output_dict['Accuracy'], output_dict[OPTIMAL_METRIC]]) 
                output_dicts.append(output_dict)
                save_predictions_to_file(f"CNN_{filter}_filters_{kernel}_kernels", y_pred, y_test, pred_file_path)

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
    df1 = pd.DataFrame([x[3] for x in rules])   # look at acc
    best_acc_index = df1[0].idxmax()    # highest acc model
    df2 = pd.DataFrame([x[4] for x in rules])   # look at f1
    df2 = df2.drop(index = best_acc_index)  # models with the highest acc from before are excluded from the f1 dataframe
    df2 = df2.sort_values(by = 0, ascending=False)
    sorted_f1 = list(df2.index)
    rules_index = sorted_f1[:5]     # top 5 best f1 models
    
    print(f"best acc index: {best_acc_index}")
    print(f"rules index: {rules_index}")
    for confident in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        y_pred_all = []

        # Use a model (with high accuracy) as baseline and use high [OPTIMAL_METRIC] models to improve it
        for ri in rules_index:
            name = "Confident " + str(confident) + "Rule " + rules[ri][2] + "for " + rules[best_acc_index][2]
            try:
                y_pred1 = (rules[ri][1] > confident).astype(int)
                if(len(y_pred_all)):
                    y_pred_all = np.squeeze(y_pred_all) | np.squeeze(y_pred1)
                else:
                    y_pred_all = y_pred1
                y_pred, output_dict = evaluate_edcr(name, y_pred1, rules[best_acc_index][0], y_test)
                output_dicts.append(output_dict)                                                                                                          
                save_predictions_to_file(name, y_pred, y_test, pred_file_path)
            except Exception as e:
                print(f"Failed to evaluate {name}: {e}")
        
        # Use a model (with high accuracy) as baseline with ensemble to improve it
        name = "Confident " + str(confident) + "Rule all" + "for " + rules[best_acc_index][2]
        try:
            y_pred, output_dict = evaluate_edcr(name, y_pred_all, rules[best_acc_index][0], y_test)
            output_dicts.append(output_dict)                                                                                                          
            save_predictions_to_file(name, y_pred, y_test, pred_file_path)
        except Exception as e:
            print(f"Failed to evaluate {name}: {e}")  

        # Use dumb model as baseline with ensemble to improve it
        for dumb in DUMB_MODELS:
            name = "Confident " + str(confident) + "Rule all" + "for " + dumb
            try:
                pred_value = 1 if dumb == 'dumb_spikes' else 0
                
                y_pred_dumb = pd.Series([pred_value] * len(y_test))
                
                y_pred, output_dict = evaluate_edcr(name, y_pred_all, y_pred_dumb, y_test)
                output_dicts.append(output_dict)                                                                                                          
                save_predictions_to_file(name, y_pred, y_test, pred_file_path)
            except Exception as e:
                print(f"Failed to evaluate {name}: {e}")
        
    # After identifying the best model, save it
    output_dicts = pd.DataFrame(output_dicts)
    output_dicts.to_csv(output_file_path)
    return output_dicts

# Function to parse model descriptor and return appropriate model architecture
def get_model_from_descriptor(descriptor, input_shape):
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

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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


    
