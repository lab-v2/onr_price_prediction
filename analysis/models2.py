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

DUMB_MODELS = ['dumb_spikes', 'dumb_non_spikes']
learning_rate = 0.01
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


#LSTM Model
def evaluate_lstm(num_layers: int, X_train, y_train, X_test, y_test, pretrain, model_path):
    adam_optimizer = Adam(learning_rate=learning_rate)

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
        model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=[auc])

    # Train the model with validation data
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_pred_ori = model.predict(X_test)

    output = make_output_dict("LSTM", f"{num_layers} layers", classification_report(y_test, y_pred, output_dict=True), prior(y_test))
    return y_pred, output, y_pred_ori, model

# Build RNN
def evaluate_rnn(num_units: int, X_train, y_train, X_test, y_test, pretrain, model_path):
    adam_optimizer = Adam(learning_rate=learning_rate)

    # Build the RNN model
    if pretrain:
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(SimpleRNN(num_units, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
        #model.add(SimpleRNN(num_units, activation='relu'))
        model.add(Dense(int(num_units/2), activation='relu'))  # Ensure num_units/2 is cast to int for layer units
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=[auc])

    # Train the model with validation data
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_pred_ori = model.predict(X_test)

    output = make_output_dict("RNN", f"{num_units} units", classification_report(y_test, y_pred, output_dict=True), prior(y_test))
    return y_pred, output, y_pred_ori, model

# Build the CNN model
def evaluate_cnn(num_filters: int, kernel_size: int, X_train, y_train, X_test, y_test, pretrain, model_path):
    adam_optimizer = Adam(learning_rate=learning_rate)

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
        model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=[auc])

    # Train the model with validation data
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_pred_ori = model.predict(X_test)

    output = make_output_dict("CNN", f"{num_filters} filters, kernel size {kernel_size}", classification_report(y_test, y_pred, output_dict=True), prior(y_test))

    # Generate classification report
    return y_pred, output, y_pred_ori, model

# Attention CNN
def create_acnn_model(input_shape, num_classes, filters, kernel_size):
    inputs = tf.keras.Input(shape=input_shape)
    # CNN layers
    conv1 = Conv1D(filters, kernel_size=kernel_size, activation='relu')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv_shape = conv1.shape.as_list()
    new_shape = (-1, conv_shape[1] * conv_shape[2] // 2)
    reshape = Reshape(new_shape)(pool1)
    attention = Attention()([reshape, reshape])
    flatten = Flatten()(attention)
    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dropout = Dropout(0.5)(dense1)
    outputs = Dense(num_classes, activation='softmax')(dropout)
    model = Model(inputs=inputs, outputs=outputs)

    return model

def evaluate_attention_cnn2(filters, kernel_size, X_train, y_train, X_test, y_test, pretrain, model_path):
    #auc = tf.keras.metrics.AUC()
    adam_optimizer = Adam(learning_rate=learning_rate)

    if pretrain:
        model = load_model(model_path)
    else:
        model = create_acnn_model(X_train.shape[1:], 2, filters, kernel_size)
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    # Train the model with validation data
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    y_pred = model.predict(X_test)
    output = make_output_dict("CNN with Attention", f"{filters} filters, kernel size {kernel_size}", classification_report(y_test, y_pred.argmax(axis=1), output_dict=True), prior(y_test))
    y_pred_ori = y_pred[:,1]
    y_pred = y_pred.argmax(axis=1)
    #y_pred_ori = y_pred
    return y_pred, output, y_pred_ori, model

def evaluate_dumb_model(y_test, model_type='non_spikes'):
    y_pred = np.ones(len(y_test), dtype=int) if model_type == 'spikes' else np.zeros(len(y_test), dtype=int)
    output_dict = make_output_dict("Dumb Model", model_type, classification_report(y_test, y_pred, output_dict=True), prior(y_test))
    
    return y_pred, output_dict


    
