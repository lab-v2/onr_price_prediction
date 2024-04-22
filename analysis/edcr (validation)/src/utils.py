#!/usr/bin/env python
# coding=utf-8
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
def get_scores(y_true, y_pred):
    try:
        y_actual = y_true
        y_hat = y_pred
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
               TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
               FP += 1
            if y_actual[i]==y_hat[i]==0:
               TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
               FN += 1
        print(f"TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}")
        
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return [pre, rec, f1]
    except:
        pre = accuracy_score(y_true, y_pred)
        f1 =         f1_score(y_true, y_pred, average='macro')
        f1micro =         f1_score(y_true, y_pred, average='micro')
        return [pre, f1, f1micro]
