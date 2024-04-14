#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd
from src.rule_correction import *
import os

# load data
def run_edcr():
    with open(f'data/test.npy', 'rb') as f:
        data = np.load(f, allow_pickle=True)

    results = []
    epsilon = [0.001 * i for i in range(1, 100, 1)]
    for ep in epsilon:
        #result = PosNegRuleLearn(all_charts, epsilon)
        result = ruleForNegativeCorrection(data, ep)
        results.append([ep] + result)
        print(f"ep:{ep}\n{result}")
    col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
    df = pd.DataFrame(results, columns = ['epsilon'] + col )
    df.to_csv(f"rule_for_Negativecorrection.csv")

    results = []
    for ep in epsilon:
        #result = PosNegRuleLearn(all_charts, epsilon)
        result = ruleForPNCorrection(data, ep)
        results.append([ep] + result)
        print(f"ep:{ep}\n{result}")
    df = pd.DataFrame(results, columns = ['epsilon'] + col )
    df.to_csv( f"rule_for_PNcorrection.csv")

    results = []
    for ep in epsilon:
        #result = PosNegRuleLearn(all_charts, epsilon)
        result = ruleForNPCorrection(data, ep)
        results.append([ep] + result)
        print(f"ep:{ep}\n{result}")
    col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
    #df = pd.DataFrame(results, columns = ['epsilon'] + col + ['acc', 'macro-F1', 'micro-F1'])
    df = pd.DataFrame(results, columns = ['epsilon'] + col )
    df.to_csv( f"rule_for_NPcorrection.csv")
    if os.path.exists('Results.xlsx'):
        os.remove('Results.xlsx')
    df.to_excel('Results.xlsx', sheet_name='EDCR Results', index=False)

    return df