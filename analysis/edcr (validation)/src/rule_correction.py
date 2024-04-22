#!/usr/bin/env python
# coding=utf-8
import numpy as np
from src.rule_select import *
from src.utils import *
from sklearn.model_selection import train_test_split

def ruleForNegativeCorrection(chart, epsilon):
    results = []
    total_results = np.copy(chart[:,0])

    chart = np.array(chart)
    NCi = GreedyNegRuleSelect(epsilon, chart)
    negi_count = 0
    posi_count = 0

    predict_result = np.copy(chart[:,0])
    tem_cond = 0
    for cc in NCi:
        tem_cond |= chart[:,cc]
    if np.sum(tem_cond) > 0:
        for ct,cv in enumerate(chart):
            if tem_cond[ct] and predict_result[ct]:
                negi_count += 1
                predict_result[ct] = 0

    CCi = []
    scores_cor = get_scores(chart[:,1], predict_result)
    results.extend(scores_cor + [ negi_count, posi_count, len(NCi), len(CCi) ])
    #results.extend(get_scores(chart[:,1], total_results))
    return results

def ruleForPNCorrection(chart, epsilon):
    results = []
    total_results = np.copy(chart[:,0])
    chart = np.array(chart)
    negi_count = 0
    posi_count = 0

    predict_result = np.copy(chart[:,0])
    CCi = []
    CCi = DetUSMPosRuleSelect(chart)
    tem_cond = 0
    for cc in CCi:
        tem_cond |= chart[:,cc]
    if np.sum(tem_cond) > 0:
        for ct,cv in enumerate(chart):
            if tem_cond[ct]:
                if not predict_result[ct]:
                    posi_count += 1
                    predict_result[ct] = 1
                    total_results[ct] = 0

    NCi = GreedyNegRuleSelect(epsilon, chart)

    tem_cond = 0
    for cc in NCi:
        tem_cond |= chart[:,cc]
    if np.sum(tem_cond) > 0:
        for ct,cv in enumerate(chart):
            if tem_cond[ct] and predict_result[ct]:
                negi_count += 1
                predict_result[ct] = 0


    scores_cor = get_scores(chart[:,1], predict_result)
    results.extend(scores_cor + [ negi_count, posi_count, len(NCi), len(CCi) ])
    #results.extend(get_scores(chart[:,1], total_results))
    return results

def ruleForNPCorrection(chart, epsilon):
    results = []
    total_results = np.copy(chart[:,0])

    chart = np.array(chart)
    train, test = train_test_split(chart, test_size=0.2, shuffle=False)

    NCi = GreedyNegRuleSelect(epsilon, train)
    negi_count = 0
    posi_count = 0

    predict_result = np.copy(chart[:,0])
    tem_cond = 0
    for cc in NCi:
        tem_cond |= chart[:,cc]
    if np.sum(tem_cond) > 0:
        for ct,cv in enumerate(chart):
            if tem_cond[ct] and predict_result[ct]:
                negi_count += 1
                predict_result[ct] = 0

    CCi = []
    CCi = DetUSMPosRuleSelect(train)
    tem_cond = 0
    rec_true = []
    rec_pred = []
    for cc in CCi:
        tem_cond |= chart[:,cc]
    if np.sum(tem_cond) > 0:
        for ct,cv in enumerate(chart):
            if tem_cond[ct]:
                if not predict_result[ct]:
                    posi_count += 1
                    predict_result[ct] = 1
                    total_results[ct] = 0
            else:
                rec_true.append(cv[1])
                rec_pred.append(cv[0])

    scores_cor = get_scores(chart[:,1], predict_result)
    results.extend(scores_cor + [ negi_count, posi_count, len(NCi), len(CCi) ])
    #results.extend(get_scores(chart[:,1], total_results))
    return results
