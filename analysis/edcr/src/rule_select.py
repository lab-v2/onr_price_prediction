#!/usr/bin/env python
# coding=utf-8
import numpy as np
import itertools
'''
input data format: list
In the list:
    each row should contain:
        pred, corr, true_positive, false_positive, rule_result1, rule_result2 ... rule_resultn
    each value should be 0/1
'''

# CorrRuleLearn
def DetUSMPosRuleSelect(chart):
    chart = np.array(chart)
    rule_indexs = [i for i in range(4, len(chart[0]))]
    each_sum = np.sum(chart, axis = 0)
    tpi = each_sum[2]
    fpi = each_sum[3]
    pi = tpi * 1.0 /(tpi + fpi)

    pb_scores = []
    for ri in rule_indexs:
        posi = np.sum(chart[:,1] * chart[:,ri], axis = 0)
        bodyi = np.sum(chart[:,ri], axis = 0)
        score = posi * 1.0 / bodyi
        if score > pi:
            pb_scores.append((score, ri))
    pb_scores = sorted(pb_scores)
    cci = []
    ccn = pb_scores
    for (score, ri) in pb_scores:

        cii = 0
        ciij = 0
        for (cs, ci) in cci:
            cii = cii | chart[:,ci]
        POScci = np.sum(cii * chart[:, 1], axis = 0)
        BODcci = np.sum(cii, axis = 0)
        POSccij = np.sum((cii | chart[:,ri]) * chart[:, 1], axis = 0)
        BODccij = np.sum((cii | chart[:,ri]), axis = 0)

        cni = 0
        cnij = 0
        for (cs, ci) in ccn:
            cni = (cni | chart[:,ci])
            if ci == ri:
                continue
            cnij = (cnij | chart[:, ci])
        POScni = np.sum(cni * chart[:, 1], axis = 0)
        BODcni = np.sum(cni, axis = 0)
        POScnij = np.sum(cnij * chart[:, 1], axis = 0)
        BODcnij = np.sum(cnij, axis = 0)

        a = POSccij * 1.0 / (BODccij + 0.001) - POScci * 1.0 / (BODcci + 0.001)
        b = POScnij * 1.0 / (BODcnij + 0.001) - POScni * 1.0 / (BODcni + 0.001)
        if a >= b:
            cci.append((score, ri))
        else:
            ccn.remove((score, ri))

    cii = 0
    for (cs, ci) in cci:
        cii = cii | chart[:,ci]
    POScci = np.sum(cii * chart[:, 1], axis = 0)
    BODcci = np.sum(cii, axis = 0)
    new_pre = POScci * 1.0 / (BODcci + 0.001)
    if new_pre < pi:
        cci = []
    cci = [c[1] for c in cci]
    # print(f"cci:{cci}, new_pre:{new_pre}, pre:{pi}")
    return cci

# DetRuleLearn
def GreedyNegRuleSelect(epsilon, chart):
    chart = np.array(chart)
    rule_indexs = [i for i in range(4, len(chart[0]))]
    len_rules = len(rule_indexs)
    each_sum = np.sum(chart, axis = 0)
    tpi = each_sum[2]   # Sum of TP by the base model   leakage happening due to no test/train split?
    fpi = each_sum[3]   # Sum of FP by the base model   leakage happening due to no test/train split?
    pi = tpi * 1.0 /(tpi + fpi) # Precision of predictions    (How much of these positives are legit)  leakage happening due to no test/train split?
    ri = tpi * 1.0 / each_sum[1]    # Ratio of ACTUAL TP    leakage happening due to no test/train split?
    ni = each_sum[0]    # Sum of P by the base model
    quantity = epsilon * ni * pi / ri   # A decision threshold? 
    # print (f'QUANTITY: {quantity}')
    # print(f"quantity:{quantity}")

    best_combins = []
    NCi = []
    NCn = []
    for rule in rule_indexs:
        neg = np.sum(chart[:,2] * chart[:,rule]) # A metric for seeing how each rule performs in terms of TP
        # print(f'neg SCORE: {neg}')
        if neg < quantity:
            # print('YUPPPP')
            NCn.append(rule)    # Flagged rule

    while(NCn):             # Go through flagged rules
        best_score = -1 
        best_index = -1
        for c in NCn:
            tem_cond = 0
            for cc in NCi:
                tem_cond |= chart[:,cc]
            tem_cond |= chart[:,c]
            pos = np.sum(chart[:,3] * tem_cond) 
            if best_score < pos:
                best_score = pos
                best_index = c
        NCi.append(best_index)
        NCn.remove(best_index)
        tem_cond = 0
        for cc in NCi:
            tem_cond |= chart[:,cc]
        tmp_NCn = []
        for c in NCn:
            tem = tem_cond | chart[:,c]
            neg = np.sum(chart[:,2] * tem)
            if neg < quantity:
                tmp_NCn.append(c)
        NCn = tmp_NCn
    # print(f"NCi:{NCi}")
    return NCi

    for r in range(1,len_rules + 1):
        combinations = list(itertools.combinations(rule_indexs, r))
        max_score = [0, 0, 0]
        max_combi = tuple()
        for cond in combinations:
            tmp_cond = 0
            for c in cond:
                tmp_cond |= chart[:,c]
            negi = chart[:,2] * tmp_cond 
            neg = np.sum(negi)
            if neg < quantity:
                posi = chart[:,3] * tmp_cond
                pos = np.sum(posi)
                if pos - neg > max_score[0]:
                    max_score[0] = pos - neg
                    max_score[1] = neg
                    max_score[2] = pos
                    max_combi = cond
        # print(f"r:{r}, max_score:{max_score[0]}, negi:{max_score[1]}, posi:{max_score[2]}, max_combi:{max_combi}")
        if max_combi:
            best_combins.append(max_combi)
    return best_combins

