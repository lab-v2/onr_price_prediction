import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

base_path = sys.argv[1]
result1 = pd.read_csv(base_path + 'rule_for_NPcorrection.csv')
result2 = pd.read_csv(base_path + 'rule_for_PNcorrection.csv')
result3 = pd.read_csv(base_path + 'rule_for_Negativecorrection.csv')
classes = ['Walk', 'Bike', 'Bus', 'Drive', 'Train']
def generate_curve(result, string):
    epsilon = result.iloc[1:,1].tolist()
    for i in range(5):
        pre =  result.iloc[1:,2 + i * 7].tolist()
        recal =result.iloc[1:,3 + i * 7].tolist()
        f1 =   result.iloc[1:,4 + i * 7].tolist()
        plt.figure()
        plt.plot(epsilon, pre, color = 'r', ls = '-', label = 'Precision')
        plt.plot(epsilon, [pre[0] for i in range(len(epsilon))], color = 'r', ls = '-', linewidth = 0.5, label = 'Pre_base')
        plt.plot(epsilon, recal, color = 'b', ls = '--',label = 'Recall')
        plt.plot(epsilon, [recal[0] for i in range(len(epsilon))], color = 'b', ls = '-', linewidth = 0.5, label = 'Rec_base')
        plt.plot(epsilon, f1, color = 'k', ls = '-.', label = 'F1')
        plt.plot(epsilon, [f1[0] for i in range(len(epsilon))], color = 'k', ls = '-', linewidth = 0.5, label = 'F1_base')
        plt.plot(epsilon, [recal[0] -i for i in epsilon], color = 'g', ls = ':', linewidth = 1.2, label = 'TR in Recall')
        plt.legend()
        plt.title(f"{classes[i]}")
        plt.savefig(base_path + f"{string}/" + f"{classes[i]}.png")
        plt.close()
    i = 5
    pre =  result.iloc[1:,2 + i * 7].tolist()
    recal =result.iloc[1:,3 + i * 7].tolist()
    f1 =   result.iloc[1:,4 + i * 7].tolist()
    plt.figure()
    plt.plot(epsilon, pre, color = 'r', ls = '-', label = 'Acc')
    plt.plot(epsilon, [pre[0] for i in range(len(epsilon))], color = 'r', ls = ':', linewidth = 0.5, label = 'Acc_base')
    plt.plot(epsilon, recal, color = 'b', ls = '--',label = 'Macro-F1')
    plt.plot(epsilon, [recal[0] for i in range(len(epsilon))], color = 'b', ls = ':', linewidth = 0.5, label = 'Macro-F1_base')
    plt.plot(epsilon, f1, color = 'k', ls = '-.', label = 'Micro-F1')
    plt.plot(epsilon, [f1[0] for i in range(len(epsilon))], color = 'k', ls = ':', linewidth = 0.5, label = 'Micro-F1_base')
    plt.legend()
    plt.title(f"{string}_all_class")
    plt.savefig(base_path + f"{string}_all_class.png")
    plt.close()
generate_curve(result1, 'NP')
generate_curve(result2, 'PN')
generate_curve(result3, 'Negative')

def generate_prf1_curve(all_charts):
    for count, chart in enumerate(all_charts):
        scores = []
        chart = np.array(chart)
        plt.figure()
        scores = np.array(scores)
        print(scores)
        plt.plot(scores[:,0],scores[:,1], color = 'r', label = "Precision")
        plt.plot(scores[:,0],scores[:,2], color = 'b', label = "Recall")
        plt.plot(scores[:,0],scores[:,3], color = 'k', label = "F1")
        plt.legend()
        plt.title(f"{count}_class")
        plt.savefig(f"{count}_class.png")
        plt.close()
