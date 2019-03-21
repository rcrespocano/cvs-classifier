# -*- coding: utf-8 -*-
import classifiers
import matplotlib.pyplot as plt


def plot_roc(roc_data, classifiers_ids):
    for i, x in enumerate(roc_data):
        _name = classifiers.get_name(classifiers_ids[i])
        plt.title('Receiver Operating Characteristic - ' + _name)
        
        for executions in x:
            plt.plot(executions[0], executions[1], alpha=0.3, label='AUC = %0.2f'% executions[3])
        
        plt.legend(loc='lower right')
        plt.plot([0,1], [0,1], 'k--', lw=1)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
