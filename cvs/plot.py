# -*- coding: utf-8 -*-
import classifiers
import numpy as np
import matplotlib.pyplot as plt


def plot_roc(roc_data, classifiers_ids):
    for i, x in enumerate(roc_data):
        _name = classifiers.get_name(classifiers_ids[i])
        plt.title('Receiver Operating Characteristic - ' + _name)
        
        for executions in x:
            plt.plot(executions[0], executions[1], alpha=0.10, color='gray')
        
        plt.legend(loc='lower right')
        plt.plot([0,1], [0,1], 'k--', lw=1)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        # Mean
        try:
            _mean = np.mean(x, axis=0)
            plt.plot(_mean[0], _mean[1], alpha=0.99, color='blue')
        except Exception as exc:
            print(exc)

        plt.show()
