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
        
        plt.plot([0,1], [0,1], 'k--', lw=1)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        # Mean ROC and AUC
        try:
            _mean_roc = np.mean(x, axis=0)
            _auc = [item[3] for item in x]
            _label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (np.mean(_auc), np.std(_auc))
            plt.plot(_mean_roc[0], _mean_roc[1], alpha=0.99, color='blue', label=_label)
        except Exception as exc:
            print(exc)

        plt.legend(loc='lower right')
        plt.show()
