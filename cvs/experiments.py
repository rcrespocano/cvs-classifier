# -*- coding: utf-8 -*-
import argparse
import dataset
import plot
import classifiers
import numpy as np
from sklearn import metrics


def run_main_experiment(n=1, verbose=False):
    _verbose_dataset = verbose
    classifiers_ids = ['random-forest', 'support-vector-machine', 'gaussian-naive-bayes', 'ada-boost']
    _num_class = len(classifiers_ids)
    
    _train_accuracy = [[] for i in range(_num_class)]
    _test_accuracy = [[] for i in range(_num_class)]
    _roc = [[] for i in range(_num_class)]
    _recall = [[] for i in range(_num_class)]
    _f1_score = [[] for i in range(_num_class)]

    # Dataset
    ds = dataset.load(verbose=_verbose_dataset)

    for x in range(0, n):
        # Train and test datasets
        train_x, test_x, train_y, test_y = dataset.train_test_datasets(ds, train_size=0.85, verbose=_verbose_dataset)
        _verbose_dataset = False
        
        for i,c in enumerate(classifiers_ids):
            # Fit and predict
            _classifier = classifiers.Classifier.factory(c)
            _classifier.fit(train_x, train_y.values.flatten())
            predictions = _classifier.predict(test_x)

            # Accuracy and confusion matrix
            _train_accuracy[i].append(metrics.accuracy_score(train_y.values.flatten(), _classifier.predict(train_x)))
            _test_accuracy[i].append(metrics.accuracy_score(test_y.values.flatten(), predictions))

            # Recall (sensitivity)
            _recall[i].append(metrics.recall_score(test_y.values.flatten(), predictions))

            # f1_score
            _f1_score[i].append(metrics.recall_score(test_y.values.flatten(), predictions))
            
            # Receiver Operating Characteristic curve
            false_pr, true_pr, thresholds = metrics.roc_curve(test_y.values.flatten(), predictions)
            roc_auc = metrics.auc(false_pr, true_pr)
            _roc[i].append((false_pr, true_pr, thresholds, roc_auc))

    # Results
    for i,c in enumerate(classifiers_ids):
        print('#####################')
        print(classifiers.get_name(classifiers_ids[i]))
        print('--')
        print('> Train accuracy = %0.2f ± %0.2f)' % (np.mean(_train_accuracy[i]), np.std(_train_accuracy[i])))
        print('> Test accuracy = %0.2f ± %0.2f)' % (np.mean(_test_accuracy[i]), np.std(_test_accuracy[i])))
        print('> Recall (sensitivity) = %0.2f ± %0.2f)' % (np.mean(_recall[i]), np.std(_recall[i])))
        print('> f1 score = %0.2f ± %0.2f)' % (np.mean(_f1_score[i]), np.std(_f1_score[i])))

    # Plot ROC results
    plot.plot_roc(_roc, classifiers_ids)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=1, type=int, help='Number of executions.', required=False)
    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose.', required=False)
    args = parser.parse_args()

    run_main_experiment(n=args.n, verbose=args.verbose)
