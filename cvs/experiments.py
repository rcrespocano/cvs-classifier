# -*- coding: utf-8 -*-
import argparse
import dataset
import plot
from classifiers import Classifier
from sklearn import metrics


def run_main_experiment(n=1, verbose=False):
    _verbose_dataset = verbose
    classifiers_ids = ['random-forest', 'support-vector-machine', 'gaussian-naive-bayes', 'ada-boost']

    _train_accuracy = [[] for i in range(len(classifiers_ids))]
    _test_accuracy = [[] for i in range(len(classifiers_ids))]
    _conf_matrix = [[] for i in range(len(classifiers_ids))]
    _roc = [[] for i in range(len(classifiers_ids))]

    # Dataset
    ds = dataset.load(verbose=_verbose_dataset)

    for x in range(0, n):
        # Train and test datasets
        train_x, test_x, train_y, test_y = dataset.train_test_datasets(ds, train_size=0.85, verbose=_verbose_dataset)
        _verbose_dataset = False
        
        for i,c in enumerate(classifiers_ids):
            # Fit and predict
            _classifier = Classifier.factory(c)
            _classifier.fit(train_x, train_y.values.flatten())
            predictions = _classifier.predict(test_x)

            # Accuracy and confusion matrix
            _train_accuracy[i].append(metrics.accuracy_score(train_y.values.flatten(), _classifier.predict(train_x)))
            _test_accuracy[i].append(metrics.accuracy_score(test_y.values.flatten(), predictions))
            _conf_matrix[i].append(metrics.confusion_matrix(test_y, predictions))

            # Receiver Operating Characteristic curve
            false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(test_y.values.flatten(), predictions)
            roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
            _roc[i].append((false_positive_rate, true_positive_rate, thresholds, roc_auc))

    if verbose:
        print(classifiers_ids)
        print('Train accuracy:\n', _train_accuracy)
        print('Test accuracy:\n', _test_accuracy)
        print('Confusion matrix:\n', _conf_matrix)

    # Plot ROC results
    plot.plot_roc(_roc, classifiers_ids)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=1, type=int, help='Number of executions.', required=False)
    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose.', required=False)
    args = parser.parse_args()

    run_main_experiment(n=args.n, verbose=args.verbose)
