# -*- coding: utf-8 -*-
import argparse
import dataset
import plot
import classifiers
import variables
import numpy as np
from sklearn import metrics
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint


def run_default_experiment(dataset_path, n=1, experiment_id=variables.EXP_N01, verbose=False):
    print('RUN DEFAULT EXPERIMENT.')

    _verbose_dataset = verbose
    classifiers_ids = ['random-forest', 'support-vector-machine', 'gaussian-naive-bayes', 'ada-boost']
    _num_class = len(classifiers_ids)

    _train_accuracy = [[] for i in range(_num_class)]
    _test_accuracy = [[] for i in range(_num_class)]
    _roc = [[] for i in range(_num_class)]
    _auc = [[] for i in range(_num_class)]
    _recall = [[] for i in range(_num_class)]
    _f1_score = [[] for i in range(_num_class)]

    # Dataset
    ds = dataset.load(dataset_path, experiment_id, verbose=_verbose_dataset)

    # Repetition of the stochastic experiment N times
    for x in range(0, n):
        # Train and test datasets
        train_x, test_x, train_y, test_y = dataset.train_test_datasets(ds, train_size=0.80, verbose=_verbose_dataset)
        _verbose_dataset = False
        
        for i,c in enumerate(classifiers_ids):
            # Fit and predict
            _classifier = classifiers.Classifier.factory(c)
            _classifier.fit(train_x, train_y)
            predictions = _classifier.predict(test_x)

            # Accuracy and confusion matrix
            _train_accuracy[i].append(metrics.accuracy_score(train_y, _classifier.predict(train_x)))
            _test_accuracy[i].append(metrics.accuracy_score(test_y, predictions))

            # Recall (sensitivity)
            _recall[i].append(metrics.recall_score(test_y, predictions))

            # f1_score
            _f1_score[i].append(metrics.recall_score(test_y, predictions))
            
            # Receiver Operating Characteristic curve
            false_pr, true_pr, thresholds = metrics.roc_curve(test_y, predictions)
            roc_auc = metrics.auc(false_pr, true_pr)
            _roc[i].append((false_pr, true_pr, thresholds, roc_auc))
            _auc[i].append(roc_auc)

    # Results
    for i,c in enumerate(classifiers_ids):
        print('#####################')
        print(classifiers.get_name(classifiers_ids[i]))
        print('--')
        print('> Train accuracy = %0.2f ± %0.2f [max: %0.2f])' % (np.mean(_train_accuracy[i]), np.std(_train_accuracy[i]), np.max(_train_accuracy[i])))
        print('> Test accuracy = %0.2f ± %0.2f [max: %0.2f])' % (np.mean(_test_accuracy[i]), np.std(_test_accuracy[i]), np.max(_test_accuracy[i])))
        print('> Recall (sensitivity) = %0.2f ± %0.2f)' % (np.mean(_recall[i]), np.std(_recall[i])))
        print('> f1 score = %0.2f ± %0.2f)' % (np.mean(_f1_score[i]), np.std(_f1_score[i])))
        print('> AUC = ' + str(np.mean(_auc[i])))

    # Plot ROC results
    print('- Plot ROC results.')
    plot.plot_roc(_roc, classifiers_ids)


def run_experiment_rf(dataset_path, n=1, experiment_id=variables.EXP_N01, verbose=False):
    print('RUN EXPERIMENT {RANDOM FOREST}.')

    # Dataset
    _verbose_dataset = verbose
    ds = dataset.load(dataset_path, experiment_id, verbose=_verbose_dataset)

    # Repetition of the stochastic experiment N times
    for x in range(0, n):
        # Train and test datasets
        train_x, test_x, train_y, test_y = dataset.train_test_datasets(ds, train_size=0.80, verbose=_verbose_dataset)
        _verbose_dataset = False

        # Build a classifier
        _classifier = classifiers.Classifier.factory('random-forest')

        # Use a full grid over all parameters
        param_grid = {'max_depth': [3, None],
                      'max_features': [1, 3, 5, 10, 12],
                      'min_samples_split': [2, 3, 7, 10, 12],
                      'bootstrap': [True, False],
                      'criterion': ['gini', 'entropy']}

        # Run grid search
        grid_search = GridSearchCV(_classifier.get_base_classifier(), param_grid=param_grid, cv=5, iid=False)
        start = time()
        grid_search.fit(train_x, train_y)

        if verbose:
            print('GridSearchCV took %.2f seconds for %d candidate parameter settings.' % (time() - start, len(grid_search.cv_results_['params'])))
            _report(grid_search.cv_results_, n_top=3)

        # Predict with best model
        grid_predictions = grid_search.predict(test_x)
        print(metrics.classification_report(test_y, grid_predictions))


def run_experiment_svm(dataset_path, n=1, experiment_id=variables.EXP_N01, verbose=False):
    print('RUN EXPERIMENT {SUPPORT VECTOR MACHINE}.')

    # Dataset
    _verbose_dataset = verbose
    ds = dataset.load(dataset_path, experiment_id, verbose=_verbose_dataset)

    # Repetition of the stochastic experiment N times
    for x in range(0, n):
        # Train and test datasets
        train_x, test_x, train_y, test_y = dataset.train_test_datasets(ds, train_size=0.80, verbose=_verbose_dataset)
        _verbose_dataset = False

        # Build a classifier
        _classifier = classifiers.Classifier.factory('support-vector-machine')

        # Specify parameters and distributions to sample from
        param_dist = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                      {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                      {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]

        # Run grid search
        grid_search = GridSearchCV(_classifier.get_base_classifier(), param_grid=param_dist, cv=5, iid=False)
        start = time()
        grid_search.fit(train_x, train_y)

        if verbose:
            print('GridSearchCV took %.2f seconds for %d candidate parameter settings.' % (time() - start, len(grid_search.cv_results_['params'])))
            _report(grid_search.cv_results_, n_top=3)

        # Predict with best model
        grid_predictions = grid_search.predict(test_x)
        print(metrics.classification_report(test_y, grid_predictions))


# Utility function to report best scores
def _report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print('Parameters: {0}'.format(results['params'][candidate]))
            print('')


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset (.sav file).', required=True)
    parser.add_argument('--n', default=1, type=int, help='Number of executions.', required=False)
    parser.add_argument('--experiment', type=str, help='Experiment ID [N01, N02, ..., N10].', required=True)
    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose.', required=False)
    parser.add_argument('--classifier', type=str, help='Classifier [rf, svm, all].', required=False)
    args = parser.parse_args()

    print('- Experiment ID: ' + args.experiment)
    exp_id = variables.get_experiment_variables_id(args.experiment)

    # Run experiment
    try:
        if args.classifier == 'rf':
            run_experiment_rf(dataset_path=args.dataset, n=args.n, experiment_id=exp_id, verbose=args.verbose)
        elif args.classifier == 'svm':
            run_experiment_svm(dataset_path=args.dataset, n=args.n, experiment_id=exp_id, verbose=args.verbose)
        else:
            run_default_experiment(dataset_path=args.dataset, n=args.n, experiment_id=exp_id, verbose=args.verbose)
    except Exception as exc:
        run_default_experiment(dataset_path=args.dataset, n=args.n, experiment_id=exp_id, verbose=args.verbose)
