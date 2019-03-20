# -*- coding: utf-8 -*-
import argparse
import os
import sys
from dataset import ds_load, ds_train_test_datasets
sys.stderr = open(os.devnull, "w")  # silence stderr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
sys.stderr = sys.__stderr__  # unsilence stderr


def main(verbose=False):
    # Dataset
    ds = ds_load(verbose=verbose)
    train_x, test_x, train_y, test_y = ds_train_test_datasets(ds, train_size=0.85, verbose=verbose)

    # Random Forest classifier
    print('Random Forest: fit')
    classifier = RandomForestClassifier(n_estimators=100, bootstrap=False)
    classifier.fit(train_x.values, train_y.values.flatten())

    # Predict
    print('Random Forest: predict')
    predictions = classifier.predict(test_x)

    # Accuracy
    train_accuracy = accuracy_score(train_y.values.flatten(), classifier.predict(train_x))
    test_accuracy = accuracy_score(test_y.values.flatten(), predictions)
    print('Train accuracy:', train_accuracy)
    print('Test accuracy:', test_accuracy)

    # Confusion matrix
    conf_matrix = confusion_matrix(test_y, predictions)
    print('Confusion matrix:\n', conf_matrix)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose.', required=False)
    args = parser.parse_args()

    main(verbose=args.verbose)
