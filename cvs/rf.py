# -*- coding: utf-8 -*-
import os
import sys
from dataset import ds_load, ds_train_test_datasets
sys.stderr = open(os.devnull, "w")  # silence stderr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
sys.stderr = sys.__stderr__  # unsilence stderr


def main():
    # Dataset
    ds = ds_load()
    train_x, test_x, train_y, test_y = ds_train_test_datasets(ds, train_size=0.75)

    # Random Forest classifier
    print('Random Forest: fit')
    classifier = RandomForestClassifier(n_estimators=10000, criterion='entropy', bootstrap=True)
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
    main()
