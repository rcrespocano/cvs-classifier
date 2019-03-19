# -*- coding: utf-8 -*-
from dataset import ds_load, ds_train_test_datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def main():
    # Dataset
    ds = ds_load()
    train_x, test_x, train_y, test_y = ds_train_test_datasets(ds, train_size=0.75)

    # Random Forest classifier
    print('Support Vector Machine: fit')
    classifier = SVC(gamma='auto')
    classifier.fit(train_x.values, train_y.values.flatten())

    # Predict
    print('Support Vector Machine: predict')
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
