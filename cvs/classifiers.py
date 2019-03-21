# -*- coding: utf-8 -*-
import os
import sys
sys.stderr = open(os.devnull, "w")
from sklearn.ensemble import RandomForestClassifier
sys.stderr = sys.__stderr__
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


class Classifier(object):
    def __init__(self):
        self.name = None
        self.classifier = None

    def factory(type):
        if type == 'random-forest': return RandomForest()
        if type == 'support-vector-machine': return SupportVectorMachine()
        if type == 'gaussian-naive-bayes': return GaussianNaiveBayes()
        if type == 'ada-boost': return AdaBoost()
        assert 0, 'Wrong algorithm: ' + type
    factory = staticmethod(factory)

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)


class RandomForest(Classifier):
    def __init__(self):
        super(RandomForest, self).__init__()
        self.name = 'random-forest'
        self.classifier = RandomForestClassifier(n_estimators=1000)


class SupportVectorMachine(Classifier):
    def __init__(self):
        super(SupportVectorMachine, self).__init__()
        self.name = 'support-vector-machine'
        self.classifier = SVC(kernel='rbf', C=1.0, gamma='auto')


class GaussianNaiveBayes(Classifier):
    def __init__(self):
        super(GaussianNaiveBayes, self).__init__()
        self.name = 'gaussian-naive-bayes'
        self.classifier = GaussianNB()


class AdaBoost(Classifier):
    def __init__(self):
        super(AdaBoost, self).__init__()
        self.name = 'ada-boost'
        self.classifier = AdaBoostClassifier()
