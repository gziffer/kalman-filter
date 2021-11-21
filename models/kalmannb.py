import numpy as np
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions


class KalmanNB(BaseSKMObject, ClassifierMixin):
    """ Incremental Kalman Naive Bayes classifier for categorical features.

    Performs classic bayesian prediction while making naive assumption that all inputs are independent.
    The trained Naive Bayes classifier predicts for every unlabelled instance the class to which it
    belongs with higher probability.
    This implementation is able to incrementally update its model.

    It stores an estimator for each probability it needs in order to make the prediction.

    In particular it uses the following data structures:

    counter_attribute_per_class: which represents the frequency of items belonging to specific class. For each class
    it has an estimator of the probability for that class.
    Example:
    {
        class 1: estimator
        class 2: estimator
        class 3: estimator
    }

    no_samples_per_class: which represents the frequency of specific value for different features, divided per class.
    Example:
    {
        class 1:
                feat 1:
                        value_1: estimator
                        value_2: estimator
                feat 2:
                        value_1: estimator
                        value_2: estimator
                        value_3: estimator

        class 2:
                feat 1:
                        value_1: estimator
                        value_2: estimator
                feat 2:
                        value_1: estimator
                        value_2: estimator
                        value_3: estimator
    }

    Each estimator is updated using the Kalman filter.
    It can use specific q and r parameter (default q=1 and r=1000) or it can use dynamic q and r in combination
    with ADWIN (by passing the parameter width in the partial_fit function.

    It uses as initial setting for each estimator: p0 = 0.0, x0 = 0.0

    Notes
    -----
    This estimator is not optimal for a mixture of categorical and numerical features.
    This implementation treats all features from a given stream as categorical.

    """

    def __init__(self, q=1.0, r=1000.0):
        super().__init__()

        self.counter_attribute_per_class = {}

        self.no_samples_per_class = {}

        # to monitor if there are new classes
        self.total_classes = None

        # to avoid 0 probabilities
        self._epsilon = 1e-9

        self.q = q

        self.r = r

        self.p0 = 0.0

        self.x0 = 0.0

    def partial_fit(self, X, y, classes=None, weights=None, width=None):

        if self.total_classes is None and classes is not None:
            self.total_classes = classes

        if y is not None:
            row_cnt, _ = get_dimensions(X)

            for i in range(row_cnt):
                self._partial_fit(X[i], y[i], width=width)

    def _partial_fit(self, X, y, width):

        # Update the estimator of the counter for the class y, adding 1.0
        try:
            p0, x0 = self.no_samples_per_class[y]
            p1, x1 = self._kalman_update(p0, x0, width, 1.0)
            self.no_samples_per_class[y] = (p1, x1)
        except KeyError:
            self.no_samples_per_class[y] = (self.p0, self.x0)
            self.counter_attribute_per_class[y] = {}

        # Update the estimator of the counter for the other classes, adding 0.0
        for key in self.no_samples_per_class.keys():
            if key != y:
                p0, x0 = self.no_samples_per_class[key]
                p1, x1 = self._kalman_update(p0, x0, width, 0.0)
                self.no_samples_per_class[key] = (p1, x1)

        for i in range(len(X)):
            try:
                tmp = self.counter_attribute_per_class[y][i]
            except KeyError:
                self.counter_attribute_per_class[y][i] = {}

            # Update the estimator of the counter for the attribute at index i of class y
            # Add 1.0 when value = X[i], while 0.0 for all the other values
            try:
                p0, x0 = self.counter_attribute_per_class[y][i][X[i]]
                p1, x1 = self._kalman_update(p0, x0, width, 1.0)
                self.counter_attribute_per_class[y][i][X[i]] = (p1, x1)
            except KeyError:
                self.counter_attribute_per_class[y][i][X[i]] = (self.p0, self.x0)

            for key in self.counter_attribute_per_class[y][i]:
                if key != X[i]:
                    p0, x0 = self.counter_attribute_per_class[y][i][key]
                    p1, x1 = self._kalman_update(p0, x0, width, 0.0)
                    self.counter_attribute_per_class[y][i][key] = (p1, x1)

    def _kalman_update(self, p0, x0, width, value):
        if width is None:
            r = self.r
            q = self.q
        else:
            r = (width ** 2) / 50.0
            q = 200.0 / width
        k = p0 / (p0 + r)
        x1 = x0 + k * (value - x0)
        p1 = p0 * (1.0 - k) + q
        return p1, x1

    def _calculate_probability(self, features, class_c):
        prior = self.no_samples_per_class[class_c][1]
        likelihood = self.get_features_prob_per_class(features, class_c)
        return prior * likelihood

    def predict(self, X):
        predictions = []
        for sample in X:
            max_prob = -np.inf
            tmp_class = None
            for class_c in self.no_samples_per_class.keys():
                prob = self._calculate_probability(sample, class_c)
                if prob > max_prob:
                    max_prob = prob
                    tmp_class = class_c
            predictions.append(tmp_class)
        return np.array(predictions)

    def get_features_prob_per_class(self, features, class_c):
        features_prob = []
        for index, value in enumerate(features):
            try:
                if value in self.counter_attribute_per_class[class_c][index].keys():
                    features_prob.append(self.counter_attribute_per_class[class_c][index][value][1])
                else:
                    features_prob.append(0.0)
            except KeyError:
                features_prob.append(0.0)
        likelihood = np.prod((np.array(features_prob) + self._epsilon))
        return likelihood

    def predict_proba(self, X):
        pass
