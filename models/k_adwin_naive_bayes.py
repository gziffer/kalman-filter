from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.core import BaseSKMObject, ClassifierMixin

from models.kalmannb import KalmanNB


class KAdwinNB(BaseSKMObject, ClassifierMixin):
    """ Incremental Kalman Adwin Naive Bayes classifier for categorical features.

    Performs classic bayesian prediction while making naive assumption that all inputs are independent.
    The trained Naive Bayes classifier predicts for every unlabelled instance the class to which it
    belongs with higher probability.
    This implementation is able to incrementally update its model.

    It works as a wrapper for the class KalmanNB, using firstly Adwin to detect concept drift, and then calling
    partial fit passing also the width of Adwin (which is used by the algorithm to dynamically adjust q and r)

    Notes
    -----
    This estimator is not optimal for a mixture of categorical and numerical features.
    This implementation treats all features from a given stream as categorical.

    """

    def __init__(self, drift_detector: BaseDriftDetector):
        super().__init__()

        self.classifier = KalmanNB()

        self.drift_detection_method = drift_detector

    def partial_fit(self, X, y, classes=None, sample_weight=None):

        add = 0.0 if self.predict(X) == y else 1.0
        self.drift_detection_method.add_element(add)
        self.drift_detection_method.detected_change()
        self.classifier.partial_fit(X, y, width=self.drift_detection_method.width)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_features_prob_per_class(self, features, class_c):
        return self.classifier.get_features_prob_per_class(features, class_c)
