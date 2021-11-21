from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.core import BaseSKMObject, ClassifierMixin, clone


class DriftDetectionMethodClassifier(BaseSKMObject, ClassifierMixin):
    """ Drift Detection Method Classifier

    Class for handling concept drift data sets with a wrapper on a classifier.

    """

    DDM_INCONTROL_LEVEL = 0

    DDM_WARNING_LEVEL = 1

    DDM_OUTCONTROL_LEVEL = 2

    def __init__(self, drift_detector: BaseDriftDetector, classifier):
        super().__init__()

        self.classifier = classifier
        self.new_classifier = clone(self.classifier)
        self.clean_classifier = clone(classifier)
        self.drift_detection_method = drift_detector
        self.ddm_level = None
        self.change_detected = 0
        self.warning_detected = 0
        self.new_classifier_reset = False
        self.n_instance = 1

    def is_warning_detected(self):
        return self.ddm_level == self.DDM_WARNING_LEVEL

    def is_change_detected(self):
        return self.ddm_level == self.DDM_OUTCONTROL_LEVEL

    def partial_fit(self, X, y, classes=None, sample_weight=None):

        add = 0.0 if self.predict(X) == y else 1.0
        self.drift_detection_method.add_element(add)

        self.ddm_level = self.DDM_INCONTROL_LEVEL
        if self.drift_detection_method.detected_change():
            self.ddm_level = self.DDM_OUTCONTROL_LEVEL
        if self.drift_detection_method.detected_warning_zone():
            self.ddm_level = self.DDM_WARNING_LEVEL

        if self.ddm_level == self.DDM_WARNING_LEVEL:

            if self.new_classifier_reset:
                self.warning_detected += 1
                self.new_classifier.reset()
                self.new_classifier_reset = False

            self.new_classifier.partial_fit(X, y, classes=classes)

        elif self.ddm_level == self.DDM_OUTCONTROL_LEVEL:
            self.change_detected += 1
            self.classifier = None
            self.classifier = self.new_classifier

            self.new_classifier = clone(self.clean_classifier)
            self.new_classifier.reset()

        elif self.ddm_level == self.DDM_INCONTROL_LEVEL:
            self.new_classifier_reset = True

        self.classifier.partial_fit(X, y, classes=classes)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)