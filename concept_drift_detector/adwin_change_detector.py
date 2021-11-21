from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class ADWINChangeDetector(BaseDriftDetector):
    """ Drift detection method based in ADWIN.
        Parameters
        ----------
        delta : float (default=0.002)
            The delta parameter for the ADWIN algorithm.
        Notes
        -----
        ADWIN [1]_ (ADaptive WINdowing) is an adaptive sliding window algorithm
        for detecting change, and keeping updated statistics about a data stream.
        ADWIN allows algorithms not adapted for drifting data, to be resistant
        to this phenomenon.
        The general idea is to keep statistics from a window of variable size while
        detecting concept drift.
        The algorithm will decide the size of the window by cutting the statistics'
        window at different points and analysing the average of some statistic over
        these two windows. If the absolute value of the difference between the two
        averages surpasses a pre-defined threshold, change is detected at that point
        and all data before that time is discarded.
        References
        ----------
        .. [1] Bifet, Albert, and Ricard Gavalda. "Learning from time-changing data with adaptive windowing."
           In Proceedings of the 2007 SIAM international conference on data mining, pp. 443-448.
           Society for Industrial and Applied Mathematics, 2007.
    """

    def __init__(self, delta=.002):
        super().__init__()
        self.adwin = ADWIN(delta=delta)
        super().reset()

    def add_element(self, input_value):
        err_estim = self.adwin.estimation
        self.adwin.add_element(input_value)
        res_input = self.adwin.detected_change()

        self.in_concept_change = False
        self.in_warning_zone = False

        if self.adwin.detected_warning_zone():
            self.in_warning_zone = True
        if res_input:
            if self.adwin.estimation > err_estim:
                self.in_concept_change = True
                self.in_warning_zone = False

        self.estimation = self.adwin.estimation