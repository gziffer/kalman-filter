import math
from scipy import stats
import numpy as np

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class STEPD(BaseDriftDetector):
    """ Statistical Test of Equal Proportions method (STEPD).
        published as:
        Kyosuke Nishida and Koichiro Yamauchi:
        Detecting Concept Drift Using Statistical Testing.
        Discovery Science 2007, Springer, vol 4755 of LNCS, pp. 264-269.
    """

    WINDOW_SIZE = 30
    ALPHA_DRIFT = 1.25
    ALPHA_WARNING = 2.0

    def __init__(self):
        super().__init__()

        self.stored_predictions = np.zeros(self.WINDOW_SIZE)
        self._first_pos = 0
        self._last_pos = -1  # this means stored predictions is empty

        self._ro = 0
        self._rr = 0
        self._wo = 0.0
        self._wr = 0.0
        self._no = 0
        self._nr = 0
        self._p = 0
        self._Z = 0
        self._size_inverted_sum = 0

        self._is_initialized = False
        self.in_concept_change = False

        super().reset()

    def reset_learning(self):
        self._first_pos = 0
        self._last_pos = 0
        self._wo = 0.0
        self._wr = 0.0
        self._no = 0
        self._nr = 0
        self.in_concept_change = False

    def add_element(self, input_value):
        """

        :param input_value: it must be 0 or 1

        """

        if self._is_initialized:
            self.reset()
            self._is_initialized = True
        elif self.in_concept_change:
            self.reset_learning()

        if self._nr == self.WINDOW_SIZE:
            self._wo = self._wo + self.stored_predictions[self._first_pos]
            self._no += 1
            self._wr = self._wr - self.stored_predictions[self._first_pos]
            self._first_pos += 1
            if self._first_pos == self.WINDOW_SIZE:
                self._first_pos = 0
        else:
            self._nr += 1

        self._last_pos += 1
        if self._last_pos == self.WINDOW_SIZE:
            self._last_pos = 0

        self.stored_predictions[self._last_pos] = input_value
        self._wr += input_value

        self.in_warning_zone = False

        if self._no >= self.WINDOW_SIZE:
            self._ro = self._no - self._wo
            self._rr = self._nr - self._wr
            self._size_inverted_sum = 1.0 / self._no + 1.0 / self._nr
            self._p = (self._ro + self._rr) / (self._no + self._nr)
            self._Z = abs(self._ro / self._no - self._rr / self._nr)
            self._Z -= self._size_inverted_sum / 2.0
            self._Z = self._Z / math.sqrt(self._p * (1.0 - self._p) * self._size_inverted_sum)

            self._Z = stats.norm.pdf(abs(self._Z))
            self._Z = 2 * (1 - self._Z)

            if self._Z < self.ALPHA_DRIFT:
                self.in_concept_change = True
            elif self._Z < self.ALPHA_WARNING:
                self.in_warning_zone = True
