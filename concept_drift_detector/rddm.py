import math

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class RDDM(BaseDriftDetector):
    """ Reactive Drift Detection Method (RDDM).

        published as:
        Roberto S. M. Barros, Danilo R. L. Cabral, Paulo M. Goncalves Jr.,
        and Silas G. T. C. Santos:
        RDDM: Reactive Drift Detection Method.
        Expert Systems With Applications 90C (2017) pp. 344-355.
        DOI: 10.1016/j.eswa.2017.08.023

    """

    MIN_SIZE_STABLE_CONCEPT = 7000
    MAX_SIZE_CONCEPT = 40000
    DRIFT_LEVEL = 2.258
    WARNING_LEVEL = 1.773
    WARNING_LIMIT = 1400

    def __init__(self):
        super().__init__()

        self.stored_predictions = list()
        self.num_stored_instances = 0
        self._pos = 0
        self._first_pos = 0
        self._last_pos = -1  # this means stored predictions is empty
        self._last_warning_pos = -1
        self._last_warning_instances = -1
        self._inst_num = 0
        self._rddm_drift = False

        self.m_n = 1.0
        self.m_p = 1.0
        self.m_s = 0.0

        self.m_p_min = math.inf
        self.m_s_min = math.inf
        self.m_ps_min = math.inf

        self._is_initialized = False
        self.in_concept_change = False
        self.in_warning_zone = False

        super().reset()

    def reset_learning(self):

        self.m_n = 1.0
        self.m_p = 1.0
        self.m_s = 0.0

        if self.in_concept_change:
            self.m_p_min = math.inf
            self.m_s_min = math.inf
            self.m_ps_min = math.inf

    def add_element(self, input_value):
        """

        :param input_value: it must be 0 or 1

        """

        if self._is_initialized:
            self.reset()
            self._is_initialized = True

        if self._rddm_drift:
            self.reset_learning()
            if self._last_warning_pos != -1:
                self._first_pos = self._last_warning_pos
                self.num_stored_instances = self._last_pos - self._first_pos + 1
                if self.num_stored_instances <= 0:
                    self.num_stored_instances += self.MIN_SIZE_STABLE_CONCEPT

            self._pos = self._first_pos
            for i in range(self.num_stored_instances):
                self.m_p = self.m_p + (self.stored_predictions[self._pos] - self.m_p) / self.m_n
                self.m_s = math.sqrt(self.m_p * (1 - self.m_p) / self.m_n)
                if self.in_concept_change and self.m_p + self.m_s > self.m_ps_min:
                    self.m_p_min = self.m_p
                    self.m_s_min = self.m_s
                    self.m_ps_min = self.m_p + self.m_s

                self.m_n += 1.0
                self._pos = (self._pos + 1) % self.MIN_SIZE_STABLE_CONCEPT

            self._last_warning_pos = -1
            self._last_warning_instances = -1
            self._rddm_drift = False
            self.in_concept_change = False

        self.stored_predictions.append(input_value)
        if self.num_stored_instances < self.MIN_SIZE_STABLE_CONCEPT:
            self.num_stored_instances += 1
        else:
            self._first_pos = self._first_pos + 1 % self.MIN_SIZE_STABLE_CONCEPT
            if self._last_warning_pos == self._last_pos:
                self._last_warning_pos = -1

        self.m_p = self.m_p + (input_value - self.m_p) / self.m_n
        self.m_s = math.sqrt(self.m_p * (1 - self.m_p) / self.m_n)

        self._inst_num += 1
        self.m_n += 1
        self.estimation = self.m_p
        self.in_warning_zone = False

        if self.m_p + self.m_s < self.m_ps_min:
            self.m_p_min = self.m_p
            self.m_s_min = self.m_s
            self.m_ps_min = self.m_p + self.m_s

        if self.m_p + self.m_s > self.m_p_min + self.DRIFT_LEVEL * self.m_s_min:
            self.in_concept_change = True
            self._rddm_drift = True
            if self._last_warning_instances == -1:
                self._first_pos = self._last_pos
                self.num_stored_instances = 1

        elif self.m_p + self.m_s > self.m_p_min + self.WARNING_LEVEL * self.m_s_min:

            # Warning level for warn limit consecutive instances will force drifts
            if self._last_warning_instances != -1 and self._last_warning_instances + self.WARNING_LIMIT <= self._inst_num:
                self.in_concept_change = True
                self._rddm_drift = True
                self._first_pos = self._last_pos
                self.num_stored_instances = 1
                self._last_warning_pos = -1
                self._last_warning_instances = -1
                return

            # Warning zone
            self.in_warning_zone = True
            if self._last_warning_instances == -1:
                self._last_warning_instances = self._inst_num
                self._last_warning_pos = self._last_pos
        else:
            self._last_warning_instances = -1
            self._last_warning_pos = -1

        if self.m_n > self.MAX_SIZE_CONCEPT and not self.in_warning_zone:
            self._rddm_drift = True


