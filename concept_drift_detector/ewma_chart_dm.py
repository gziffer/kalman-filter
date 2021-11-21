import math

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class EWMAChartDM(BaseDriftDetector):
    """ Drift detection method based in EWMA Charts.
        Ross, Adams, Tasoulis and Hand, 2012
    """

    def __init__(self, lambda_p=0.2, min_num_instances=0):
        super().__init__()

        self.m_n = 1.0
        self.m_sum = 0.0
        self.m_p = 0.0
        self.m_s = 0.0
        self.lambda_p = lambda_p
        self.z_t = 0.0

        self.min_num_instances = min_num_instances

        self._is_initialized = False
        self.in_concept_change = False

        super().reset()

    def reset(self):
        """ Reset detectors

        Resets statistics.

        Returns
        -------
        EWMAChartDM
            self

        """
        self.__init__(lambda_p=self.lambda_p, min_num_instances=self.min_num_instances)

    def add_element(self, input_value):
        """

        :param input_value: it must be 0 or 1 since EWMA monitors the error rate

        """
        if self.in_concept_change or self._is_initialized:
            self.reset()
            self._is_initialized = True

        self.m_sum += input_value

        self.m_p = self.m_sum / self.m_n

        self.m_s = math.sqrt(self.m_p * (1. - self.m_p) * self.lambda_p *
                             (1.0 - math.pow(1.0 - self.lambda_p, 2.0 * self.m_n)) / (2.0 - self.lambda_p))
        self.m_n += 1.0

        self.z_t += self.lambda_p * (input_value - self.z_t)

        l_t = 3.97 - 6.56 * self.m_p + 48.73 * math.pow(self.m_p, 3) - 330.13 * math.pow(self.m_p, 5) + \
              848.18 * math.pow(self.m_p, 7)

        self.estimation = self.m_p
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if self.m_n < self.min_num_instances:
            return

        if self.z_t > self.m_p + l_t * self.m_s:
            self.in_concept_change = True
        elif self.z_t > self.m_p + 0.5 * l_t * self.m_s:
            self.in_warning_zone = True
        else:
            self.in_warning_zone = False
