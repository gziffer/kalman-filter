from skmultiflow.metrics import WindowClassificationMeasurements
from random import randint
import copy
import numpy as np
import pandas as pd
from skmultiflow.utils.utils import calculate_object_size
from timeit import default_timer as timer
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.led_generator_drift import LEDGeneratorDrift


class KFoldDistributedCrossValidation:
    """ K Fold Distributed Cross Validation

        Class for implementing k fold distributed cross validation, where k models are trained and tested in parallel.
        In the k-fold distributed cross-validation, each instance is used for testing one randomly selected model and
        for training all the others. By applying the prequential evaluation mode, all models are tested
        before the training phase.

    """

    def __init__(self, window_size=1000, iterations=1e6, metric='kappa', width=1):

        self.base_learners = {}
        self.data = None
        self._window_size = window_size
        self._num_iterations = int(iterations)
        self.ensemble_size = 0
        self.metric_to_evaluate = metric
        self._instances_processed = 0
        self._position = 1000
        self._width = width
        self._random_state = np.random.RandomState(112)

    def prepare_for_use(self, model, ensemble_size, single_training=False, artificial=True, name=None, frequency=1e6,
                        concepts=None, stream=None):

        for id in range(ensemble_size):
            self.base_learners[id] = {
                'model': copy.deepcopy(model),
                'window_measurement': WindowClassificationMeasurements(window_size=self._window_size),
                'metric': [],
                'memory': [],
                'processing_time': 0.0
            }
        self.ensemble_size = ensemble_size
        self._special_case = single_training
        self.artificial = artificial
        self.name = name
        self.frequency = int(frequency)
        self._concepts = concepts
        self._stream = stream
        if stream == 'stagger':
            self._stream = STAGGERGenerator(classification_function=0, balance_classes=True)
            self._drift_stream = STAGGERGenerator(classification_function=1, balance_classes=True)
        else:
            self._stream = LEDGeneratorDrift(has_noise=True, noise_percentage=0.1, n_drift_features=0)
            self._drift_stream = LEDGeneratorDrift(has_noise=True, noise_percentage=0.1, n_drift_features=1)

    def evaluate(self):
        if self.artificial:
            for it in range(self._num_iterations):
                if int((it + 1 + self._width/2)) % 1000 == 0:
                    self._stream = self._drift_stream
                    c = self._concepts[int(it/1000)]
                    if self._stream == 'stagger':
                        self._drift_stream = STAGGERGenerator(classification_function=c, balance_classes=True)
                    else: # stream == 'led':
                        self._drift_stream = LEDGeneratorDrift(has_noise=True, noise_percentage=0.1, n_drift_features=c)

                x = -4.0 * float(((it+1) % 1000) - self._position) / float(self._width)
                probability_drift = 1.0 / (1.0 + np.exp(x))
                if self._random_state.rand() > probability_drift:
                    X, y = self._stream.next_sample(1)
                else:
                    X, y = self._drift_stream.next_sample(1)
                self._instances_processed += 1
                self._evaluate(X, y)
        else:
            dataset = pd.read_csv("./../../dataset/" + self.name + ".csv",
                                  sep=",",
                                  index_col=False,
                                  header=None,
                                  chunksize=1,
                                  iterator=True)
            for index, chunk in enumerate(dataset):
                X = pd.DataFrame.to_numpy(chunk.iloc[:, :-1])
                Y = pd.DataFrame.to_numpy(chunk.iloc[:, -1])
                if self.name == 'spam':
                    Y = np.array([1 if x == "spam" else 0 for x in Y])
                elif self.name == 'usenet':
                    X = [np.array([0 if x == 'f' else 1 for x in X[0]])]
                    Y = np.array([1 if x == "no" else 0 for x in Y])
                elif self.name == 'elist':
                    Y = np.array([1 if x == "yes" else 0 for x in Y])
                self._evaluate(X, Y)

    def _evaluate(self, X, Y):
        id_not_train = randint(0, self.ensemble_size - 1)
        for key, b_l in self.base_learners.items():
            # test each instance
            t_1 = timer()
            prediction = b_l['model'].predict(X)[0]
            t_2 = timer()
            b_l['processing_time'] += t_2 - t_1
            b_l['window_measurement'].add_result(Y[0], prediction)
            b_l['memory'].append(calculate_object_size(b_l['model'], 'kB'))
            # cross-validation: train all instances but one
            if key != id_not_train or self.ensemble_size == 1:
                if self._special_case:
                    t_1 = timer()
                    b_l['model'].partial_fit([X[0]], [Y[0]])
                    t_2 = timer()
                else:
                    t_1 = timer()
                    b_l['model'].partial_fit(X, Y)
                    t_2 = timer()
            b_l['processing_time'] += t_2 - t_1
            if self._instances_processed % self.frequency == 0:
                if self.metric_to_evaluate == 'kappa':
                    b_l['metric'].append(b_l['window_measurement'].get_kappa())
                else:  # metric = 'accuracy'
                    b_l['metric'].append(b_l['window_measurement'].get_accuracy())

    def get_measurements(self):
        metrics = []
        memory = []
        time = []
        for _, b_l in self.base_learners.items():
            metrics.append(np.mean(b_l['metric']))
            memory.append(np.mean(b_l['memory']))
            time.append(np.sum(b_l['processing_time']))
        return np.mean(metrics), np.mean(memory), np.mean(time)
