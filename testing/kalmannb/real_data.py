from tqdm import tqdm

import csv
from skmultiflow.bayes.naive_bayes import NaiveBayes
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection.hddm_w import HDDM_W

from models.kalmannb import KalmanNB
from models.k_adwin_naive_bayes import KAdwinNB
from models.drift_detection_method_classifier import DriftDetectionMethodClassifier
from concept_drift_detector.adwin import ADWIN
from concept_drift_detector.adwin_change_detector import ADWINChangeDetector
from concept_drift_detector.ewma_chart_dm import EWMAChartDM
from concept_drift_detector.rddm import RDDM
from concept_drift_detector.stepd import STEPD
from evaluation.kfold_distributed_cross_validation import KFoldDistributedCrossValidation

""" Testing the accuracy of the  algorithms with real datasets using the 10-fold 
distributed cross-validation with the prequential evaluation mode. 

The results are saved in a csv file.

"""

ALGORITHMS = ['Standard NB', 'KalmanNB Q=1 R=1000', 'KalmanNB Q=10 R=2000',
              'K-Adwin', 'Adwin', 'DDM', 'EDDM', 'HDMM_A', 'HDMM_W', 'ECDD', 'RDDM', 'STEPD']
DATASET = "usenet2"  # options: 'usenet', 'usenet1', 'usenet2', 'elist', and 'spam'
BATCH_SIZE = 1
METRIC_TO_EVALUATE = 'kappa'
FREQUENCY = 1

if DATASET == "usenet1" or DATASET == "usenet2":
    nominal_attributes = list(range(100))
    classes = [0, 1]
if DATASET == "usenet":
    nominal_attributes = list(range(659))
    classes = [0, 1]
if DATASET == 'elist':
    classes = [0, 1]
    nominal_attributes = list(range(28000))
else: # SPAM == 'spam'
    classes = [0, 1]
    nominal_attributes = list(range(500))


# Standard naive bayes
std_nb = NaiveBayes(nominal_attributes=nominal_attributes)

# Kalman models with different parameters
methodx1 = KalmanNB(q=1, r=1000)
methodx2 = KalmanNB(q=10.0, r=2000.)

# K-ADWIN model with ADWIN externally
adwin_detector = ADWIN()
k_adwin = KAdwinNB(drift_detector=adwin_detector)

# Naive Bayes + ADWIN as external wrapper
drift_detector = ADWINChangeDetector()
classifier = NaiveBayes(nominal_attributes=nominal_attributes)
adwin = DriftDetectionMethodClassifier(drift_detector=drift_detector, classifier=classifier)

# Naive Bayes + DDM as external wrapper
drift_detector2 = DDM()
classifier2 = NaiveBayes(nominal_attributes=nominal_attributes)
ddm = DriftDetectionMethodClassifier(drift_detector=drift_detector2, classifier=classifier2)

# Naive Bayes + EDDM as external wrapper
drift_detector3 = EDDM()
classifier3 = NaiveBayes(nominal_attributes=nominal_attributes)
eddm = DriftDetectionMethodClassifier(drift_detector=drift_detector3, classifier=classifier3)

# Naive Bayes + HDDM_A as external wrapper
drift_detector4 = HDDM_A()
classifier4 = NaiveBayes(nominal_attributes=nominal_attributes)
hddm_a = DriftDetectionMethodClassifier(drift_detector=drift_detector4, classifier=classifier4)

# Naive Bayes + HDDM_W as external wrapper
drift_detector5 = HDDM_W()
classifier5 = NaiveBayes(nominal_attributes=nominal_attributes)
hddm_w = DriftDetectionMethodClassifier(drift_detector=drift_detector5, classifier=classifier5)

# Naive Bayes + ECDD as external wrapper
drift_detector6 = EWMAChartDM()
classifier6 = NaiveBayes(nominal_attributes=nominal_attributes)
ecdd = DriftDetectionMethodClassifier(drift_detector=drift_detector6, classifier=classifier6)

# Naive Bayes + RDDM as external wrapper
drift_detector7 = RDDM()
classifier7 = NaiveBayes(nominal_attributes=nominal_attributes)
rddm = DriftDetectionMethodClassifier(drift_detector=drift_detector7, classifier=classifier7)

# Naive Bayes + STEPD as external wrapper
drift_detector8 = STEPD()
classifier8 = NaiveBayes(nominal_attributes=nominal_attributes)
stepd = DriftDetectionMethodClassifier(drift_detector=drift_detector8, classifier=classifier8)

models = [std_nb, methodx1, methodx2, k_adwin, adwin, ddm, eddm, hddm_a, hddm_w, ecdd, rddm, stepd]

metrics = {}

for id, alg in tqdm(enumerate(models), total=len(models), leave=True):
    if id < 3:
        single_training = False
    else:
        single_training = True

    evaluator = KFoldDistributedCrossValidation(metric=METRIC_TO_EVALUATE, window_size=BATCH_SIZE)
    evaluator.prepare_for_use(model=alg, ensemble_size=1, single_training=single_training,
                              artificial=False, name=DATASET, frequency=FREQUENCY)
    evaluator.evaluate()
    metrics[ALGORITHMS[id]] = evaluator.get_measurements()


path = "./results/" + DATASET + "_10fold.csv"

print()
print("Dataset " + DATASET)
print("------------------------")
print()
print("Metrics evaluated: " + METRIC_TO_EVALUATE)
print()
print("%36s %8s %6s" %(METRIC_TO_EVALUATE, 'Memory', 'Time'))

with open(path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', METRIC_TO_EVALUATE, 'Memory', 'Time'])
    for key, metric in metrics.items():
        writer.writerow([key, metric[0], metric[1], metric[2]])
        print("%-30s %4.2f %6.2f %8.2f" % (key+": ", metric[0]*100., metric[1], metric[2]))
    print()
