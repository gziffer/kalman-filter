from tqdm import tqdm

import csv
from rewritten_code.trees.hoeffding_tree import HoeffdingTreeClassifier
from rewritten_code.trees.hoeffding_adaptive_tree import HoeffdingAdaptiveTreeClassifier
from rewritten_code.trees.extremely_fast_decision_tree import ExtremelyFastDecisionTreeClassifier
from rewritten_code.meta.adaptive_random_forests import AdaptiveRandomForestClassifier
from evaluation.kfold_distributed_cross_validation import KFoldDistributedCrossValidation

""" Testing the accuracy of the  algorithms with real datasets using the 10-fold 
distributed cross-validation with the prequential evaluation mode. 

The results are saved in a csv file.

"""

ALGORITHMS = ['Hoeffding Tree', 'Hoeffding Tree Kalman', 'HAT', 'HAT Kalman',
              'EFDT', 'EFDT Kalman', 'ARF', 'ARF Kalman']
DATASET = "elist"  # options: 'usenet', 'usenet1', 'usenet2', 'elist', and 'spam'
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
if DATASET == 'phishing':
    classes = [-1, 0, 1]
    nominal_attributes = list(range(9))
else:
    classes = [0, 1]
    nominal_attributes = list(range(500))



# Tree models with different parameters
ht = HoeffdingTreeClassifier(nominal_attributes=nominal_attributes)
ht_kalman = HoeffdingTreeClassifier(leaf_prediction='kalman', nominal_attributes=nominal_attributes)
hat = HoeffdingAdaptiveTreeClassifier(nominal_attributes=nominal_attributes)
hat_kalman = HoeffdingAdaptiveTreeClassifier(leaf_prediction='kalman', nominal_attributes=nominal_attributes)
efdt = ExtremelyFastDecisionTreeClassifier(nominal_attributes=nominal_attributes)
efdt_kalman = ExtremelyFastDecisionTreeClassifier(leaf_prediction='kalman', nominal_attributes=nominal_attributes)
arf = AdaptiveRandomForestClassifier(nominal_attributes=nominal_attributes,)
arf_kalman = AdaptiveRandomForestClassifier(leaf_prediction='kalman', nominal_attributes=nominal_attributes)

models = [ht, ht_kalman, hat, hat_kalman, efdt, efdt_kalman, arf, arf_kalman]
metrics = {}

for id, alg in tqdm(enumerate(models), total=len(models), leave=True):

    evaluator = KFoldDistributedCrossValidation(metric=METRIC_TO_EVALUATE, window_size=BATCH_SIZE)
    evaluator.prepare_for_use(model=alg, ensemble_size=1, single_training=False,
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
