from pathlib import Path
from tqdm import tqdm
import csv
from random import randint

from rewritten_code.trees.hoeffding_tree import HoeffdingTreeClassifier
from rewritten_code.trees.hoeffding_adaptive_tree import HoeffdingAdaptiveTreeClassifier
from rewritten_code.trees.extremely_fast_decision_tree import ExtremelyFastDecisionTreeClassifier
from rewritten_code.meta.adaptive_random_forests import AdaptiveRandomForestClassifier
from evaluation.kfold_distributed_cross_validation import KFoldDistributedCrossValidation

""" Test the algorithms with different Stream generator with different width of concept drift using the 10-fold 
distributed cross-validation with the prequential evaluation mode. 

The results are saved in a csv file.

"""

STREAM = 'stagger'  # options: 'stagger' and 'led'
METRIC_TO_EVALUATE = 'kappa'
ALGORITHMS = ['Hoeffding Tree', 'Hoeffding Tree Kalman', 'HAT', 'HAT Kalman', 'EFDT', 'EFDT Kalman',
            'ARF', 'ARF Kalman']
N_ITERATIONS = 100001  # total number of iterations
WIDTH = 1  # width of the concept drift

if STREAM == 'led':
    path = "./results/" + STREAM + "_10fold_" + str(WIDTH) + ".csv"
    nominal_attributes = range(24)
    n_concepts = 7
else:
    path = "./results/" + STREAM + "_10fold_" + str(WIDTH) + "2.csv"
    nominal_attributes = range(3)
    n_concepts = 2

# generate concepts
concepts = [0]
for i in range(1000):
    c = randint(0, n_concepts)
    while c == concepts[-1]:
        c = randint(0, n_concepts)
    concepts.append(c)

Path("./results/").mkdir(parents=True, exist_ok=True)

# Tree models with different parameters
ht = HoeffdingTreeClassifier(nominal_attributes=nominal_attributes)
ht_kalman = HoeffdingTreeClassifier(leaf_prediction='kalman', nominal_attributes=nominal_attributes)
hat = HoeffdingAdaptiveTreeClassifier(nominal_attributes=nominal_attributes)
hat_kalman = HoeffdingAdaptiveTreeClassifier(leaf_prediction='kalman', nominal_attributes=nominal_attributes)
efdt = ExtremelyFastDecisionTreeClassifier(nominal_attributes=nominal_attributes)
efdt_kalman = ExtremelyFastDecisionTreeClassifier(leaf_prediction='kalman', nominal_attributes=nominal_attributes)
arf = AdaptiveRandomForestClassifier(nominal_attributes=nominal_attributes)
arf_kalman = AdaptiveRandomForestClassifier(leaf_prediction='kalman', nominal_attributes=nominal_attributes)

models = [ht, ht_kalman, hat, hat_kalman, efdt, efdt_kalman, arf, arf_kalman]
metrics = {}

for id, alg in tqdm(enumerate(models), total=len(models), leave=True):
    evaluator = KFoldDistributedCrossValidation(iterations=N_ITERATIONS,
                                                metric=METRIC_TO_EVALUATE,
                                                width=WIDTH)
    evaluator.prepare_for_use(model=alg,
                              ensemble_size=10,
                              single_training=False,
                              frequency=N_ITERATIONS,
                              concepts=concepts,
                              stream=STREAM)
    evaluator.evaluate()
    metrics[ALGORITHMS[id]] = evaluator.get_measurements()

print()
print(STREAM)
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
