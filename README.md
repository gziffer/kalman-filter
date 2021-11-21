## Kalman Filtering for Learning with Evolving Data


The source code implemented to run the experiments shown in the article is organized as follows:
- *concept_drift_detector* : it contains the concept drift detectors that are not present in the scikit-multiflow
library 
    - Adwin
    - ECDD
    - RDDM
    - STEPD
- *dataset* : it contains the datasets used in the experimental evaluation 
    - elist
    - spam
    - usent
    - usenet1
    - usenet2
- *evaluation* : it contains the code of the k-fold distributed cross validation with the prequential evaluation mode
- *models* : it contains the models developed and used in the experimental evaluation
    - DriftDetectionMethodClassifier
    - K-Adwin classifier
    - KalmanNB classifier
- *rewritten_code* : it contains some python files from the scikit-multiflow library
that have been adjusted in order to include KalmanNB at the leaves of the tree-based classifier
- *testing* : it contains the tests run for the experimental evaluation
    - kalmannb: it contains the tests with synthetic and real datasets for kalmannb
    - tree: it contains the tests with artificial and real datasets for tree-based models