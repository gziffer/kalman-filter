# Kalman Filtering for Learning with Evolving Data


The source code implemented to run the experiments shown in the article is organized as follows:
- *concept_drift_detector* : it contains the concept drift detectors that are not present in the scikit-multiflow
library 
    - Adwin
    - ECDD
    - RDDM
    - STEPD
- *dataset* : it contains the datasets used in the experimental evaluation
    - spam
    - usent
    - usenet1
    - usenet2
    - elist (not included) available [here](http://mlkd.csd.auth.gr/concept_drift.html)
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

## Citing `KalmanNB` and `HoeffdingKalmanTree`

If `KalmanNB` and `HoeffdingKalmanTree` have been useful for your research and you would like to cite them in a academic publication, please use the following Bibtex entry:

```bibtex
@INPROCEEDINGS{ziffer2021kalman,
  author={Ziffer, Giacomo and Bernardo, Alessio and Valle, Emanuele Della and Bifet, Albert},
  booktitle={2021 IEEE International Conference on Big Data (Big Data)}, 
  title={Kalman Filtering for Learning with Evolving Data Streams}, 
  year={2021},
  organization={IEEE},
  volume={},
  number={},
  pages={5337-5346},
  doi={10.1109/BigData52589.2021.9671365}
}
```
