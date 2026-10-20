# Full-data evaluation (biased)

| Model               | Dataset |  Sites | ROC AUC | PR AUC |
| ------------------- | ------- | -----: | ------: | -----: |
| extra_trees         | data0   | 121838 |  1.0000 | 1.0000 |
| extra_trees         | data1   |  90810 |  1.0000 | 1.0000 |
| extra_trees         | data2   |   1323 |  1.0000 | 1.0000 |
| random_forest       | data0   | 121838 |  1.0000 | 1.0000 |
| random_forest       | data1   |  90810 |  1.0000 | 1.0000 |
| random_forest       | data2   |   1323 |  1.0000 | 1.0000 |
| gradient_boosting   | data0   | 121838 |  0.9916 | 0.7921 |
| gradient_boosting   | data1   |  90810 |  0.9882 | 0.7986 |
| gradient_boosting   | data2   |   1323 |  0.9841 | 0.9454 |
| logistic_regression | data0   | 121838 |  0.8977 | 0.4038 |
| logistic_regression | data1   |  90810 |  0.8140 | 0.3307 |
| logistic_regression | data2   |   1323 |  0.7706 | 0.3057 |
| mil                 | data0   | 121838 |  0.9427 | 0.5144 |
| mil                 | data1   |  90810 |  0.8750 | 0.4128 |
| mil                 | data2   |   1323 |  0.9653 | 0.8831 |
