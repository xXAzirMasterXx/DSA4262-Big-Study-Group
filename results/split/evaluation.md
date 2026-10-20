# Held-out split evaluation

| Model               | Dataset | Sites | ROC AUC | PR AUC |
| ------------------- | ------- | ----: | ------: | -----: |
| extra_trees         | data0   | 18276 |  0.9376 | 0.5739 |
| extra_trees         | data1   | 13622 |  0.8632 | 0.4704 |
| extra_trees         | data2   |   199 |  0.9462 | 0.7271 |
| random_forest       | data0   | 18276 |  0.9326 | 0.5282 |
| random_forest       | data1   | 13622 |  0.8461 | 0.4176 |
| random_forest       | data2   |   199 |  0.9318 | 0.6595 |
| gradient_boosting   | data0   | 18276 |  0.9316 | 0.5226 |
| gradient_boosting   | data1   | 13622 |  0.8313 | 0.3893 |
| gradient_boosting   | data2   |   199 |  0.8826 | 0.5590 |
| logistic_regression | data0   | 18276 |  0.8992 | 0.4002 |
| logistic_regression | data1   | 13622 |  0.8122 | 0.3098 |
| logistic_regression | data2   |   199 |  0.6897 | 0.2350 |
| mil                 | data0   | 18276 |  0.9342 | 0.4971 |
| mil                 | data1   | 13622 |  0.8393 | 0.3872 |
| mil                 | data2   |   199 |  0.9377 | 0.8501 |
