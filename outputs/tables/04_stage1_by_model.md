# 04_stage1_by_model

| feature_set          | n_features   | DecisionTree   | RandomForest   | XGBoost   |
|:---------------------|:-------------|:---------------|:---------------|:----------|
| attack7_intersection | 2            | 0.9907         | 0.9910         | 0.9916    |
| normal8_intersection | 2            | 0.9907         | 0.9910         | 0.9916    |
| attack7_mean         | 15           | 0.999782       | 0.9965         | 0.999437  |
| binary_topk          | 15           | 0.999891       | 0.9970         | 0.999583  |
| normal8_mean         | 15           | 0.999800       | 0.9962         | 0.999474  |
| binary_dynamic       | 21           | 0.999873       | 0.9969         | 0.999619  |
| attack7_union        | 37           | 0.999855       | 0.9961         | 0.999492  |
| normal8_union        | 39           | 0.999891       | 0.9961         | 0.999655  |
| attack7_dynamic      | 40           | 0.999909       | 0.9964         | 0.999601  |
| normal8_dynamic      | 41           | 0.999891       | 0.9961         | 0.999673  |
| all_features         | 65           | 0.999891       | 0.9954         | 0.999583  |
