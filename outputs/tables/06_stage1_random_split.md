# 06_stage1_random_split

| variant          | model        | n_features   | recall   | fp_rate   | latency_ms_per_1k   |
|:-----------------|:-------------|:-------------|:---------|:----------|:--------------------|
| ไม่มี Port (ปัจจุบัน) | DecisionTree | 65           | 0.999891 | 0.0006    | 0.1251              |
| ไม่มี Port (ปัจจุบัน) | RandomForest | 65           | 0.9954   | 7.31e-05  | 0.9691              |
| ไม่มี Port (ปัจจุบัน) | XGBoost      | 65           | 0.999583 | 7.31e-05  | 0.2419              |
| มี Port           | DecisionTree | 67           | 0.999855 | 0.0004    | 0.1363              |
| มี Port           | RandomForest | 67           | 0.9965   | 7.31e-05  | 0.8527              |
| มี Port           | XGBoost      | 67           | 0.999764 | 7.31e-05  | 0.2415              |
