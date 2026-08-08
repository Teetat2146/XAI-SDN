"""
02b — Hyperparameter Optimization ด้วย Random Search

ที่มา:
    เดิมเราใช้ค่า default ของ library ทั้งหมด ซึ่งมีปัญหาว่า `max_depth` ไม่เท่ากัน
    (sklearn DT/RF = None ไม่จำกัด · XGBoost = 6) → เทียบกันไม่ยุติธรรม

    งานที่ตีพิมพ์บน InSDN เดียวกันใช้ Random Search 30 settings
    และ **จำกัด max_depth ทุกโมเดลที่ 6-12** ไฟล์นี้ทำตาม methodology เดียวกัน

    อ้างอิง: Table I "Hyperparameter space of random search"
    (2022 ASYU, GBDT + SHAP on CIC-IDS2017 / CSE-CIC-IDS2018 / InSDN)

วิธี:
    RandomizedSearchCV(n_iter=30, cv=3, scoring='f1_macro')
    จูนบน **train set เท่านั้น** ด้วย cross-validation
    → ไม่แตะ test set เลย ไม่งั้นเป็น leakage

    จูนที่ระดับ no_const (67 ฟีเจอร์) ระดับเดียว แล้วเอาค่าที่ได้ไปใช้ทุกระดับ
    เพราะถ้าจูนทุกระดับจะเป็น 7 × 3 × 90 fits = หลายชั่วโมง

รัน:  python 02b_hyperparam_search.py   (ต้องรัน 01 ก่อน)
      แล้วรัน 02 ใหม่เพื่อใช้ค่าที่จูนได้
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import common
import config

TUNE_LEVEL = "no_const"     # ระดับที่ใช้จูน
N_ITER = 30                 # ตามที่ paper ใช้
CV = 3
SEARCH_SAMPLE = 120_000     # subsample ตอนจูนเพื่อความเร็ว (stratified)

# ---- ช่วงค่าจาก Table I ของ paper ----
# max_depth 6,8,10,12 เหมือนกันทุกโมเดล ← จุดสำคัญที่แก้ปัญหาเดิม
SPACES = {
    "DecisionTree": {
        # paper ไม่มี DT (เขาใช้ CatBoost/LightGBM แทน)
        # ใช้ช่วงเดียวกับ RandomForest เพื่อให้เทียบกันได้
        "max_depth": [6, 8, 10, 12],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": [None, "sqrt", "log2"],
    },
    "RandomForest": {
        "n_estimators": [80, 100, 120, 140],
        "max_depth": [6, 8, 10, 12],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"],
    },
    "XGBoost": {
        "n_estimators": [80, 100, 120, 140],
        "max_depth": [6, 8, 10, 12],
        "learning_rate": [0.10, 0.20, 0.30],
        "reg_lambda": [0.01, 1, 10, 100],      # Regularizer ในตาราง
        "colsample_bytree": [0.6, 0.8, 1.0],   # Feature Fraction
    },
}


def base_model(name, n_classes):
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=config.SEED)
    if name == "RandomForest":
        return RandomForestClassifier(random_state=config.SEED, n_jobs=-1)
    return XGBClassifier(random_state=config.SEED, n_jobs=-1, tree_method="hist",
                         objective="multi:softprob", num_class=n_classes)


def main():
    df = common.load_clean()
    feats = common.feature_columns(df, level=TUNE_LEVEL)
    train, _ = common.make_split(df, "random")

    le = LabelEncoder().fit(df["attack_class"])
    y = le.transform(train["attack_class"])
    X = train[feats]

    # subsample แบบ stratified เพื่อให้คลาสเล็ก (U2R 17 แถว) ไม่หายไป
    if len(X) > SEARCH_SAMPLE:
        X, _, y, _ = train_test_split(X, y, train_size=SEARCH_SAMPLE,
                                      random_state=config.SEED, stratify=y)

    common.banner(f"Random Search — ระดับ {TUNE_LEVEL} ({len(feats)} ฟีเจอร์)")
    print(f"  ข้อมูลที่ใช้จูน {len(X):,} แถว (subsample จาก {len(train):,})")
    print(f"  {N_ITER} settings × {CV}-fold CV = {N_ITER * CV} fits ต่อโมเดล")
    print(f"  scoring = f1_macro | จูนบน train เท่านั้น ไม่แตะ test\n")

    best = {}
    for name, space in SPACES.items():
        t0 = time.perf_counter()
        search = RandomizedSearchCV(
            base_model(name, len(le.classes_)), space,
            n_iter=N_ITER, cv=CV, scoring="f1_macro",
            random_state=config.SEED, n_jobs=1, verbose=0,
        )
        search.fit(X, y)
        dt = time.perf_counter() - t0

        best[name] = search.best_params_
        print(f"  {name:14s} CV f1_macro {search.best_score_:.4f}  ({dt/60:.1f} นาที)")
        for k, v in sorted(search.best_params_.items()):
            print(f"      {k:20s} {v}")
        print()

    path = config.OUT_DIR / "02b_best_params.json"
    path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"[saved] {path}")
    print("\n→ รัน `python 02_train_stage2.py` ใหม่เพื่อใช้ค่าที่จูนได้")


if __name__ == "__main__":
    main()
