"""
common.py — ฟังก์ชันที่ใช้ร่วมกันหลายสคริปต์

ไฟล์นี้ไม่ได้รันเอง แต่ถูก import โดย 01–05
เหตุผลที่ต้องมี: ถ้าฟังก์ชันวัดผล (evaluate) หรือฟังก์ชันสร้างโมเดล (build_models)
เขียนซ้ำในทุกไฟล์ แล้ววันหนึ่งแก้ไม่ครบ ผลการทดลองจะเทียบกันไม่ได้
รวมไว้ที่เดียว = ทุกสคริปต์ใช้นิยามเดียวกันแน่นอน
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

import config

# คอลัมน์ที่ "ไม่ใช่ฟีเจอร์" — เป็นคำตอบหรือข้อมูลกำกับ
#   Label        = label ดิบจากไฟล์ต้นฉบับ (เช่น 'DDoS')
#   binary_label = 0/1 สำหรับชั้นที่ 1
#   attack_class = ชนิด attack สำหรับชั้นที่ 2
#   source       = มาจากไฟล์ไหน (Normal/OVS/metasploitable) ใช้ตอน zero-shot
# ถ้าเผลอเอาคอลัมน์พวกนี้ไปเป็น input โมเดลจะเห็นคำตอบ = leakage ทันที
META_COLS = ["Label", "binary_label", "attack_class", "source"]


def load_clean() -> pd.DataFrame:
    """โหลดข้อมูลที่ 01_prepare_data.py เตรียมไว้แล้ว

    ใช้ parquet ไม่ใช่ csv เพราะ parquet เก็บชนิดข้อมูล (int/float) ไว้ด้วย
    ถ้าใช้ csv จะต้องมานั่งแปลงชนิดใหม่ทุกครั้ง และเสี่ยงแปลงไม่ตรงกันในแต่ละสคริปต์
    """
    if not config.CLEAN_PARQUET.exists():
        raise SystemExit("ยังไม่มีไฟล์ clean — รัน `python 01_prepare_data.py` ก่อน")
    return pd.read_parquet(config.CLEAN_PARQUET)


def feature_columns(df) -> list[str]:
    """คืนเฉพาะชื่อคอลัมน์ที่เป็นฟีเจอร์จริง (ตัด META_COLS ออก)"""
    return [c for c in df.columns if c not in META_COLS]


def split_xy(df, target):
    """แยกเป็น X (ฟีเจอร์) กับ y (คำตอบ)

    target ระบุว่าจะทำนายอะไร — 'binary_label' สำหรับชั้น 1, 'attack_class' สำหรับชั้น 2
    """
    return df[feature_columns(df)], df[target]


def evaluate(model, X_test, y_test, average="binary") -> dict:
    """วัดผลโมเดล 1 ตัว คืนเป็น dict

    average="binary" → สำหรับ 2 คลาส (ชั้นที่ 1)
    average="macro"  → สำหรับหลายคลาส (ชั้นที่ 2)

    ทำไมต้อง macro ไม่ใช่ weighted:
      macro = เฉลี่ยแบบให้ทุกคลาสน้ำหนักเท่ากัน คลาสที่มี 17 แถวมีเสียงเท่าคลาสที่มี 120,000 แถว
      weighted = ถ่วงตามจำนวนแถว → คลาสใหญ่กลบคลาสเล็กจนมองไม่เห็นว่าคลาสเล็กพัง
    งานตรวจจับ attack สนใจคลาสเล็ก (attack หายาก) จึงต้องใช้ macro

    latency: จับเวลา predict แล้วแปลงเป็น "กี่ ms ต่อ 1000 flows"
    เป็น KPI ของชั้นที่ 1 เพราะอาจารย์กำหนดว่าชั้นแรกต้องเบาและเร็ว
    """
    t0 = time.perf_counter()
    pred = model.predict(X_test)
    elapsed = time.perf_counter() - t0

    return {
        # สัดส่วนที่ทายถูกทั้งหมด — ดูอย่างเดียวไม่ได้เมื่อข้อมูล imbalance
        "accuracy": accuracy_score(y_test, pred),
        # ที่ทายว่าเป็น attack ถูกจริงกี่ % (ยิ่งสูง = false alarm น้อย)
        "precision": precision_score(y_test, pred, average=average, zero_division=0),
        # attack ทั้งหมดที่มี จับได้กี่ % (ยิ่งสูง = พลาดน้อย) ← สำคัญที่สุดสำหรับชั้น 1
        "recall": recall_score(y_test, pred, average=average, zero_division=0),
        # ค่าเฉลี่ยแบบ harmonic ของ precision กับ recall
        "f1": f1_score(y_test, pred, average=average, zero_division=0),
        # elapsed(วินาที) / จำนวนแถว = วินาทีต่อแถว, × 1e6 = มิลลิวินาทีต่อ 1000 แถว
        "latency_ms_per_1k": elapsed / len(X_test) * 1e6,
    }


def build_models(n_classes: int = 2) -> dict:
    """สร้าง 3 โมเดลตามที่อาจารย์กำหนดให้เทียบกัน

    hyperparameter ต้องเหมือนกันทุกที่ที่เรียกใช้ ไม่งั้นเวลาเทียบผลจะไม่รู้ว่า
    ที่ต่างกันเพราะตัวโมเดล หรือเพราะตั้งค่าไม่เหมือนกัน

    n_estimators=100 : follow ค่าที่งานวิจัยสาย SDN-IDS นิยมใช้ เพื่อให้เทียบกับ paper อื่นได้
    random_state     : ตั้ง seed ให้ผลเหมือนเดิมทุกครั้งที่รัน (reproducible)
    n_jobs=-1        : ใช้ CPU ทุก core
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    xgb_kwargs = dict(
        n_estimators=config.N_ESTIMATORS,
        random_state=config.SEED,
        n_jobs=-1,
        tree_method="hist",   # แบ่งค่าฟีเจอร์เป็น bin ก่อน → เร็วกว่ามากบนข้อมูลหลักแสนแถว
    )

    # objective บอก XGBoost ว่างานนี้เป็นแบบไหน — มันจะเลือก loss function ให้เอง
    if n_classes > 2:
        # softmax: แปลง output เป็นความน่าจะเป็นของแต่ละคลาส รวมกันได้ 1
        xgb_kwargs["objective"] = "multi:softprob"
    else:
        # sigmoid (logistic): ได้ค่า 0–1 ตัวเดียว >0.5 = ใช่, <0.5 = ไม่ใช่
        xgb_kwargs["objective"] = "binary:logistic"
        xgb_kwargs["eval_metric"] = "logloss"

    return {
        # ต้นไม้ต้นเดียว — เบาและเร็วที่สุด อ่านกฎได้ตรง ๆ แต่ overfit ง่าย
        "DecisionTree": DecisionTreeClassifier(random_state=config.SEED),
        # หลายต้นแบบสุ่ม แล้วโหวตกัน — ลด overfit แต่ช้ากว่าและอธิบายยากขึ้น
        "RandomForest": RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS, random_state=config.SEED, n_jobs=-1
        ),
        # ต้นไม้ที่สร้างทีละต้นเพื่อแก้ที่ต้นก่อนหน้าทำผิด (boosting) — แม่นสุดในงานตารางแบบนี้
        "XGBoost": XGBClassifier(**xgb_kwargs),
    }


def to_shap_array(sv) -> np.ndarray:
    """ทำ output ของ SHAP ให้เป็นรูปเดียวกันเสมอ: (n_samples, n_features, n_classes)

    ทำไมต้องมีฟังก์ชันนี้: shap คืนค่าไม่เหมือนกันขึ้นกับชนิดโมเดลและเวอร์ชัน
      - sklearn RandomForest แบบหลายคลาส → คืน list ของ array (ทีละคลาส)
      - XGBoost แบบ 2 คลาส              → คืน array 2 มิติ (samples, features)
      - XGBoost แบบหลายคลาส             → คืน array 3 มิติอยู่แล้ว
    ถ้าไม่ normalize ตรงนี้ ทุกสคริปต์ที่ใช้ SHAP ต้องเขียน if-else แยกเคสเอง
    """
    if isinstance(sv, list):
        # list ของ array ทีละคลาส → ซ้อนกันเป็นมิติที่ 3
        return np.stack(sv, axis=-1)

    sv = np.asarray(sv)
    # 2 มิติ (กรณี binary) → เติมมิติคลาสให้เป็น 1 คลาส
    return sv[:, :, None] if sv.ndim == 2 else sv


def mean_abs_shap(sv_array, feature_names, class_names=None) -> pd.DataFrame:
    """แปลง SHAP ดิบ → ตารางความสำคัญของฟีเจอร์ (index=feature, column=class)

    ขั้นตอน:
      1. abs()          — เอาค่าสัมบูรณ์ เพราะเราสนใจ "มีผลมากแค่ไหน"
                          ไม่สนว่าดันไปทางบวกหรือลบ (ถ้าไม่ abs ค่าบวกลบจะหักล้างกันเหลือ ~0)
      2. mean(axis=0)   — เฉลี่ยข้ามทุกแถวข้อมูล → เหลือ (n_features, n_classes)
      3. normalize      — หารด้วยผลรวมของแต่ละคลาส ให้แต่ละคอลัมน์รวมกันได้ 1

    ทำไมข้อ 3 สำคัญมาก:
      คลาสที่โมเดลทายง่าย (เช่น DDoS) จะมีค่า SHAP ใหญ่กว่าคลาสอื่นทั้งแถบ
      ถ้าไม่ normalize แล้วเอาไป union กัน ฟีเจอร์ของ DDoS จะกลืนทุกคลาสหมด
      พอ normalize แล้ว ทุกคลาสมี "งบความสำคัญ" เท่ากันคือ 1 → เทียบกันได้อย่างยุติธรรม
    """
    imp = np.abs(sv_array).mean(axis=0)                # (n_features, n_classes)
    imp = imp / np.clip(imp.sum(axis=0), 1e-12, None)  # clip กันหารด้วย 0
    cols = class_names if class_names is not None else range(imp.shape[1])
    return pd.DataFrame(imp, index=feature_names, columns=cols)


def banner(title):
    """พิมพ์หัวข้อคั่น ให้ output ใน terminal อ่านง่าย"""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def save_table(df, name, index=True):
    """บันทึกตารางเป็น CSV ใน outputs/ — เอาไปวางในรายงานได้เลย"""
    path = config.OUT_DIR / name
    df.to_csv(path, index=index)
    print(f"\n[saved] {path}")
