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


_DROP_CACHE = None


def drop_lists() -> dict:
    """อ่านรายชื่อคอลัมน์ที่ตัดได้แต่ละกลุ่ม (01 เป็นคนสร้าง)"""
    global _DROP_CACHE
    if _DROP_CACHE is None:
        import json
        if not config.DROP_LISTS_JSON.exists():
            raise SystemExit("ยังไม่มี drop lists — รัน `python 01_prepare_data.py` ก่อน")
        _DROP_CACHE = json.loads(config.DROP_LISTS_JSON.read_text(encoding="utf-8"))
    return _DROP_CACHE


# ระดับการตัด: สะสมกันไปเรื่อย ๆ (ระดับหลังตัดทุกอย่างของระดับก่อน + กลุ่มใหม่)
LEVEL_GROUPS = {
    "raw":          [],
    "no_id":        ["id"],
    "no_const":     ["id", "const"],
    "no_port":      ["id", "const", "port"],
    "no_dup":       ["id", "const", "port", "dup"],
    "no_expensive": ["id", "const", "port", "dup", "expensive"],
}


def feature_columns(df, level: str = "no_port") -> list[str]:
    """คืนชื่อคอลัมน์ที่เป็นฟีเจอร์ ตามระดับการตัดที่เลือก

    level="raw"          ไม่ตัดอะไร (identifier ถูก encode เป็นเลขแล้ว)
    level="no_port"      ค่าเริ่มต้น = ที่ใช้กันมาตลอด
    level="shap_top15"   ไม่รองรับที่นี่ — อ่านจาก 03_feature_sets.json แทน
    """
    if level not in LEVEL_GROUPS:
        raise ValueError(f"ไม่รู้จักระดับ '{level}' — เลือกจาก {list(LEVEL_GROUPS)}")

    dl = drop_lists()
    skip = set(META_COLS)
    for g in LEVEL_GROUPS[level]:
        skip |= set(dl[g])
    return [c for c in df.columns if c not in skip]


def make_split(df, kind: str = "random"):
    """แบ่ง train/test 3 แบบ — คืน (train_df, test_df)

    random    สุ่ม 80/20 จากข้อมูลปนกันหมด (stratify ตามคลาส)
              → วัดประสิทธิภาพในสภาพแวดล้อมเดิม จับ leakage ไม่ได้
    ovs2meta  train = Normal(ครึ่ง) + OVS   test = Normal(อีกครึ่ง) + metasploitable
    meta2ovs  ทิศกลับ
              → วัดว่า generalize ข้ามสภาพแวดล้อมได้ไหม ตรงนี้แหละที่ leakage โผล่

    ทำไม Normal ต้องแบ่งครึ่ง: OVS/metasploitable ไม่มีแถว Normal เลย
    ถ้าไม่ใส่ Normal ใน test จะวัด false positive ไม่ได้
    และถ้าใส่ทั้งหมดทั้งสองฝั่ง แถวเดียวกันจะอยู่ทั้ง train และ test = leakage
    """
    from sklearn.model_selection import train_test_split

    if kind == "random":
        return train_test_split(df, test_size=config.TEST_SIZE,
                                random_state=config.SEED,
                                stratify=df["attack_class"])

    # B/C ใช้เฉพาะคลาสที่มีทั้งสอง testbed ไม่งั้น macro-F1 เพี้ยน
    d = df[df["attack_class"].isin(config.SHARED_CLASSES)]
    normal = d[d["source"] == "Normal"]
    n_a, n_b = train_test_split(normal, test_size=0.5, random_state=config.SEED)

    ovs = d[d["source"] == "OVS"]
    mt = d[d["source"] == "metasploitable"]
    a, b = (ovs, mt) if kind == "ovs2meta" else (mt, ovs)

    return (pd.concat([n_a, a], ignore_index=True),
            pd.concat([n_b, b], ignore_index=True))


def split_xy(df, target, level: str = "no_port"):
    """แยกเป็น X (ฟีเจอร์) กับ y (คำตอบ)

    target ระบุว่าจะทำนายอะไร — 'binary_label' สำหรับชั้น 1, 'attack_class' สำหรับชั้น 2
    """
    return df[feature_columns(df, level)], df[target]


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

    # ถ้ามีผลจาก 02b (Random Search) ให้ใช้ค่าที่จูนแล้ว
    # ไม่งั้นใช้ค่า default — แต่ต้องบังคับ max_depth ให้เท่ากันทุกโมเดล
    # ไม่งั้น sklearn (None = ไม่จำกัด) กับ XGBoost (6) จะเทียบกันไม่ยุติธรรม
    tuned = {}
    _p = config.OUT_DIR / "02b_best_params.json"
    if _p.exists():
        import json as _json
        tuned = _json.loads(_p.read_text(encoding="utf-8"))

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

    dt_kwargs = dict(random_state=config.SEED)
    rf_kwargs = dict(n_estimators=config.N_ESTIMATORS,
                     random_state=config.SEED, n_jobs=-1)

    dt_kwargs.update(tuned.get("DecisionTree", {}))
    rf_kwargs.update(tuned.get("RandomForest", {}))
    xgb_kwargs.update(tuned.get("XGBoost", {}))

    return {
        # ต้นไม้ต้นเดียว — เบาและเร็วที่สุด อ่านกฎได้ตรง ๆ แต่ overfit ง่าย
        "DecisionTree": DecisionTreeClassifier(**dt_kwargs),
        # หลายต้นแบบสุ่ม แล้วโหวตกัน — ลด overfit แต่ช้ากว่าและอธิบายยากขึ้น
        "RandomForest": RandomForestClassifier(**rf_kwargs),
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


def _fmt(v):
    """จัดรูปแบบตัวเลขให้อ่านง่ายแต่ไม่เสียข้อมูล

    ปัด 4 ตำแหน่งเฉย ๆ ไม่ได้ เพราะค่าอย่าง fp_rate = 0.000073 จะกลายเป็น 0.0
    จึงต้องดูขนาดของค่าก่อนแล้วเลือกรูปแบบ
    """
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, float):
        if v != v:                      # NaN
            return "-"
        if v == 0:
            return "0"
        if abs(v) < 1e-4:               # เล็กมาก → เลขยกกำลัง
            return f"{v:.2e}"
        if 0.999 < abs(v) < 1:          # ใกล้ 1 มาก → ต้องใช้ 6 ตำแหน่ง
            # ไม่งั้น 0.999973 กับ 0.999583 จะกลายเป็น 1.0000 เหมือนกันหมด
            # ซึ่งเป็นตัวเลขที่งานนี้ต้องแยกให้ออก (recall ของ stage 1)
            return f"{v:.6f}"
        if abs(v) >= 1000:              # ใหญ่ → ใส่คอมมา
            return f"{v:,.1f}"
        return f"{v:.4f}"
    return str(v)


def save_table(df, name, index=True):
    """บันทึกตารางเป็น 2 รูปแบบ

    1. outputs/<name>.csv        — ค่าดิบเต็มความละเอียด สำหรับให้สคริปต์อื่นอ่านต่อ
    2. outputs/tables/<name>.md  — ตาราง markdown ปัดเลขแล้ว สำหรับคนอ่าน
                                   เปิดใน VS Code กด Cmd+Shift+V เห็นเป็นตารางเลย

    ที่ต้องมี 2 แบบเพราะ CSV ที่มีทศนิยม 16 ตำแหน่งอ่านด้วยตาไม่ไหว
    แต่ถ้าปัดเลขใน CSV แล้วสคริปต์ปลายน้ำจะได้ค่าที่คลาดเคลื่อน
    """
    path = config.OUT_DIR / name
    df.to_csv(path, index=index)

    md_dir = config.OUT_DIR / "tables"
    md_dir.mkdir(exist_ok=True)
    md_path = md_dir / (path.stem + ".md")

    pretty = df.copy()
    for c in pretty.columns:
        pretty[c] = pretty[c].map(_fmt)

    # disable_numparse=True สำคัญ: ไม่งั้น tabulate จะแปลง string ที่เราจัดรูปแบบไว้แล้ว
    # กลับเป็นตัวเลขและจัดใหม่เอง ทำให้ "0.999973" กลายเป็น "1"
    lines = [f"# {path.stem}", "",
             pretty.to_markdown(index=index, disable_numparse=True), ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[saved] {path}")
    print(f"[saved] {md_path}")
