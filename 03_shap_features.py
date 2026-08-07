"""
03 — SHAP: หาฟีเจอร์สำคัญของแต่ละ attack แล้วรวมเป็นชุดฟีเจอร์

ตำแหน่งของไฟล์นี้ใน flow ของงาน:
    01 เตรียมข้อมูล → 02 เทรน stage 2 → [03 ตัวนี้] → 04 เอาฟีเจอร์ไปเทรน stage 1

แนวคิดของอาจารย์:
    เทรน stage 2 (multi-class) ก่อน แล้วรัน SHAP ดูว่า attack แต่ละชนิด
    ใช้ฟีเจอร์อะไรบ้าง จากนั้นรวมฟีเจอร์จากทุก attack เข้าด้วยกัน
    แล้วเอาชุดที่คัดแล้วไปเทรน stage 1 ให้เบาลง

ปัญหาที่ต้องตอบ:
    แต่ละคลาสได้ฟีเจอร์ไม่เหมือนกัน แล้วจะรวมยังไง?
    ไฟล์นี้ไม่ฟันธงวิธีเดียว แต่ผลิตชุดฟีเจอร์ออกมาหลายแบบให้ 04 เอาไปทดสอบเทียบกัน

ไฟล์นี้ไม่เทรน stage 1 เอง — หน้าที่เดียวคือผลิตชุดฟีเจอร์ออกมาเป็น JSON

รัน:  python 03_shap_features.py   (ต้องรัน 01 และ 02 ก่อน)
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import common
import config


def merge_strategies(imp: pd.DataFrame) -> dict[str, list[str]]:
    """สร้างชุดฟีเจอร์ตาม 4 วิธีรวม

    imp: ตารางความสำคัญจาก common.mean_abs_shap
         index = ชื่อฟีเจอร์, columns = ชื่อคลาส, ค่า = ความสำคัญ (normalize แล้ว)
    """
    k = config.TOPK

    # Top-K ของแต่ละคลาส เก็บเป็น set เพื่อให้ทำ union/intersection ได้ง่าย
    top_per_class = {c: set(imp[c].nlargest(k).index) for c in imp.columns}

    # --- วิธีที่ 1: Union ---
    # ฟีเจอร์ที่ติด Top-K ของคลาสใดคลาสหนึ่งก็เอาหมด
    # ข้อดี: ไม่พลาดฟีเจอร์ที่สำคัญเฉพาะบางคลาส
    # ข้อเสีย: ฟีเจอร์บวม และตัวที่สำคัญเฉพาะคลาสเล็กอาจเป็น noise สำหรับคลาสใหญ่
    union = set().union(*top_per_class.values())

    # --- วิธีที่ 2: Intersection ---
    # เอาเฉพาะที่ติด Top-K ของทุกคลาสพร้อมกัน
    # ข้อดี: ได้ฟีเจอร์น้อยมากและสำคัญกับทุกคลาสจริง
    # ข้อเสีย: มักเหลือน้อยเกินไปจนแยกอะไรไม่ออก
    inter = set(imp.index).intersection(*top_per_class.values())

    # --- วิธีที่ 3: Global mean ---
    # เฉลี่ยความสำคัญข้ามทุกคลาสก่อน แล้วตัด Top-K ครั้งเดียว
    # = สมมติฐาน "ใช้ฟีเจอร์ชุดเดียวกันทุกคลาส" ที่อาจารย์ให้ลอง
    global_top = set(imp.mean(axis=1).nlargest(k).index)

    # --- วิธีที่ 4: Dynamic-K ---
    # แต่ละคลาสเก็บฟีเจอร์จนความสำคัญสะสมถึงเกณฑ์แล้วหยุด → K ต่างกันตามคลาส
    # ตรงกับที่อาจารย์เสนอว่า "ไม่ fix ตายตัวว่าต้อง 5 หรือ 10 อันแรก"
    dynamic = set()
    dyn_k = {}
    for c in imp.columns:
        s = imp[c].sort_values(ascending=False)
        # cumsum = ผลรวมสะสม นับว่ามีกี่ตัวที่ยังไม่ถึงเกณฑ์ แล้ว +1 เพื่อรวมตัวที่ทำให้ถึง
        n = int((s.cumsum() < config.CUMULATIVE_THRESHOLD).sum()) + 1
        dyn_k[c] = n
        dynamic |= set(s.index[:n])

    print(f"\n  dynamic-K เลือกได้กี่ฟีเจอร์ต่อคลาส (threshold {config.CUMULATIVE_THRESHOLD}):")
    for c, n in dyn_k.items():
        print(f"    {str(c):16s} K = {n}")

    return {
        f"union_top{k}": sorted(union),
        f"intersection_top{k}": sorted(inter),
        f"global_mean_top{k}": sorted(global_top),
        "dynamic_k": sorted(dynamic),
    }


def shap_from_binary_model(df) -> list[str]:
    """ตัวควบคุม: หาฟีเจอร์สำคัญจากโมเดล binary โดยตรง

    ทำไมต้องมี:
        SHAP จาก stage 2 ตอบคำถามว่า "ฟีเจอร์ไหนช่วยแยก DDoS ออกจาก Probe"
        แต่ stage 1 ถามคนละคำถาม: "ฟีเจอร์ไหนช่วยแยก Normal ออกจาก Attack"
        ฟีเจอร์ที่เก่งเรื่องแรกไม่จำเป็นต้องเก่งเรื่องที่สอง

        ชุดนี้จึงเป็น "ตัวควบคุม" ไว้เทียบว่าการยืมฟีเจอร์จาก stage 2 มาใช้กับ stage 1
        (ตามแนวคิดของอาจารย์) ให้ผลดีพอ ๆ กับการหาจาก stage 1 เองหรือไม่
        ถ้าพอ ๆ กัน = พิสูจน์ว่าแนวคิดใช้ได้จริง
        ถ้าแย่กว่าชัดเจน = เป็นข้อค้นพบที่เอาไปเขียน discussion ได้
    """
    X, y = common.split_xy(df, "binary_label")
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )

    m = XGBClassifier(
        n_estimators=config.N_ESTIMATORS, random_state=config.SEED,
        n_jobs=-1, tree_method="hist", objective="binary:logistic",
    )
    m.fit(X_train, y_train)

    n = min(config.SHAP_SAMPLE, len(X_train))
    X_shap = X_train.sample(n=n, random_state=config.SEED)
    sv = common.to_shap_array(shap.TreeExplainer(m).shap_values(X_shap))
    imp = common.mean_abs_shap(sv, list(X_shap.columns))   # binary → คอลัมน์เดียว

    return sorted(imp.iloc[:, 0].nlargest(config.TOPK).index)


def main():
    # ================================================================
    # ส่วนที่ 1 — เตรียมข้อมูลให้ตรงกับที่ 02 ใช้เทรน
    # ================================================================
    # ต้องใช้ seed และ test_size เดียวกับ 02 เป๊ะ ไม่งั้น train set คนละชุด
    # แล้ว SHAP จะคำนวณจากข้อมูลที่โมเดลไม่ได้เห็นตอนเทรน
    df_all = common.load_clean()
    df = df_all[df_all["binary_label"] == 1].reset_index(drop=True)

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df["attack_class"]))
    X = df[common.feature_columns(df)]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )

    model_path = config.MODEL_DIR / "stage2_best.pkl"
    if not model_path.exists():
        raise SystemExit("ยังไม่มีโมเดล stage 2 — รัน `python 02_train_stage2.py` ก่อน")
    model = joblib.load(model_path)

    # ================================================================
    # ส่วนที่ 2 — คำนวณ SHAP บน stage 2
    # ================================================================
    common.banner("คำนวณ SHAP จากโมเดล stage 2 (TreeExplainer)")

    # ใช้ train set เท่านั้น — ถ้าคำนวณบน test set แล้วเอาไปเลือกฟีเจอร์
    # เท่ากับแอบดูคำตอบของ test set มาช่วยตัดสินใจ = leakage
    #
    # subsample เพราะ SHAP กินแรม: array ขนาด (แถว × ฟีเจอร์ × คลาส)
    # 5,000 แถวก็พอให้ค่าเฉลี่ยนิ่งแล้ว (เป็นวิธีมาตรฐาน เขียนระบุใน methodology ได้)
    n = min(config.SHAP_SAMPLE, len(X_train))
    X_shap = X_train.sample(n=n, random_state=config.SEED)
    print(f"  subsample {n:,} แถวจาก train set ({len(X_train):,} แถว)")

    # TreeExplainer = SHAP เวอร์ชันเฉพาะโมเดลต้นไม้ เร็วกว่าเวอร์ชันทั่วไปหลายพันเท่า
    explainer = shap.TreeExplainer(model)
    sv = common.to_shap_array(explainer.shap_values(X_shap))
    print(f"  shap values shape = {sv.shape}  (samples, features, classes)")

    imp = common.mean_abs_shap(sv, list(X_shap.columns), list(le.classes_))
    common.save_table(imp, "03_shap_importance_per_class.csv")

    common.banner("Top 10 ฟีเจอร์ของแต่ละคลาส")
    for c in imp.columns:
        print(f"\n  [{c}]")
        for f, v in imp[c].nlargest(10).items():
            print(f"    {v:.4f}  {f}")

    # ================================================================
    # ส่วนที่ 3 — สร้างชุดฟีเจอร์ทุกแบบ
    # ================================================================
    common.banner("สร้างชุดฟีเจอร์ตามวิธีรวมต่าง ๆ")

    sets = {"all_features": list(X.columns)}    # baseline: ใช้ครบ = เพดาน
    sets.update(merge_strategies(imp))

    print("\n  กำลังหาฟีเจอร์จากโมเดล binary โดยตรง (ตัวควบคุม)...")
    sets["shap_from_binary"] = shap_from_binary_model(df_all)

    print()
    for name, feats in sets.items():
        print(f"  {name:22s} {len(feats):>3} ฟีเจอร์")

    # ---- เซฟเป็น JSON ให้ 04 และ 05 อ่านไปใช้ ----
    # JSON คือ single source of truth — ถ้าเปลี่ยนวิธีเลือกฟีเจอร์ แก้ที่นี่ที่เดียว
    # แล้วสคริปต์ปลายน้ำได้ชุดใหม่ทันทีโดยไม่ต้องแก้อะไร
    config.FEATURE_SETS_JSON.write_text(
        json.dumps(sets, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[saved] {config.FEATURE_SETS_JSON}")

    # ---- ตารางสรุปว่าแต่ละชุดมีฟีเจอร์อะไรบ้าง ----
    overlap = pd.DataFrame(
        {name: [f in feats for f in X.columns] for name, feats in sets.items()},
        index=X.columns,
    )
    common.save_table(overlap.astype(int), "03_feature_set_membership.csv")

    # ================================================================
    # ส่วนที่ 4 (ผลรอง) — ลดฟีเจอร์แล้ว stage 2 เองยังทำงานได้ไหม
    # ================================================================
    # ไม่ใช่ประเด็นหลักของงาน (ประเด็นหลักคือเอาฟีเจอร์ไปให้ stage 1 ใน 04)
    # แต่ตอบคำถามที่มีค่าว่า "stage 2 ก็ลดฟีเจอร์ได้ด้วยไหม" จึงเก็บไว้
    common.banner("ผลรอง: เทรน stage 2 ซ้ำด้วยแต่ละชุดฟีเจอร์")

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )

    rows = {}
    for name, feats in sets.items():
        if not feats:
            continue
        m = XGBClassifier(
            n_estimators=config.N_ESTIMATORS, random_state=config.SEED,
            n_jobs=-1, tree_method="hist", objective="multi:softprob",
        )
        m.fit(X_train[feats], y_train)
        r = common.evaluate(m, X_test[feats], y_test, average="macro")
        r["n_features"] = len(feats)
        rows[name] = r
        print(f"  {name:22s} {len(feats):>3} feat | macro-F1 {r['f1']:.4f}")

    table = pd.DataFrame(rows).T[["n_features", "f1", "precision", "recall", "latency_ms_per_1k"]]
    table["f1_เทียบ_full_%"] = table["f1"] / table.loc["all_features", "f1"] * 100
    table = table.sort_values("n_features")
    print()
    print(table.to_string())
    common.save_table(table, "03_stage2_with_reduced_features.csv")


if __name__ == "__main__":
    main()
