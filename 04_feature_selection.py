"""
04 — หัวใจของงาน: SHAP ต่อคลาส + เทียบ 4 วิธีรวมฟีเจอร์

ปัญหาที่อาจารย์ตั้งไว้ในที่ประชุม:
    เมื่อรัน SHAP แยกต่อคลาส แต่ละคลาสจะได้ฟีเจอร์สำคัญไม่เหมือนกัน
    เช่น DDoS ให้ความสำคัญกับฟีเจอร์ชุด A แต่ BFA ให้ความสำคัญกับชุด B
    แล้วสุดท้ายจะเอาฟีเจอร์ชุดไหนไปใช้จริง?
      - รวมทุกคลาส (union) → ฟีเจอร์บวมเป็นหลักสิบ ไม่ต่างจากไม่กรองเลย
      - เลือกเฉพาะที่ทุกคลาสเห็นตรงกัน (intersection) → อาจเหลือน้อยเกินไป
      - ใช้ชุดเดียวกันทุกคลาส → อาจารย์ให้ลองว่าผลจะเปลี่ยนไหม

วิธีตอบของสคริปต์นี้:
    ไม่ฟันธงวิธีเดียว แต่ทำทุกวิธีแล้ววัดด้วยแกนเดียวกัน 3 คอลัมน์
    (จำนวนฟีเจอร์ / Macro-F1 / latency)
    วิธีที่ได้ F1 ใกล้ full-feature ที่สุดด้วยฟีเจอร์น้อยที่สุด = คำตอบ

รัน:  python 04_feature_selection.py   (ต้องรัน 01 และ 03 ก่อน)
"""
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import common
import config


def strategies(imp: pd.DataFrame) -> dict[str, list[str]]:
    """สร้างชุดฟีเจอร์ตาม 4 วิธี + ชุดเต็มไว้เป็น baseline

    imp: ตารางความสำคัญจาก common.mean_abs_shap
         index = ชื่อฟีเจอร์, columns = ชื่อคลาส, ค่า = ความสำคัญ (normalize แล้ว)
    คืน: dict {ชื่อวิธี: [รายชื่อฟีเจอร์]}
    """
    k = config.TOPK

    # หา Top-K ของแต่ละคลาสก่อน — เก็บเป็น set เพื่อให้ทำ union/intersection ได้ง่าย
    # nlargest(k) = เอา k ตัวที่ค่าสูงสุด
    top_per_class = {c: set(imp[c].nlargest(k).index) for c in imp.columns}

    # --- วิธีที่ 1: Union ---
    # เอา Top-K ของทุกคลาสมารวมกัน ฟีเจอร์ตัวไหนติด Top-K ของคลาสใดคลาสหนึ่งก็เอาหมด
    # ข้อดี: ไม่พลาดฟีเจอร์ที่สำคัญเฉพาะบางคลาส
    # ข้อเสีย: ฟีเจอร์บวม และฟีเจอร์ที่สำคัญเฉพาะคลาสเล็กอาจเป็น noise สำหรับคลาสใหญ่
    union = set().union(*top_per_class.values())

    # --- วิธีที่ 2: Intersection ---
    # เอาเฉพาะฟีเจอร์ที่ติด Top-K ของ "ทุกคลาส" พร้อมกัน
    # ข้อดี: ได้ฟีเจอร์น้อยมาก และเป็นตัวที่สำคัญกับทุกคลาสจริง ๆ
    # ข้อเสีย: มักเหลือน้อยเกินไปจนโมเดลแยกคลาสไม่ออก
    inter = set(imp.index).intersection(*top_per_class.values())

    # --- วิธีที่ 3: Global mean ---
    # เฉลี่ยความสำคัญข้ามทุกคลาสก่อน แล้วค่อยตัด Top-K ครั้งเดียว
    # นี่คือสมมติฐาน "ใช้ฟีเจอร์เดียวกันทุกคลาส" ที่อาจารย์ให้ลองพอดี
    # mean(axis=1) = เฉลี่ยตามแนวนอน (เฉลี่ยข้ามคลาส)
    global_top = set(imp.mean(axis=1).nlargest(k).index)

    # --- วิธีที่ 4: Dynamic-K ---
    # แทนที่จะ fix ว่าทุกคลาสเอา 15 ตัวเท่ากัน ให้แต่ละคลาสเก็บฟีเจอร์
    # ไปเรื่อย ๆ จนความสำคัญสะสมถึงเกณฑ์ (90%) แล้วหยุด
    # → คลาสที่ทายง่ายจะใช้ฟีเจอร์น้อย คลาสที่ซับซ้อนจะใช้เยอะ
    # ตรงกับที่อาจารย์เสนอว่า "ไม่ fix ตายตัวว่าต้อง 5 หรือ 10 อันแรก"
    dynamic = set()
    dyn_k = {}
    for c in imp.columns:
        s = imp[c].sort_values(ascending=False)     # เรียงจากสำคัญมาก → น้อย
        # cumsum() = ผลรวมสะสม เช่น [0.3, 0.5, 0.65, ...]
        # นับว่ามีกี่ตัวที่ผลรวมสะสมยังไม่ถึง threshold แล้ว +1 เพื่อรวมตัวที่ทำให้ถึงด้วย
        n = int((s.cumsum() < config.CUMULATIVE_THRESHOLD).sum()) + 1
        dyn_k[c] = n
        dynamic |= set(s.index[:n])                # |= คือ union เข้ากับ set เดิม

    print(f"\n  dynamic-K เลือกได้กี่ฟีเจอร์ต่อคลาส (threshold {config.CUMULATIVE_THRESHOLD}):")
    for c, n in dyn_k.items():
        print(f"    {str(c):16s} K = {n}")

    return {
        "all_features": list(imp.index),           # baseline: ใช้ทุกฟีเจอร์ = เพดาน
        f"union_top{k}": sorted(union),
        f"intersection_top{k}": sorted(inter),
        f"global_mean_top{k}": sorted(global_top),
        "dynamic_k": sorted(dynamic),
    }


def main():
    # ---- เตรียมข้อมูลให้เหมือนกับสคริปต์ 03 เป๊ะ ----
    # ต้องใช้ seed และ test_size เดียวกัน ไม่งั้น test set จะคนละชุด เทียบผลไม่ได้
    df = common.load_clean()
    df = df[df["binary_label"] == 1].reset_index(drop=True)

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df["attack_class"]))
    X = df[common.feature_columns(df)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )

    # โหลดโมเดลที่ 03 เทรนไว้ ไม่ต้องเทรนใหม่
    model_path = config.MODEL_DIR / "multi_best.pkl"
    if not model_path.exists():
        raise SystemExit("ยังไม่มีโมเดล multi-class — รัน `python 03_train_multiclass.py` ก่อน")
    model = joblib.load(model_path)

    # ================================================================
    # ส่วนที่ 1 — คำนวณ SHAP
    # ================================================================
    common.banner("คำนวณ SHAP (TreeExplainer)")

    # ใช้ train set เท่านั้น! ถ้าคำนวณ SHAP บน test set แล้วเอาไปเลือกฟีเจอร์
    # เท่ากับเราแอบดูคำตอบของ test set มาช่วยตัดสินใจ = leakage
    #
    # subsample เพราะ SHAP ช้าและกินแรม: ต้องเก็บ array ขนาด
    # (แถว × ฟีเจอร์ × คลาส) = 220,000 × 65 × 7 ซึ่งใหญ่เกินจำเป็น
    # 5,000 แถวก็พอให้ค่าเฉลี่ยนิ่งแล้ว (เป็นวิธีมาตรฐาน ต้องเขียนระบุใน methodology)
    n = min(config.SHAP_SAMPLE, len(X_train))
    X_shap = X_train.sample(n=n, random_state=config.SEED)
    print(f"  subsample {n:,} แถวจาก train set ({len(X_train):,} แถว)")

    # TreeExplainer = อัลกอริทึม SHAP เวอร์ชันเฉพาะสำหรับโมเดลต้นไม้
    # เร็วกว่า KernelExplainer (เวอร์ชันทั่วไป) หลายพันเท่า เพราะใช้โครงสร้างต้นไม้ช่วยคำนวณ
    explainer = shap.TreeExplainer(model)
    sv = common.to_shap_array(explainer.shap_values(X_shap))
    print(f"  shap values shape = {sv.shape}  (samples, features, classes)")

    # sv คือค่า SHAP ดิบ: บอกว่าแต่ละฟีเจอร์ในแต่ละแถว ดันผลลัพธ์ไปทางไหนเท่าไหร่
    # mean_abs_shap ยุบให้เหลือ "ความสำคัญเฉลี่ย" ต่อฟีเจอร์ต่อคลาส (+ normalize)
    imp = common.mean_abs_shap(sv, list(X_shap.columns), list(le.classes_))
    common.save_table(imp, "04_shap_importance_per_class.csv")

    # แสดง Top 10 ของแต่ละคลาส — ตรงนี้จะเห็นด้วยตาว่าแต่ละคลาสได้ฟีเจอร์ไม่เหมือนกันจริง
    common.banner("Top 10 ฟีเจอร์ของแต่ละคลาส")
    for c in imp.columns:
        top = imp[c].nlargest(10)
        print(f"\n  [{c}]")
        for f, v in top.items():
            print(f"    {v:.4f}  {f}")

    # ================================================================
    # ส่วนที่ 2 — เทรนใหม่ด้วยฟีเจอร์แต่ละชุด แล้วเทียบกัน
    # ================================================================
    common.banner("เทียบ 4 วิธีรวมฟีเจอร์")
    sets = strategies(imp)

    rows = {}
    for name, feats in sets.items():
        if not feats:
            print(f"\n--- {name}: ว่างเปล่า ข้าม ---")
            continue

        print(f"\n--- {name} ({len(feats)} ฟีเจอร์) ---")

        # ใช้ XGBoost อย่างเดียวในการเทียบ (ไม่เทียบ 3 โมเดลซ้ำ)
        # เพราะตัวแปรที่เราอยากวัดคือ "ชุดฟีเจอร์" ไม่ใช่ "ชนิดโมเดล"
        # ถ้าเปลี่ยนทั้งสองอย่างพร้อมกันจะแยกไม่ออกว่าอะไรทำให้ผลต่าง
        m = XGBClassifier(
            n_estimators=config.N_ESTIMATORS, random_state=config.SEED,
            n_jobs=-1, tree_method="hist", objective="multi:softprob",
        )
        # X_train[feats] = เอาเฉพาะคอลัมน์ในชุดฟีเจอร์นี้
        m.fit(X_train[feats], y_train)

        r = common.evaluate(m, X_test[feats], y_test, average="macro")
        r["n_features"] = len(feats)
        rows[name] = r
        print(f"  macro-F1 {r['f1']:.4f} | latency {r['latency_ms_per_1k']:.2f} ms/1k")

    # ---- ประกอบเป็นตารางสรุป ----
    table = pd.DataFrame(rows).T[
        ["n_features", "f1", "precision", "recall", "accuracy", "latency_ms_per_1k"]
    ]
    # เทียบทุกวิธีกับ baseline (ใช้ฟีเจอร์ครบ) เป็น % จะอ่านง่ายกว่าดูค่า F1 ดิบ
    baseline = table.loc["all_features", "f1"]
    table["f1_เทียบ_full_%"] = table["f1"] / baseline * 100
    table = table.sort_values("n_features")

    common.banner("ตารางสรุป — เอาไปใส่รายงานได้เลย")
    print(table.to_string())
    common.save_table(table, "04_feature_selection_comparison.csv")

    print("\nอ่านตารางนี้ยังไง:")
    print("  หาแถวที่ f1_เทียบ_full_% ใกล้ 100 ที่สุด โดยที่ n_features ต่ำที่สุด")
    print("  นั่นคือวิธีรวมฟีเจอร์ที่คุ้มที่สุด = ข้อสรุปหลักของงาน")

    # เก็บรายชื่อฟีเจอร์ที่แต่ละวิธีเลือกไว้ด้วย เผื่อต้องอ้างอิงในรายงาน
    pd.Series({k: ", ".join(v) for k, v in sets.items()}).to_csv(
        config.OUT_DIR / "04_selected_features.csv"
    )
    print(f"[saved] {config.OUT_DIR / '04_selected_features.csv'}")


if __name__ == "__main__":
    main()
