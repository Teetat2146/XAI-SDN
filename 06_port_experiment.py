"""
06 — พิสูจน์ด้วยการทดลองว่าควรตัด Src Port / Dst Port ออกหรือไม่

ที่มา (ประชุม 2026-08-08):
    อาจารย์ท้วงว่าเราตัด Port ออกโดยอ้างเหตุผลอย่างเดียว ยังไม่เคยพิสูจน์
    "ประเด็นคือเราได้ลองบอกไหม บอกจริง ๆ ไหมว่ามันผิดหรือเปล่า"
    "ถ้าไม่ลองมันจะเหมือนว่าเอ้าถาดขึ้นมาก็รู้ได้ไงว่าถ้าไม่ตัดมันจะผิด"
    → ต้องเป็น by experiment ไม่ใช่ by argument

สมมติฐาน:
    Port เป็น "shortcut" ที่ผูกกับ testbed (BFA ยิง port 22, Web-Attack ยิง port 80)
    ถ้าจริง จะเห็นรูปแบบนี้:
        - split แบบสุ่ม (train/test จาก capture เดียวกัน) → ใส่ Port แล้ว **ดีขึ้น**
        - ข้ามสภาพแวดล้อม (train OVS → test metasploitable) → ใส่ Port แล้ว **แย่ลง**
    เพราะโมเดลจำ mapping port→attack ของ testbed หนึ่ง แล้วเอาไปใช้กับอีก testbed ไม่ได้

    ถ้าผลออกมาว่าใส่ Port แล้วดีขึ้น **ทั้งสองแบบ** = เราตัดทิ้งไปโดยไม่จำเป็น
    ต้องเอากลับเข้ามา และรายงานตามจริง

การทดลอง 3 ส่วน:
    A. stage 2 (จำแนกชนิด attack) — Port ติดอันดับความสำคัญแค่ไหน
    B. stage 1 (ตัวกรอง) — split แบบสุ่ม
    C. stage 1 (ตัวกรอง) — ข้ามสภาพแวดล้อม  ← ตัวชี้ขาด

รัน:  python 06_port_experiment.py   (ต้องรัน 01 ก่อน)
"""
from importlib import import_module

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import common
import config

# ยืมฟังก์ชันเลือก threshold / วัดผล จาก 04 มาใช้ ไม่ก๊อปโค้ด
m4 = import_module("04_train_stage1")

VARIANTS = {"ไม่มี Port (ปัจจุบัน)": False, "มี Port": True}


def stage2_experiment(df):
    """A. Port สำคัญแค่ไหนในสายตาของ SHAP + กระทบ macro-F1 ไหม"""
    common.banner("A. stage 2 — Port ติดอันดับความสำคัญแค่ไหน")

    data = df[df["binary_label"] == 1].reset_index(drop=True)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(data["attack_class"]))

    rows = {}
    for label, use_ports in VARIANTS.items():
        X = data[common.feature_columns(data, ports=use_ports)]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
        )
        m = XGBClassifier(n_estimators=config.N_ESTIMATORS, random_state=config.SEED,
                          n_jobs=-1, tree_method="hist", objective="multi:softprob")
        m.fit(X_tr, y_tr)
        r = common.evaluate(m, X_te, y_te, average="macro")
        r["n_features"] = X.shape[1]
        rows[label] = r

        if use_ports:
            # ดูว่า Port ติดอันดับที่เท่าไหร่จาก 67 ฟีเจอร์ ในแต่ละคลาส
            n = min(config.SHAP_SAMPLE, len(X_tr))
            X_shap = X_tr.sample(n=n, random_state=config.SEED)
            sv = common.to_shap_array(shap.TreeExplainer(m).shap_values(X_shap))
            imp = common.mean_abs_shap(sv, list(X_shap.columns), list(le.classes_))

            rank = imp.rank(ascending=False).astype(int)
            print("\n  อันดับความสำคัญของ Port ในแต่ละคลาส (จาก 67 ฟีเจอร์):")
            print(rank.loc[config.PORT_COLS].to_string())
            print("\n  ค่าความสำคัญ (normalize แล้ว):")
            print(imp.loc[config.PORT_COLS].round(4).to_string())
            common.save_table(imp.loc[config.PORT_COLS], "06_port_shap_importance.csv")

    t = pd.DataFrame(rows).T[["n_features", "f1", "precision", "recall"]]
    print()
    print(t.to_string())
    common.save_table(t, "06_stage2_port_effect.csv")
    return t


def stage1_random_split(df):
    """B. stage 1 แบบ split สุ่ม — train/test มาจาก capture เดียวกัน"""
    common.banner("B. stage 1 — split แบบสุ่ม (train/test จาก capture เดียวกัน)")

    rows = []
    for label, use_ports in VARIANTS.items():
        feats = common.feature_columns(df, ports=use_ports)
        X, y = df[feats], df["binary_label"]

        X_tmp, X_te, y_tmp, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.2, random_state=config.SEED, stratify=y_tmp
        )

        # อาจารย์ระบุว่า stage 1 ควรเป็น DecisionTree / RandomForest ไม่ใช่ XGBoost
        for name, model in common.build_models(n_classes=2).items():
            model.fit(X_tr, y_tr)
            thr = m4.pick_threshold(model.predict_proba(X_val)[:, 1], y_val,
                                    config.TARGET_RECALL)
            r = m4.eval_at_threshold(model, X_te, y_te, thr)
            r.update(variant=label, model=name, n_features=len(feats))
            rows.append(r)

    t = pd.DataFrame(rows)
    piv = t.pivot_table(index="model", columns="variant",
                        values=["recall", "fp_rate"])
    print(piv.to_string())
    common.save_table(t[["variant", "model", "n_features", "recall", "fp_rate",
                         "latency_ms_per_1k"]], "06_stage1_random_split.csv", index=False)
    return t


def stage1_cross_env(df):
    """C. stage 1 ข้ามสภาพแวดล้อม — ตัวชี้ขาด

    train: Normal(ครึ่ง) + OVS ทั้งหมด
    test : Normal(อีกครึ่ง) + metasploitable ทั้งหมด
    IP/port/สภาพแวดล้อมคนละชุด → ถ้า Port เป็น shortcut จะเห็นผลตรงนี้
    """
    common.banner("C. stage 1 — ข้ามสภาพแวดล้อม (train OVS → test metasploitable)")

    normal = df[df["source"] == "Normal"]
    n_tr, n_te = train_test_split(normal, test_size=0.5, random_state=config.SEED)
    train = pd.concat([n_tr, df[df["source"] == "OVS"]], ignore_index=True)
    test = pd.concat([n_te, df[df["source"] == "metasploitable"]], ignore_index=True)

    rows, per_class = [], []
    for label, use_ports in VARIANTS.items():
        feats = common.feature_columns(df, ports=use_ports)

        for name, model in common.build_models(n_classes=2).items():
            model.fit(train[feats], train["binary_label"])
            r = common.evaluate(model, test[feats], test["binary_label"])
            r.update(variant=label, model=name, n_features=len(feats))
            rows.append(r)

            if name == "DecisionTree":     # โมเดลที่อาจารย์อยากใช้เป็น stage 1
                t2 = test.copy()
                t2["pred"] = model.predict(test[feats])
                for cls, grp in t2.groupby("attack_class"):
                    per_class.append({"variant": label, "attack_class": cls,
                                      "n": len(grp),
                                      "recall": (grp["pred"] == 1).mean()
                                      if cls != config.NORMAL_LABEL
                                      else 1 - (grp["pred"] == 1).mean()})

    t = pd.DataFrame(rows)
    print(t.pivot_table(index="model", columns="variant",
                        values=["recall", "precision"]).to_string())
    common.save_table(t[["variant", "model", "n_features", "recall", "precision",
                         "f1"]], "06_stage1_cross_env.csv", index=False)

    pc = pd.DataFrame(per_class).pivot_table(
        index=["attack_class", "n"], columns="variant", values="recall")
    print("\n  รายคลาส (DecisionTree) — แถว Normal คือ 1-FP rate:")
    print(pc.round(4).to_string())
    common.save_table(pc, "06_cross_env_per_class.csv")
    return t


def main():
    df = common.load_clean()
    print(f"ฟีเจอร์: ไม่มี Port = {len(common.feature_columns(df))} ตัว "
          f"| มี Port = {len(common.feature_columns(df, ports=True))} ตัว")

    stage2_experiment(df)
    rnd = stage1_random_split(df)
    crs = stage1_cross_env(df)

    # ================================================================
    # สรุป — ตัดสินสมมติฐาน
    # ================================================================
    common.banner("สรุป — ควรตัด Port ออกหรือไม่")

    def delta(t, metric):
        """ผลของการใส่ Port = (มี Port) - (ไม่มี Port) เฉลี่ยข้ามโมเดล"""
        p = t.pivot_table(index="model", columns="variant", values=metric)
        return (p["มี Port"] - p["ไม่มี Port (ปัจจุบัน)"]).mean()

    d_rnd = delta(rnd, "recall")
    d_crs = delta(crs, "recall")

    print(f"  ผลของการใส่ Port ต่อ recall (เฉลี่ย 3 โมเดล):")
    print(f"    split แบบสุ่ม      {d_rnd:+.6f}")
    print(f"    ข้ามสภาพแวดล้อม   {d_crs:+.6f}")
    print()

    if d_rnd > 0 and d_crs < 0:
        print("  → ยืนยันสมมติฐาน: Port ช่วยตอน split สุ่ม แต่ทำร้ายตอนข้ามสภาพแวดล้อม")
        print("    = เป็น shortcut ที่ผูกกับ testbed **การตัดออกถูกต้องแล้ว**")
    elif d_rnd > 0 and d_crs > 0:
        print("  → ขัดกับสมมติฐาน: Port ช่วยทั้งสองแบบ")
        print("    = เราตัดทิ้งไปโดยไม่จำเป็น ควรพิจารณาเอากลับเข้ามา")
    elif d_rnd <= 0:
        print("  → Port ไม่ได้ช่วยแม้แต่ตอน split สุ่ม = ตัดทิ้งได้สบายใจ")
    else:
        print("  → ผลไม่ชัด ต้องดูรายละเอียดรายคลาสประกอบ")

    print("\n  หมายเหตุ: ดูตารางรายคลาสในส่วน C ด้วย")
    print("  ถ้าคลาสที่ผูกกับ port ชัด (BFA→22, Web-Attack→80) ร่วงแรงตอนข้ามสภาพแวดล้อม")
    print("  นั่นคือหลักฐานตรงที่สุดว่าโมเดลจำ port แทนที่จะเรียนพฤติกรรม")


if __name__ == "__main__":
    main()
