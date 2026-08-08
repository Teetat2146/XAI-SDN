"""
02 — stage 2 (multi-class): ทดลองว่าควรตัดฟีเจอร์กลุ่มไหนบ้าง

ที่มา (ประชุม 2026-08-08):
    อาจารย์ท้วงว่าเราตัดฟีเจอร์โดยอ้างเหตุผล ยังไม่เคยพิสูจน์
    "รู้ได้ไงว่าถ้าไม่ตัดมันจะผิด" → ต้องเป็น by experiment

การทดลอง: 7 ระดับการตัด × 3 แบบการแบ่งข้อมูล × 3 โมเดล = 63 รอบ

หัวใจอยู่ที่ **การเทียบระหว่าง split**:
    random อย่างเดียวจับ leakage ไม่ได้ — โมเดลที่จำ IP ได้ก็ยังทายถูก
    เพราะ IP เดียวกันอยู่ทั้ง train และ test
    ต้องดู ovs2meta / meta2ovs ซึ่ง IP เป็นคนละชุด ความจำใช้ไม่ได้

    ช่องว่าง (random − cross-env) = ขนาดของการ "จำ" แทนที่จะ "เรียน"

รัน:  python 02_train_stage2.py   (ต้องรัน 01 ก่อน)
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import common
import config


def run_one(df, level, split_kind, feature_sets=None):
    """เทรน 3 โมเดลที่ระดับ/split หนึ่ง — คืน (rows, per_class_rows)

    per_class สำคัญมาก: macro-F1 รวมบอกแค่ว่า "ประมาณ 0.45" แต่ไม่บอกว่า
    คลาสไหนพัง ซึ่งเป็นข้อมูลที่ตีความผลได้จริง (เช่น DoS ข้ามสภาพแวดล้อมไม่ได้)
    """
    # shap_top15 อ่านรายชื่อจาก JSON ที่ 03 สร้าง (ถ้ายังไม่มีก็ข้าม)
    if level == "shap_top15":
        if not feature_sets:
            return [], []
        feats = feature_sets
    else:
        feats = common.feature_columns(df, level=level)

    train, test = common.make_split(df, split_kind)

    # encode label ให้เป็นเลข — fit บน train+test เพื่อให้ mapping ตรงกัน
    le = LabelEncoder().fit(pd.concat([train["attack_class"], test["attack_class"]]))
    y_tr = le.transform(train["attack_class"])
    y_te = le.transform(test["attack_class"])

    rows, pc_rows = [], []
    for name, model in common.build_models(n_classes=len(le.classes_)).items():
        model.fit(train[feats], y_tr)
        r = common.evaluate(model, test[feats], y_te, average="macro")

        # เก็บ F1/recall รายคลาสไว้ด้วย (เฉพาะ XGBoost พอ ไม่งั้นตารางบวม)
        if name == "XGBoost":
            rep = classification_report(y_te, model.predict(test[feats]),
                                        target_names=le.classes_,
                                        output_dict=True, zero_division=0)
            for cls in le.classes_:
                pc_rows.append({"level": level, "split": split_kind, "attack_class": cls,
                                "support": int(rep[cls]["support"]),
                                "f1": rep[cls]["f1-score"],
                                "recall": rep[cls]["recall"]})
        r.update(level=level, split=split_kind, model=name,
                 n_features=len(feats), n_classes=len(le.classes_),
                 n_train=len(train), n_test=len(test))
        rows.append(r)
        print(f"    {name:14s} macro-F1 {r['f1']:.4f} | acc {r['accuracy']:.4f} "
              f"| {r['latency_ms_per_1k']:.2f} ms/1k")

        # เก็บโมเดลไว้ให้ 03 ใช้ดึง SHAP (เฉพาะ split random)
        if split_kind == "random":
            joblib.dump(model, config.MODEL_DIR / f"stage2_{level}_{name}.pkl")
            if name == "XGBoost":
                joblib.dump(le, config.MODEL_DIR / f"label_encoder_{level}.pkl")

    return rows, pc_rows


def main():
    df = common.load_clean()

    # shap_top15 ต้องรอ 03 สร้างก่อน — รอบแรกจะยังไม่มี
    shap_feats = None
    if config.FEATURE_SETS_JSON.exists():
        sets = json.loads(config.FEATURE_SETS_JSON.read_text(encoding="utf-8"))
        shap_feats = sets.get(f"mean")  # ชุด global-mean top-K

    all_rows, all_pc = [], []
    for level in config.FEATURE_LEVELS:
        for split_kind in config.SPLITS:
            n_feat = ("?" if level == "shap_top15"
                      else len(common.feature_columns(df, level=level)))
            common.banner(f"[{level}] ({n_feat} ฟีเจอร์) — split: {split_kind}")
            rows, pc = run_one(df, level, split_kind, shap_feats)
            if not rows:
                print("    (ข้าม — ยังไม่มีชุดฟีเจอร์จาก 03)")
            all_rows += rows
            all_pc += pc

    grid = pd.DataFrame(all_rows)[
        ["level", "split", "model", "n_features", "n_classes",
         "f1", "accuracy", "precision", "recall", "latency_ms_per_1k"]
    ]
    common.save_table(grid, "02_stage2_grid.csv", index=False)

    # ================================================================
    # ตารางหลัก — generalization gap
    # ================================================================
    # ตรึงโมเดลไว้ที่ XGBoost เพื่อให้ตัวแปรที่ต่างมีแค่ระดับการตัดฟีเจอร์
    common.banner("ตารางหลัก — macro-F1 แต่ละระดับ × แต่ละ split (XGBoost)")

    x = grid[grid.model == "XGBoost"]
    piv = x.pivot_table(index="level", columns="split", values="f1")
    piv = piv.reindex([lv for lv in config.FEATURE_LEVELS if lv in piv.index])
    piv.insert(0, "n_features",
               x.groupby("level")["n_features"].first().reindex(piv.index))

    # ช่องว่าง = random − ค่าเฉลี่ยของ cross-environment
    cross = piv[["ovs2meta", "meta2ovs"]].mean(axis=1)
    piv["gap"] = piv["random"] - cross

    print(piv.round(4).to_string())
    common.save_table(piv, "02_generalization_gap.csv")

    print("\nอ่านตารางนี้ยังไง:")
    print("  random    = ทำข้อสอบชุดเดิม (train/test จาก capture เดียวกัน)")
    print("  ovs2meta  = ทำข้อสอบชุดใหม่ (คนละ testbed)")
    print("  gap       = ช่องว่าง = ขนาดของการ 'จำ' แทนที่จะ 'เรียน'")
    print("\n  ถ้า raw มี gap สูงกว่าระดับอื่นชัดเจน = ยืนยันว่าการตัด identifier ถูกต้อง")
    print("  ถ้า gap ใกล้กันหมด = ตัดไปโดยไม่จำเป็น ต้องรายงานตามจริง")

    # ================================================================
    # ตารางรอง — แยกตามโมเดล
    # ================================================================
    common.banner("แยกตามโมเดล (macro-F1)")
    by_model = grid.pivot_table(index=["level", "split"], columns="model", values="f1")
    print(by_model.round(4).to_string())
    common.save_table(by_model, "02_stage2_by_model.csv")

    # ================================================================
    # per-class — คลาสไหนพังตอนข้ามสภาพแวดล้อม
    # ================================================================
    if all_pc:
        pc = pd.DataFrame(all_pc)
        common.save_table(pc, "02_per_class.csv", index=False)

        for split_kind in config.SPLITS:
            sub = pc[pc.split == split_kind]
            if sub.empty:
                continue
            common.banner(f"F1 รายคลาส — split {split_kind} (XGBoost)")
            t = sub.pivot_table(index="attack_class", columns="level", values="f1")
            t = t[[lv for lv in config.FEATURE_LEVELS if lv in t.columns]]
            t.insert(0, "support", sub.groupby("attack_class")["support"].first())
            print(t.round(4).to_string())
            common.save_table(t, f"02_per_class_{split_kind}.csv")

        print("\n  อ่านยังไง: แถวที่ F1 ต่ำคือคลาสที่ทำให้ macro-F1 รวมตกลง")
        print("  ถ้าคลาสเดียวพังแล้วลาก macro ลง ต้องระบุในรายงาน ไม่ใช่สรุปว่าโมเดลแย่ทั้งหมด")

    common.banner("latency (ms ต่อ 1000 flows) — split random")
    lat = grid[grid.split == "random"].pivot_table(
        index="level", columns="model", values="latency_ms_per_1k")
    lat = lat.reindex([lv for lv in config.FEATURE_LEVELS if lv in lat.index])
    print(lat.round(3).to_string())
    common.save_table(lat, "02_stage2_latency.csv")


if __name__ == "__main__":
    main()
