"""
04 — โมเดลชั้นที่ 1 (stage 1): ตัวกรอง Normal vs Attack

ตำแหน่งของไฟล์นี้ใน flow ของงาน:
    01 เตรียมข้อมูล → 02 เทรน stage 2 → 03 SHAP หาชุดฟีเจอร์ → [04 ตัวนี้]

นี่คือจุดที่แนวคิดของอาจารย์ถูกทดสอบจริง:
    เอาฟีเจอร์ที่คัดมาจาก SHAP ของ stage 2 มาเทรน stage 1
    ถ้าใช้ฟีเจอร์น้อยลงแล้ว stage 1 ยังจับ attack ได้ครบ แต่เร็วขึ้น
    = พิสูจน์ว่าแนวคิดนี้ใช้ได้จริง

ทำไม stage 1 ต้องวัดต่างจาก stage 2:
    stage 1 เป็นแค่ "ตัวกรอง" ไม่ได้ฟันธงว่าเป็น attack ชนิดไหน
    หน้าที่คือ "ห้ามพลาด attack" (recall สูง) ส่วน false positive ยอมได้
    เพราะยังมี stage 2 คัดกรองต่ออีกชั้น

    ดังนั้นจึงไม่วัดที่ threshold 0.5 ตามค่าเริ่มต้น แต่:
      1. ตรึง recall ไว้ที่ TARGET_RECALL (99%) เท่ากันทุกชุดฟีเจอร์
      2. แล้วค่อยเทียบว่าชุดไหนให้ FP ต่ำสุด และเร็วสุด
    ถ้าไม่ตรึง recall จะเทียบกันไม่ยุติธรรม เพราะแต่ละชุดฟีเจอร์ทำให้โมเดล
    "เข้มงวด" ไม่เท่ากันโดยธรรมชาติ

รัน:  python 04_train_stage1.py   (ต้องรัน 01, 02, 03 ก่อน)
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import common
import config


def pick_threshold(proba, y_true, target_recall):
    """หา threshold ที่สูงที่สุด ที่ยังได้ recall ตามเป้า

    ทำไมเอา "สูงที่สุด": threshold ยิ่งสูง = ยิ่งเข้มงวด = false positive ยิ่งน้อย
    เราจึงอยากได้ตัวที่เข้มงวดที่สุดเท่าที่ยังไม่พลาด attack เกินเกณฑ์

    หมายเหตุ: เลือกจาก validation set เท่านั้น ห้ามใช้ test set
    ไม่งั้นเท่ากับปรับจูนโดยแอบดูคำตอบ = leakage
    """
    n_pos = int((y_true == 1).sum())
    best = 0.0
    # ไล่จากเข้มงวดมาก → เข้มงวดน้อย แล้วหยุดเมื่อ recall ถึงเป้าครั้งแรก
    for thr in np.linspace(0.99, 0.0, 200):
        recall = int(((y_true == 1) & (proba >= thr)).sum()) / n_pos
        if recall >= target_recall:
            best = thr
            break
    return best


def eval_at_threshold(model, X_test, y_test, thr):
    """วัดผล stage 1 ที่ threshold ที่กำหนด"""
    # จับเวลา predict_proba ไม่ใช่ predict เพราะระบบจริงต้องได้ความน่าจะเป็น
    # มาเทียบกับ threshold ที่เราตั้งเอง
    t0 = time.perf_counter()
    proba = model.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - t0

    flagged = proba >= thr
    is_attack = y_test == 1

    tp = int((is_attack & flagged).sum())
    fp = int((~is_attack & flagged).sum())

    recall = tp / int(is_attack.sum())
    precision = tp / max(tp + fp, 1)

    return {
        "threshold": thr,
        "recall": recall,
        "fp_rate": fp / int((~is_attack).sum()),
        "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        # สัดส่วน traffic ที่ต้องส่งต่อให้ stage 2 — ยิ่งน้อยยิ่งดี
        "ส่งต่อ_stage2_%": float(flagged.mean() * 100),
        "latency_ms_per_1k": elapsed / len(X_test) * 1e6,
    }


def main():
    # ---- โหลดชุดฟีเจอร์ที่ 03 ผลิตไว้ ----
    if not config.FEATURE_SETS_JSON.exists():
        raise SystemExit("ยังไม่มีชุดฟีเจอร์ — รัน `python 03_shap_features.py` ก่อน")
    sets = json.loads(config.FEATURE_SETS_JSON.read_text(encoding="utf-8"))

    df = common.load_clean()
    X, y = common.split_xy(df, "binary_label")

    # ---- แบ่ง 3 ส่วน: train / val / test ----
    # ต้องมี val แยกต่างหากเพราะต้องใช้เลือก threshold
    # ถ้าเลือก threshold จาก test set = ปรับจูนโดยดูคำตอบ ผลจะดูดีเกินจริง
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2, random_state=config.SEED, stratify=y_tmp
    )

    common.banner("Stage 1 — ตัวกรอง Normal vs Attack")
    print(f"  train {len(X_train):,} | val {len(X_val):,} | test {len(X_test):,}")
    print(f"  เกณฑ์: ตรึง recall ≥ {config.TARGET_RECALL:.0%} แล้วเทียบ FP กับ latency")
    print(f"  ชุดฟีเจอร์ที่จะทดสอบ: {len(sets)} ชุด × 3 โมเดล")

    # ---- วนทุกชุดฟีเจอร์ × ทุกโมเดล ----
    rows = []
    for set_name, feats in sets.items():
        if not feats:
            continue
        print(f"\n--- {set_name} ({len(feats)} ฟีเจอร์) ---")

        for model_name, model in common.build_models(n_classes=2).items():
            model.fit(X_train[feats], y_train)

            # เลือก threshold จาก val set
            proba_val = model.predict_proba(X_val[feats])[:, 1]
            thr = pick_threshold(proba_val, y_val, config.TARGET_RECALL)

            # แล้ววัดผลจริงบน test set ที่ threshold นั้น
            r = eval_at_threshold(model, X_test[feats], y_test, thr)
            r["feature_set"] = set_name
            r["model"] = model_name
            r["n_features"] = len(feats)
            rows.append(r)

            print(f"  {model_name:14s} recall {r['recall']:.4f} | "
                  f"FP {r['fp_rate']*100:.3f}% | {r['latency_ms_per_1k']:.2f} ms/1k")

            joblib.dump(model, config.MODEL_DIR / f"stage1_{set_name}_{model_name}.pkl")

    table = pd.DataFrame(rows)[
        ["feature_set", "model", "n_features", "recall", "fp_rate",
         "precision", "f1", "ส่งต่อ_stage2_%", "latency_ms_per_1k", "threshold"]
    ]
    common.save_table(table, "04_stage1_all.csv", index=False)

    # ================================================================
    # ตารางหลัก: ตรึงโมเดลไว้ตัวเดียว แล้วเทียบเฉพาะชุดฟีเจอร์
    # ================================================================
    # ต้องตรึงโมเดล ไม่ใช่ "เลือกตัวที่ดีที่สุดของแต่ละชุด" เพราะถ้าปล่อยให้
    # แต่ละชุดฟีเจอร์ได้คนละโมเดล คอลัมน์ latency จะเทียบกันไม่ได้เลย
    # (RandomForest ช้ากว่า XGBoost โดยธรรมชาติอยู่แล้ว ไม่เกี่ยวกับจำนวนฟีเจอร์)
    #
    # ตัวแปรที่งานนี้สนใจคือ "ชุดฟีเจอร์" จึงต้องคุมตัวแปรอื่นให้คงที่
    MAIN_MODEL = "XGBoost"
    common.banner(f"ตารางหลัก — เทียบชุดฟีเจอร์ (ตรึงโมเดล = {MAIN_MODEL})")

    best = (table[table["model"] == MAIN_MODEL]
            .sort_values("n_features")
            .reset_index(drop=True))
    print(best.to_string(index=False))
    common.save_table(best, "04_stage1_comparison.csv", index=False)

    # ตารางรอง: ดูว่าถ้าเปลี่ยนโมเดลผลจะต่างไหม (ไว้ตอบถ้าอาจารย์ถาม)
    common.banner("ตารางรอง — แยกตามโมเดล (recall)")
    pivot = table.pivot_table(index="feature_set", columns="model", values="recall")
    n_feat = table.groupby("feature_set")["n_features"].first()
    pivot.insert(0, "n_features", n_feat)
    pivot = pivot.sort_values("n_features")
    print(pivot.to_string())
    common.save_table(pivot, "04_stage1_by_model.csv")

    # ---- เทียบกับ baseline เพื่อสรุปว่าคุ้มไหม ----
    base = best[best["feature_set"] == "all_features"].iloc[0]
    common.banner("สรุป — เทียบกับการใช้ฟีเจอร์ครบ")
    print(f"  baseline (all_features, {int(base['n_features'])} ฟีเจอร์):")
    print(f"    recall {base['recall']:.4f} | FP {base['fp_rate']*100:.3f}% "
          f"| {base['latency_ms_per_1k']:.2f} ms/1k\n")

    for _, r in best.iterrows():
        if r["feature_set"] == "all_features":
            continue
        d_feat = (1 - r["n_features"] / base["n_features"]) * 100
        d_lat = (1 - r["latency_ms_per_1k"] / base["latency_ms_per_1k"]) * 100
        d_rec = (r["recall"] - base["recall"]) * 100
        print(f"  {r['feature_set']:22s} ฟีเจอร์ -{d_feat:5.1f}% | "
              f"latency {d_lat:+6.1f}% | recall {d_rec:+.3f} จุด")

    print("\n  อ่านยังไง: ฟีเจอร์ลดเยอะ + latency ลด + recall ไม่ตก = ชุดที่ดีที่สุด")
    print("  เทียบ global_mean_top15 กับ shap_from_binary เพื่อดูว่าการยืมฟีเจอร์")
    print("  จาก stage 2 มาใช้กับ stage 1 (แนวคิดของอาจารย์) ได้ผลดีแค่ไหน")

    # เก็บชุดที่ดีที่สุดไว้ให้ 05 ใช้ต่อ
    winner = best[best["feature_set"] != "all_features"].sort_values("fp_rate").iloc[0]
    (config.OUT_DIR / "04_winning_feature_set.txt").write_text(
        winner["feature_set"], encoding="utf-8"
    )
    print(f"\n  ชุดฟีเจอร์ที่ดีที่สุด: {winner['feature_set']} "
          f"({int(winner['n_features'])} ฟีเจอร์)")


if __name__ == "__main__":
    main()
