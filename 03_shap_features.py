"""
03 — SHAP: ดูว่าแต่ละระดับการตัดฟีเจอร์ทำให้ SHAP เปลี่ยนไปยังไง
      แล้วผลิตชุดฟีเจอร์ `shap_top15` ให้ 02 เอาไปทดสอบเป็นระดับที่ 7

ตอบ 2 คำถาม:

  1. ในระดับ `raw` — identifier (Src IP, Dst IP, Flow ID) กับ Port
     ติดอันดับความสำคัญที่เท่าไหร่
     ถ้าติดอันดับต้น ๆ = หลักฐานว่าโมเดลพึ่งมันจริง (leakage)
     ซึ่งอธิบายว่าทำไม gap ระหว่าง random กับ cross-env ถึงกว้าง

  2. ตัดฟีเจอร์ออกไปแล้ว ฟีเจอร์ที่เหลือถูกจัดอันดับใหม่ยังไง
     ถ้าอันดับสลับไปมามาก = ฟีเจอร์แย่งความสำคัญกันอยู่ (เช่นคู่ที่ correlation สูง)

จากนั้นสร้างชุดฟีเจอร์ด้วย 4 วิธีรวม จากระดับ `no_port` (65 ฟีเจอร์)

รัน:  python 03_shap_features.py   (ต้องรัน 01 และ 02 ก่อน)
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap

import common
import config

# ระดับที่คำนวณ SHAP (ข้าม shap_top15 เพราะมันเป็นผลผลิตของไฟล์นี้เอง)
LEVELS = [lv for lv in config.FEATURE_LEVELS if lv != "shap_top15"]

# ระดับที่ใช้สร้างชุดฟีเจอร์ — ตรงกับที่ใช้กันมาตลอด
BASE_LEVEL = "no_port"


def shap_importance(model, X_train, class_names, title) -> pd.DataFrame:
    """คำนวณ SHAP คืนตารางความสำคัญ (index=feature, column=class) normalize แล้ว

    ใช้ train set เท่านั้น — ถ้าคำนวณบน test แล้วเอาไปเลือกฟีเจอร์ = leakage
    subsample เพราะ SHAP กินแรม: array ขนาด (แถว × ฟีเจอร์ × คลาส)
    """
    print(f"\n  SHAP — {title}")
    n = min(config.SHAP_SAMPLE, len(X_train))
    X_shap = X_train.sample(n=n, random_state=config.SEED)

    sv = common.to_shap_array(shap.TreeExplainer(model).shap_values(X_shap))
    print(f"    subsample {n:,} แถว | shape {sv.shape}")
    return common.mean_abs_shap(sv, list(X_shap.columns), class_names)


def merge_strategies(imp: pd.DataFrame) -> dict[str, list[str]]:
    """4 วิธีรวมฟีเจอร์ข้ามคลาส"""
    k = config.TOPK
    top_per_class = {c: set(imp[c].nlargest(k).index) for c in imp.columns}

    # union: ติด Top-K ของคลาสไหนก็เอา — ไม่พลาดของเฉพาะคลาส แต่บวม
    union = set().union(*top_per_class.values())
    # intersection: ต้องติดทุกคลาสพร้อมกัน — เข้มมาก มักเหลือน้อยเกินไป
    inter = set(imp.index).intersection(*top_per_class.values())
    # mean: เฉลี่ยข้ามคลาสก่อนแล้วตัดครั้งเดียว = "ใช้ชุดเดียวกันทุกคลาส"
    mean = set(imp.mean(axis=1).nlargest(k).index)

    # dynamic: แต่ละคลาสเก็บจนความสำคัญสะสมถึงเกณฑ์ → K ต่างกันตามคลาส
    dynamic, dyn_k = set(), {}
    for c in imp.columns:
        s = imp[c].sort_values(ascending=False)
        n = int((s.cumsum() < config.CUMULATIVE_THRESHOLD).sum()) + 1
        dyn_k[c] = n
        dynamic |= set(s.index[:n])

    print(f"\n  dynamic-K ต่อคลาส (threshold {config.CUMULATIVE_THRESHOLD}):")
    for c, n in dyn_k.items():
        print(f"    {str(c):16s} K = {n}")

    return {"union": sorted(union), "intersection": sorted(inter),
            "mean": sorted(mean), "dynamic": sorted(dynamic)}


def main():
    df = common.load_clean()

    # ================================================================
    # ส่วนที่ 1 — SHAP ทุกระดับ ดูว่าอันดับเปลี่ยนยังไง
    # ================================================================
    common.banner("SHAP แต่ละระดับการตัดฟีเจอร์")

    watch = config.ID_COLS + config.PORT_COLS   # คอลัมน์ที่สงสัยว่า leak
    rank_rows, imps = [], {}

    for level in LEVELS:
        path = config.MODEL_DIR / f"stage2_{level}_XGBoost.pkl"
        if not path.exists():
            print(f"\n  ข้าม {level} — ยังไม่มีโมเดล (รัน 02 ก่อน)")
            continue

        model = joblib.load(path)
        le = joblib.load(config.MODEL_DIR / f"label_encoder_{level}.pkl")
        feats = common.feature_columns(df, level=level)
        train, _ = common.make_split(df, "random")

        imp = shap_importance(model, train[feats], list(le.classes_),
                              f"{level} ({len(feats)} ฟีเจอร์)")
        imps[level] = imp
        common.save_table(imp, f"03_shap_{level}.csv")

        # อันดับเฉลี่ยข้ามคลาสของคอลัมน์ที่สงสัย
        mean_rank = imp.mean(axis=1).rank(ascending=False).astype(int)
        for c in watch:
            if c in mean_rank.index:
                rank_rows.append({"level": level, "n_features": len(feats),
                                  "column": c, "rank": int(mean_rank[c]),
                                  "importance": float(imp.loc[c].mean())})

    if rank_rows:
        rk = pd.DataFrame(rank_rows).pivot_table(
            index="column", columns="level", values="rank")
        rk = rk[[lv for lv in LEVELS if lv in rk.columns]]
        common.banner("อันดับความสำคัญของคอลัมน์ที่สงสัยว่า leak")
        print(rk.to_string())
        print("\n  ตัวเลข = อันดับจากทั้งหมด (1 = สำคัญที่สุด)")
        print("  ถ้าติดอันดับต้น ๆ ในระดับ raw = โมเดลพึ่งมันจริง")
        common.save_table(rk, "03_leak_column_rank.csv")

    # ================================================================
    # ส่วนที่ 2 — สร้างชุดฟีเจอร์จากระดับ no_port
    # ================================================================
    if BASE_LEVEL not in imps:
        raise SystemExit(f"ยังไม่มี SHAP ของระดับ {BASE_LEVEL} — รัน 02 ก่อน")

    common.banner(f"สร้างชุดฟีเจอร์จากระดับ {BASE_LEVEL}")
    imp = imps[BASE_LEVEL]

    common.banner("Top 10 ฟีเจอร์ของแต่ละคลาส")
    for c in imp.columns:
        print(f"\n  [{c}]")
        for f, v in imp[c].nlargest(10).items():
            print(f"    {v:.4f}  {f}")

    sets = merge_strategies(imp)
    sets["all_features"] = common.feature_columns(df, level=BASE_LEVEL)

    print()
    for name, feats in sets.items():
        print(f"  {name:16s} {len(feats):>3} ฟีเจอร์")

    config.FEATURE_SETS_JSON.write_text(
        json.dumps(sets, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {config.FEATURE_SETS_JSON}")
    print("\n→ รัน 02 อีกรอบเพื่อให้ระดับ shap_top15 ถูกทดสอบด้วย")


if __name__ == "__main__":
    main()
