"""
05 — Zero-shot: ตรวจจับ attack ที่โมเดลไม่เคยเห็น

คำถามที่ไฟล์นี้ตอบ 2 ข้อ:
    1. stage 1 flag attack รูปแบบใหม่ที่ไม่เคยเทรนว่า "น่าสงสัย" ได้ไหม
    2. ถ้าลดฟีเจอร์ลงตามที่ SHAP คัดมา ความสามารถข้อ 1 หายไปหรือเปล่า
       ← ข้อนี้สำคัญมาก เพราะถ้าลดฟีเจอร์แล้ว generalize แย่ลง
         แนวทางทั้งหมดของงานก็มีปัญหา

ทำไมไม่ต้องประดิษฐ์ split เอง:
    InSDN แยกสภาพแวดล้อมมาให้แล้ว — OVS กับ metasploitable เก็บจากคนละ testbed
    และ U2R มีเฉพาะใน metasploitable
    เทรนด้วย OVS แล้วเทสด้วย metasploitable จึงได้ทั้ง
      (1) คลาสที่ไม่เคยเห็นเลย (U2R)
      (2) การทดสอบข้ามสภาพแวดล้อม ซึ่งท้าทายกว่าการสุ่มแบ่งภายในไฟล์เดียวกันมาก

การแบ่งข้อมูล:
    train = Normal ครึ่งหนึ่ง + attack ทั้งหมดจาก OVS
    test  = Normal อีกครึ่ง   + attack ทั้งหมดจาก metasploitable
    (ต้องแบ่ง Normal ครึ่ง ๆ เพราะ metasploitable ไม่มีแถว Normal เลย
     ถ้าไม่ใส่ Normal ใน test จะวัด false positive ไม่ได้)

รัน:  python 05_zero_shot.py   (ต้องรัน 01 และ 03 ก่อน)
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

import common
import config


def main():
    if not config.FEATURE_SETS_JSON.exists():
        raise SystemExit("ยังไม่มีชุดฟีเจอร์ — รัน `python 03_shap_features.py` ก่อน")
    sets = json.loads(config.FEATURE_SETS_JSON.read_text(encoding="utf-8"))

    df = common.load_clean()

    # แยกตามไฟล์ต้นทาง (คอลัมน์ source สร้างไว้ตั้งแต่ 01)
    normal = df[df["source"] == "Normal"]
    ovs = df[df["source"] == "OVS"]
    meta = df[df["source"] == "metasploitable"]

    normal_train, normal_test = train_test_split(
        normal, test_size=0.5, random_state=config.SEED
    )
    train = pd.concat([normal_train, ovs], ignore_index=True)
    test = pd.concat([normal_test, meta], ignore_index=True)

    seen = set(ovs["attack_class"].unique())
    unseen = set(meta["attack_class"].unique()) - seen

    common.banner("Zero-shot setup")
    print(f"  train: {len(train):,} แถว  (Normal {len(normal_train):,} + OVS {len(ovs):,})")
    print(f"  test : {len(test):,} แถว  (Normal {len(normal_test):,} + metasploitable {len(meta):,})")
    print(f"\n  attack ที่เคยเห็นตอนเทรน : {sorted(seen)}")
    print(f"  attack ที่ไม่เคยเห็นเลย   : {sorted(unseen) if unseen else '(ไม่มี)'}")

    y_train, y_test = train["binary_label"], test["binary_label"]

    # ================================================================
    # ทดสอบทุกชุดฟีเจอร์ ว่าลดฟีเจอร์แล้วยัง generalize ได้ไหม
    # ================================================================
    summary_rows = []
    per_class_frames = []

    for set_name, feats in sets.items():
        if not feats:
            continue
        print(f"\n--- {set_name} ({len(feats)} ฟีเจอร์) ---")

        # ใช้ XGBoost อย่างเดียว เพราะตัวแปรที่สนใจคือชุดฟีเจอร์ ไม่ใช่ชนิดโมเดล
        model = common.build_models(n_classes=2)["XGBoost"]
        model.fit(train[feats], y_train)

        r = common.evaluate(model, test[feats], y_test, average="binary")
        r["feature_set"] = set_name
        r["n_features"] = len(feats)
        summary_rows.append(r)
        print(f"  recall {r['recall']:.4f} | precision {r['precision']:.4f} "
              f"| {r['latency_ms_per_1k']:.2f} ms/1k")

        # ---- แยกดูรายคลาส ----
        # ตัวเลขรวมบอกไม่ได้ว่าจับคลาสที่ไม่เคยเห็นได้ไหม
        # เพราะ DDoS มี 73,529 แถว ส่วน U2R มี 17 แถว → ตัวเลขรวมถูก DDoS กลบหมด
        t = test.copy()
        t["pred"] = model.predict(test[feats])

        for cls, grp in t.groupby("attack_class"):
            if cls == config.NORMAL_LABEL:
                continue
            per_class_frames.append({
                "feature_set": set_name,
                "attack_class": cls,
                "เคยเห็น": "ใช่" if cls in seen else "ไม่ (zero-shot)",
                "n_test": len(grp),
                "recall": (grp["pred"] == 1).mean(),
            })

        fp = (t.loc[t["attack_class"] == config.NORMAL_LABEL, "pred"] == 1).mean()
        for row in per_class_frames[-len(t["attack_class"].unique()):]:
            row["fp_rate_normal"] = fp

    # ---- ตารางสรุปรวม ----
    summary = pd.DataFrame(summary_rows)[
        ["feature_set", "n_features", "recall", "precision", "f1", "latency_ms_per_1k"]
    ].sort_values("n_features")
    common.banner("สรุป zero-shot ต่อชุดฟีเจอร์ (เรียงตามจำนวนฟีเจอร์)")
    print(summary.to_string(index=False))
    common.save_table(summary, "05_zero_shot_by_feature_set.csv", index=False)

    # ---- ตาราง pivot: แถว = คลาส, คอลัมน์ = ชุดฟีเจอร์ ----
    # อ่านง่ายที่สุดสำหรับคำถาม "ลดฟีเจอร์แล้วคลาสไหนพัง"
    per_class = pd.DataFrame(per_class_frames)
    pivot = per_class.pivot_table(
        index=["attack_class", "เคยเห็น", "n_test"],
        columns="feature_set", values="recall",
    ).reset_index()

    common.banner("Recall รายคลาส × ชุดฟีเจอร์")
    print(pivot.to_string(index=False))
    common.save_table(pivot, "05_zero_shot_per_class.csv", index=False)

    print("\n  อ่านยังไง:")
    print("    - แถว U2R คือคลาสที่ไม่เคยเทรน — ดูว่าชุดฟีเจอร์ที่เล็กลงยังจับได้ไหม")
    print("    - ถ้า recall ของ U2R ไม่ตกตอนลดฟีเจอร์ = การลดฟีเจอร์ไม่ทำลาย generalization")
    print("    - ถ้าตกชัดเจน = ต้องระวัง เพราะฟีเจอร์ที่ตัดไปอาจจำเป็นกับ attack แบบใหม่")


if __name__ == "__main__":
    main()
