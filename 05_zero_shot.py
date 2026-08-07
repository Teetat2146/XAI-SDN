"""
05 — Zero-shot: ตรวจจับ attack ที่โมเดลไม่เคยเห็น

แนวคิดจากที่ประชุม:
    ในโลกจริง attack รูปแบบใหม่เกิดขึ้นตลอด โมเดลที่เทรนจากข้อมูลเก่าจะเจอของใหม่แน่นอน
    คำถามคือ: ชั้นแรกยัง flag ว่า "น่าสงสัย" ได้ไหม แม้จะไม่รู้จักชนิดของมัน
    (ไม่ต้องจำแนกชนิดถูก แค่ต้องไม่ปล่อยผ่าน)

ทำไมไม่ต้องประดิษฐ์ split เอง:
    InSDN แยกสภาพแวดล้อมมาให้อยู่แล้ว — OVS กับ metasploitable เก็บจากคนละ testbed
    และ U2R มีเฉพาะใน metasploitable เท่านั้น
    เทรนด้วย OVS แล้วเทสด้วย metasploitable จึงได้ทั้ง
      (1) คลาสที่ไม่เคยเห็นเลย (U2R)
      (2) การทดสอบข้ามสภาพแวดล้อม ซึ่งท้าทายกว่าการสุ่มแบ่งภายในไฟล์เดียวกันมาก

การแบ่งข้อมูล:
    train = Normal ครึ่งหนึ่ง + attack ทั้งหมดจาก OVS
    test  = Normal อีกครึ่ง   + attack ทั้งหมดจาก metasploitable
    (ต้องแบ่ง Normal ครึ่ง ๆ เพราะ metasploitable ไม่มีแถว Normal เลย
     ถ้าไม่ใส่ Normal เข้าไปใน test เราจะวัด false positive ไม่ได้)

รัน:  python 05_zero_shot.py   (ต้องรัน 01 ก่อน)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import common
import config


def main():
    df = common.load_clean()

    # แยกข้อมูลตามไฟล์ต้นทาง (คอลัมน์ source สร้างไว้ตั้งแต่สคริปต์ 01)
    normal = df[df["source"] == "Normal"]
    ovs = df[df["source"] == "OVS"]
    meta = df[df["source"] == "metasploitable"]

    # แบ่ง Normal ครึ่งต่อครึ่ง ให้ทั้ง train และ test มี traffic ปกติ
    normal_train, normal_test = train_test_split(
        normal, test_size=0.5, random_state=config.SEED
    )
    train = pd.concat([normal_train, ovs], ignore_index=True)
    test = pd.concat([normal_test, meta], ignore_index=True)

    # หาว่าคลาสไหนที่อยู่ใน test แต่ไม่เคยอยู่ใน train = คลาส zero-shot
    seen = set(ovs["attack_class"].unique())
    unseen = set(meta["attack_class"].unique()) - seen      # ลบ set = เอาที่ไม่ซ้ำกัน

    common.banner("Zero-shot setup")
    print(f"  train: {len(train):,} แถว  (Normal {len(normal_train):,} + OVS {len(ovs):,})")
    print(f"  test : {len(test):,} แถว  (Normal {len(normal_test):,} + metasploitable {len(meta):,})")
    print(f"\n  attack ที่เคยเห็นตอนเทรน : {sorted(seen)}")
    print(f"  attack ที่ไม่เคยเห็นเลย   : {sorted(unseen) if unseen else '(ไม่มี)'}")

    # ใช้ binary_label เพราะทดสอบชั้นที่ 1 (แค่ถามว่าน่าสงสัยไหม)
    # จะใช้ multi-class ไม่ได้ เพราะโมเดลไม่มีทางทายคลาสที่ไม่เคยเห็นได้อยู่แล้ว
    feats = common.feature_columns(df)
    X_train, y_train = train[feats], train["binary_label"]
    X_test, y_test = test[feats], test["binary_label"]

    rows = {}
    preds = {}
    for name, model in common.build_models(n_classes=2).items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        rows[name] = common.evaluate(model, X_test, y_test, average="binary")
        preds[name] = model.predict(X_test)      # เก็บคำทำนายไว้วิเคราะห์รายคลาสต่อ
        for k, v in rows[name].items():
            print(f"  {k:20s} {v:.4f}")

    # เรียงตาม recall ไม่ใช่ f1 — เพราะชั้นแรกให้ความสำคัญกับ "ห้ามพลาด attack" มากที่สุด
    table = pd.DataFrame(rows).T.sort_values("recall", ascending=False)
    common.banner("สรุป zero-shot (เรียงตาม recall — ชั้นแรกห้ามพลาด attack)")
    print(table.to_string())
    common.save_table(table, "05_zero_shot_comparison.csv")

    # ================================================================
    # ตัวเลขที่ขายงานได้ที่สุด: recall แยกรายคลาส
    # ================================================================
    # ตัวเลขรวมบอกไม่ได้ว่าโมเดลจับคลาสที่ไม่เคยเห็นได้ไหม
    # เพราะ DDoS มี 73,529 แถว ส่วน U2R มี 17 แถว — ตัวเลขรวมถูก DDoS กลบหมด
    # ต้องแยกดูทีละคลาสเท่านั้น
    common.banner("Recall รายคลาสบน test set")
    best = table.index[0]
    test = test.copy()                            # copy กันแก้ DataFrame ต้นฉบับ
    test["pred"] = preds[best]

    detail = []
    for cls, grp in test.groupby("attack_class"):
        if cls == config.NORMAL_LABEL:            # Normal ไม่ใช่ attack ข้ามไป
            continue
        detail.append({
            "attack_class": cls,
            "เคยเห็นตอนเทรน": "ใช่" if cls in seen else "ไม่ (zero-shot)",
            "n_test": len(grp),
            # recall = สัดส่วนของแถวคลาสนี้ที่ถูก flag ว่าเป็น attack (pred == 1)
            "recall": (grp["pred"] == 1).mean(),
        })

    detail_df = pd.DataFrame(detail).sort_values("recall")
    print(f"  โมเดล: {best}\n")
    print(detail_df.to_string(index=False))
    common.save_table(detail_df, "05_zero_shot_per_class.csv", index=False)

    # false positive rate: traffic ปกติที่ถูกแจ้งเตือนผิด
    # ต้องรายงานคู่กับ recall เสมอ ไม่งั้นโมเดลที่ตอบว่า "attack" ทุกแถว
    # ก็จะได้ recall 100% ทั้งที่ไร้ประโยชน์
    normal_fp = (test.loc[test["attack_class"] == config.NORMAL_LABEL, "pred"] == 1).mean()
    print(f"\n  False positive rate บน Normal: {normal_fp * 100:.2f}%")
    print("\n  แถวที่ 'ไม่ (zero-shot)' คือตัวเลขที่เอาไปตอบอาจารย์ได้ตรงที่สุด")
    print("  ว่าโมเดลจับ attack รูปแบบใหม่ที่ไม่เคยเทรนได้จริงหรือไม่")


if __name__ == "__main__":
    main()
