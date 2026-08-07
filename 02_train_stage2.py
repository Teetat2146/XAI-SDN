"""
02 — โมเดลชั้นที่ 2 (stage 2): Multi-class จำแนกชนิด attack

หน้าที่ของชั้นนี้ในระบบจริง:
    รับเฉพาะ flow ที่ชั้นที่ 1 บอกว่า "น่าสงสัย" แล้วจำแนกว่าเป็น attack ชนิดไหน
    ซับซ้อนกว่าชั้นแรกได้ เพราะไม่ต้องรันกับทุก flow

ทำไมเทรนเฉพาะแถวที่เป็น attack:
    ในระบบจริงชั้นนี้จะไม่มีวันเห็น traffic ปกติ (ชั้นแรกกรองไปแล้ว)
    ถ้าใส่ Normal เข้าไปเทรนด้วย จะกลายเป็นทำงานซ้ำกับชั้นแรก
    และตัวเลขที่ได้จะไม่สะท้อนการใช้งานจริง

รัน:  python 02_train_stage2.py   (ต้องรัน 01 ก่อน)

หมายเหตุ: ไฟล์นี้ต้องรัน *ก่อน* stage 1 เพราะ SHAP ที่ได้จากโมเดลนี้
         คือตัวที่ใช้คัดฟีเจอร์ไปให้ stage 1 (ดู 03 และ 04)
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import common
import config


def main():
    df = common.load_clean()

    # เอาเฉพาะแถวที่เป็น attack (binary_label == 1) ตัด Normal ทิ้ง
    df = df[df["binary_label"] == 1].reset_index(drop=True)

    # ---- แปลงชื่อคลาสเป็นตัวเลข ----
    # XGBoost รับ label เป็นตัวเลข 0,1,2,... เท่านั้น รับ string ไม่ได้
    # LabelEncoder แปลง ['BFA','DDoS',...] → [0,1,...] และจำ mapping ไว้แปลงกลับได้
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df["attack_class"]), name="y")
    X = df[common.feature_columns(df)]

    common.banner(f"Multi-class: {len(le.classes_)} ชนิด attack — {len(X):,} แถว")

    # ---- ดูความไม่สมดุลของข้อมูลก่อน ----
    # ต้องดูก่อนเทรนเสมอ เพราะมันกำหนดว่าจะใช้ metric ตัวไหน
    dist = df["attack_class"].value_counts()
    print(dist.to_string())
    print(f"\n  imbalance ratio (มากสุด/น้อยสุด) = {dist.max() / dist.min():,.0f} เท่า")
    print("  → ใช้ Macro-F1 เป็น metric หลัก ไม่ใช่ accuracy")
    print("     เพราะ accuracy จะถูกกลบด้วยคลาสใหญ่จนมองไม่เห็นว่าคลาสเล็กพัง")

    # เตือนเรื่องคลาสที่เล็กเกินไปจนผลไม่มีความหมายทางสถิติ
    # เช่น U2R มี 17 แถว → หลังแบ่ง test 20% เหลือทดสอบแค่ 3 แถว
    # ทายถูก 3 จาก 3 ได้ recall 1.00 ซึ่งไม่ได้แปลว่าโมเดลเก่ง แค่บังเอิญ
    rare = dist[dist < 50]
    if len(rare):
        print(f"\n  เตือน: คลาส {list(rare.index)} มีน้อยกว่า 50 แถว")
        print("         ผลที่ได้แทบไม่มีนัยสำคัญทางสถิติ — พิจารณาใช้เป็น zero-shot class แทน")

    # stratify=y สำคัญมากตรงนี้ — ถ้าไม่ใส่ คลาสที่มี 17 แถวอาจไม่มีเลยใน test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )

    # ---- เทรน 3 โมเดล ----
    rows = {}
    for name, model in common.build_models(n_classes=len(le.classes_)).items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)

        # average="macro" : เฉลี่ยแบบให้ทุกคลาสน้ำหนักเท่ากัน (ดูคำอธิบายใน common.evaluate)
        rows[name] = common.evaluate(model, X_test, y_test, average="macro")

        for k, v in rows[name].items():
            print(f"  {k:20s} {v:.4f}")
        joblib.dump(model, config.MODEL_DIR / f"stage2_{name}.pkl")

    table = pd.DataFrame(rows).T.sort_values("f1", ascending=False)
    common.banner("สรุปเปรียบเทียบ (multi-class, macro average)")
    print(table.to_string())
    common.save_table(table, "02_stage2_comparison.csv")

    # ---- บันทึกตัวที่ดีที่สุด + encoder ----
    # สคริปต์ 04 จะโหลดสองไฟล์นี้ไปใช้ต่อ จึงไม่ต้องเทรนซ้ำ
    best = table.index[0]
    best_model = joblib.load(config.MODEL_DIR / f"stage2_{best}.pkl")
    joblib.dump(best_model, config.MODEL_DIR / "stage2_best.pkl")
    joblib.dump(le, config.MODEL_DIR / "label_encoder.pkl")   # ต้องเก็บไว้แปลงเลขกลับเป็นชื่อคลาส

    # ---- ดูผลรายคลาส ----
    # ตรงนี้สำคัญกว่าตัวเลขรวม เพราะบอกว่าคลาสไหนพัง
    common.banner(f"per-class ของตัวที่ดีที่สุด: {best}")
    pred = best_model.predict(X_test)
    print(classification_report(y_test, pred, target_names=le.classes_,
                               digits=4, zero_division=0))

    # confusion matrix บอกว่าโมเดล "สับสน" ระหว่างคลาสไหนกับคลาสไหน
    # อ่านแนวนอน: แถว BFA คอลัมน์ Probe = จำนวน BFA ที่ถูกทายผิดเป็น Probe
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix (แถว=จริง, คอลัมน์=ทำนาย):")
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())


if __name__ == "__main__":
    main()
