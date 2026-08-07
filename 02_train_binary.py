"""
02 — โมเดลชั้นที่ 1: Binary (Normal vs Attack)

หน้าที่ของชั้นนี้ในระบบจริง:
    เป็น "ตัวกรอง" ที่รันกับ traffic ทุก flow ที่วิ่งเข้ามา ตอบแค่ว่า "น่าสงสัยไหม"
    ไม่ต้องฟันธงว่าเป็น attack ชนิดอะไร — ปล่อยให้ชั้นที่ 2 (สคริปต์ 03) ทำต่อ
    เพราะต้องรันตลอดเวลา จึงต้องเบาและเร็ว

สิ่งที่ไฟล์นี้ทำ:
    เทรน 3 โมเดลบนฟีเจอร์ครบทุกตัว → ผลที่ได้คือ "เพดาน"
    ที่ทุกวิธี feature selection ในสคริปต์ 04 จะถูกวัดเทียบ
    ถ้าไม่มีตัวเลขนี้ ตารางเปรียบเทียบทีหลังจะไม่มีความหมาย

รัน:  python 02_train_binary.py   (ต้องรัน 01 ก่อน)
"""
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import common
import config


def main():
    # ---- โหลดข้อมูลที่ 01 เตรียมไว้ ----
    df = common.load_clean()
    # target = 'binary_label' คือคอลัมน์ 0/1 ที่สร้างไว้ในขั้นที่ 4 ของสคริปต์ 01
    X, y = common.split_xy(df, "binary_label")

    common.banner(f"Binary: Normal vs Attack — {X.shape[1]} ฟีเจอร์, {len(X):,} แถว")

    # ---- แบ่ง train / test ----
    # stratify=y : บังคับให้สัดส่วน Normal:Attack ใน train กับ test เท่ากับข้อมูลเดิม
    #              ถ้าไม่ใส่ การสุ่มอาจทำให้ test มี attack น้อยผิดปกติ → ตัวเลขเพี้ยน
    # random_state : ตั้ง seed ให้แบ่งเหมือนเดิมทุกครั้ง ผลจึง reproduce ได้
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )
    print(f"  train {len(X_train):,} | test {len(X_test):,}")

    # ---- เทรนทั้ง 3 โมเดล ----
    rows = {}
    for name, model in common.build_models(n_classes=2).items():
        print(f"\n--- {name} ---")

        model.fit(X_train, y_train)          # เทรน: ให้โมเดลเรียนจาก train set
        rows[name] = common.evaluate(model, X_test, y_test, average="binary")

        for k, v in rows[name].items():
            print(f"  {k:20s} {v:.4f}")

        # เก็บโมเดลไว้ใช้ต่อ จะได้ไม่ต้องเทรนใหม่ทุกครั้งที่อยากดูอะไรเพิ่ม
        joblib.dump(model, config.MODEL_DIR / f"binary_{name}.pkl")

    # ---- ตารางเปรียบเทียบ ----
    # .T = transpose สลับแถวกับคอลัมน์ ให้ชื่อโมเดลเป็นแถว metric เป็นคอลัมน์
    table = pd.DataFrame(rows).T.sort_values("f1", ascending=False)
    common.banner("สรุปเปรียบเทียบ (binary)")
    print(table.to_string())
    common.save_table(table, "02_binary_comparison.csv")

    # ---- เลือกตัวที่ดีที่สุดเก็บไว้เป็น binary_best.pkl ----
    best = table.index[0]                     # แถวแรกหลัง sort = F1 สูงสุด
    best_model = joblib.load(config.MODEL_DIR / f"binary_{best}.pkl")
    joblib.dump(best_model, config.MODEL_DIR / "binary_best.pkl")

    # ---- ดูรายละเอียดของตัวที่ดีที่สุด ----
    common.banner(f"รายละเอียดของตัวที่ดีที่สุด: {best}")
    pred = best_model.predict(X_test)
    print(classification_report(y_test, pred, target_names=["Normal", "Attack"], digits=4))

    # confusion matrix บอกว่าผิดพลาดแบบไหน ซึ่ง accuracy ตัวเดียวบอกไม่ได้
    #   ช่องขวาบน = Normal ที่ถูกแจ้งผิดว่าเป็น attack (false positive, กวนใจแต่ไม่อันตราย)
    #   ช่องซ้ายล่าง = attack ที่หลุดรอด (false negative, อันตรายกว่ามาก)
    print("Confusion matrix (แถว=จริง, คอลัมน์=ทำนาย):")
    print(pd.DataFrame(
        confusion_matrix(y_test, pred),
        index=["จริง Normal", "จริง Attack"],
        columns=["ทาย Normal", "ทาย Attack"],
    ).to_string())

    # ================================================================
    # Threshold sweep — จุดที่คนมักเข้าใจผิด
    # ================================================================
    # ปกติโมเดล binary ใช้เกณฑ์ 0.5 คือ ถ้าความน่าจะเป็น > 0.5 ให้ตอบว่า attack
    # แต่สำหรับ "ตัวกรองชั้นแรก" เกณฑ์ 0.5 ผิดเจตนา เพราะ:
    #   - หน้าที่ของชั้นนี้คือ "ห้ามพลาด attack" → ต้องการ recall สูงมาก (~99%+)
    #   - false positive ยอมได้ เพราะยังมีชั้นที่ 2 คัดกรองต่ออีกที
    # ดังนั้นควรลดเกณฑ์ลงให้ recall สูงขึ้น แล้วดูว่าต้องแลกกับอะไร
    #
    # KPI ที่แท้จริงของชั้นนี้จึงไม่ใช่ accuracy แต่คือ
    # "ที่ recall 99% เหลืองานส่งต่อให้ชั้นสองกี่ %" — ยิ่งน้อยยิ่งดี
    if hasattr(best_model, "predict_proba"):
        common.banner("Threshold sweep — ชั้นแรกต้องห้ามพลาด attack (recall สูง)")

        # predict_proba คืนความน่าจะเป็นของทุกคลาส, [:, 1] = เอาเฉพาะคอลัมน์ของคลาส attack
        proba = best_model.predict_proba(X_test)[:, 1]

        sweep = []
        for thr in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            flagged = proba >= thr            # array True/False ว่าแถวไหนถูก flag

            tp = int(((y_test == 1) & flagged).sum())   # attack จริงที่ flag ถูก
            recall = tp / int((y_test == 1).sum())

            sweep.append({
                "threshold": thr,
                "attack_recall": recall,
                # สัดส่วนของ traffic ทั้งหมดที่ต้องส่งต่อให้ชั้นสองประมวลผล
                "ส่งต่อชั้น2_%": flagged.mean() * 100,
                # false positive rate: Normal ที่ถูก flag ผิด
                "normal_ที่หลุด_%": float(flagged[y_test == 0].mean() * 100),
            })

        sweep_df = pd.DataFrame(sweep)
        print(sweep_df.to_string(index=False))
        common.save_table(sweep_df, "02_threshold_sweep.csv", index=False)

        print("\nอ่านตารางนี้ยังไง: เลือก threshold ที่ attack_recall สูงพอ (เช่น >0.99)")
        print("แล้วดูคอลัมน์ ส่งต่อชั้น2_% ว่าเหลืองานให้ชั้นสองกี่ % — ยิ่งน้อยยิ่งดี")


if __name__ == "__main__":
    main()
