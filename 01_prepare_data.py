"""
01 — เตรียมข้อมูล InSDN

สิ่งที่ไฟล์นี้ทำ:
    รวม 3 ไฟล์ CSV → ตัดคอลัมน์ที่ทำให้โมเดลโกง → ตัดคอลัมน์ที่ไม่มีข้อมูล
    → สร้าง label 2 ชั้น → บันทึกเป็น parquet ให้สคริปต์อื่นใช้ต่อ

ทำไมต้องแยกเป็นไฟล์ต่างหาก:
    การเตรียมข้อมูลต้องทำ "ครั้งเดียว" แล้วทุกการทดลองใช้ข้อมูลชุดเดียวกัน
    ถ้าแต่ละสคริปต์ clean เอง แล้ววันหนึ่ง clean ไม่เหมือนกัน ผลจะเทียบกันไม่ได้เลย

รัน:  python 01_prepare_data.py
"""
import json

import numpy as np
import pandas as pd

import common
import config


def main():
    # ================================================================
    # ขั้นที่ 1 — โหลดทั้ง 3 ไฟล์แล้วรวมเป็นตารางเดียว
    # ================================================================
    common.banner("1. โหลด 3 ไฟล์")
    frames = []

    for source, path in config.RAW_FILES.items():
        if not path.exists():
            raise SystemExit(f"ไม่พบไฟล์: {path}")

        # low_memory=False : ให้ pandas อ่านทั้งไฟล์ก่อนเดาชนิดข้อมูล
        # ถ้าไม่ใส่ pandas จะอ่านทีละก้อนแล้วเดาชนิดไม่ตรงกันในแต่ละก้อน → warning + ชนิดเพี้ยน
        df = pd.read_csv(path, low_memory=False)

        # ชื่อคอลัมน์บางตัวอาจมีช่องว่างนำหน้า/ต่อท้ายติดมา ทำให้อ้างชื่อไม่เจอ
        df.columns = df.columns.str.strip()

        # บั๊กของ InSDN ที่เจอตอนรันจริง:
        #   OVS.csv เขียน 'DDoS ' (มีเว้นวรรคท้าย) แต่ metasploitable-2.csv เขียน 'DDoS'
        # ถ้าไม่ strip → Python มองเป็นคนละ string → กลายเป็น 8 คลาสแทนที่จะเป็น 7
        # และ 05_zero_shot.py จะเข้าใจผิดว่า DDoS ของ metasploitable เป็นคลาสที่ไม่เคยเห็น
        df[config.LABEL_COL] = df[config.LABEL_COL].str.strip()

        # จำไว้ว่าแถวนี้มาจากไฟล์ไหน — 05_zero_shot.py ใช้คอลัมน์นี้แบ่ง train/test
        df["source"] = source

        print(f"  {source:16s} {df.shape[0]:>7,} แถว × {df.shape[1]} คอลัมน์")
        frames.append(df)

    # ต่อทั้ง 3 ตารางในแนวตั้ง (แถวต่อแถว) เพราะทุกไฟล์มีคอลัมน์ชุดเดียวกัน
    # ignore_index=True : สร้างเลข index ใหม่ 0,1,2,... ไม่งั้น index จะซ้ำกันจาก 3 ไฟล์
    df = pd.concat(frames, ignore_index=True)
    print(f"\n  รวม             {df.shape[0]:>7,} แถว × {df.shape[1]} คอลัมน์")

    # ================================================================
    # ขั้นที่ 2 — encode คอลัมน์ที่เป็นตัวหนังสือ (ไม่ตัดทิ้ง)
    # ================================================================
    # เดิมเราตัด identifier ทิ้งตรงนี้เลย แต่ประชุม 2026-08-08 อาจารย์ท้วงว่า
    # ต้องพิสูจน์ *ด้วยการทดลอง* ว่าตัดถูก ไม่ใช่แค่อ้างเหตุผล
    # → เก็บไว้ทั้งหมด แล้วให้แต่ละระดับใน 02 เลือกเองว่าจะตัดถึงไหน
    #
    # โมเดลรับได้แต่ตัวเลข จึงต้อง encode ก่อน:
    #   Timestamp → epoch seconds (มีความหมายเชิงลำดับจริง)
    #   ที่เหลือ  → LabelEncoder (รหัสประจำค่า ไม่มีความหมายเชิงลำดับ
    #               แต่ tree แบ่งด้วยจุดตัด ไม่สนลำดับ จึงใช้ได้)
    common.banner("2. encode คอลัมน์ตัวหนังสือ (เก็บไว้ ไม่ตัด)")

    from sklearn.preprocessing import LabelEncoder

    for c in config.ID_COLS:
        if c not in df.columns:
            continue
        n_uniq = df[c].nunique()
        if c == "Timestamp":
            ts = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            df[c] = ts.astype("int64") // 10**9      # epoch seconds
            df[c] = df[c].fillna(-1)
        else:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        print(f"  encode {c:12s} ค่าไม่ซ้ำ {n_uniq:>7,} ({n_uniq/len(df)*100:5.2f}% ของแถว)")

    # ================================================================
    # ขั้นที่ 3 — จัดการค่า inf และ NaN
    # ================================================================
    # CICFlowMeter คำนวณ Flow Byts/s = bytes ÷ duration
    # พอ duration = 0 (flow สั้นมากจนวัดไม่ได้) ผลลัพธ์เป็น infinity
    # sklearn/xgboost รับค่า inf ไม่ได้ ต้องแปลงเป็น NaN แล้วตัดทิ้ง
    common.banner("3. จัดการ inf / NaN")

    num_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(df[num_cols].to_numpy()).sum())
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna().reset_index(drop=True)

    print(f"  พบค่า inf {n_inf:,} ช่อง → แปลงเป็น NaN")
    print(f"  ตัดแถวที่มี NaN: {before - len(df):,} แถว ({(before-len(df))/before*100:.2f}%)")
    print(f"  เหลือ {len(df):,} แถว")

    # ================================================================
    # ขั้นที่ 4 — สร้าง label 2 ชั้น ให้ตรงกับ pipeline 2 ชั้น
    # ================================================================
    common.banner("4. สร้าง label 2 ชั้น")

    # ชั้นที่ 1 (binary filter): แค่ถามว่า "น่าสงสัยไหม" → Normal=0, ที่เหลือทั้งหมด=1
    df["binary_label"] = (df[config.LABEL_COL] != config.NORMAL_LABEL).astype(int)

    # ชั้นที่ 2 (multi-class): ถามว่า "เป็น attack ชนิดไหน" → เก็บชื่อคลาสไว้ตรง ๆ
    df["attack_class"] = df[config.LABEL_COL]

    print(df["binary_label"].value_counts().rename({0: "Normal", 1: "Attack"}).to_string())
    print()
    # crosstab = ตารางไขว้ ดูว่าแต่ละ attack มาจากไฟล์ไหนบ้าง
    # สำคัญมากสำหรับ zero-shot — ต้องรู้ว่าคลาสไหนมีเฉพาะในไฟล์เดียว
    print(pd.crosstab(df["attack_class"], df["source"]).to_string())

    # ================================================================
    # ขั้นที่ 5 — หากลุ่มคอลัมน์ที่ "ตัดได้" แล้วบันทึกรายชื่อ (ยังไม่ตัด)
    # ================================================================
    common.banner("5. หากลุ่มคอลัมน์ที่ตัดได้")

    feats = [c for c in df.columns if c not in common.META_COLS]

    # กลุ่ม constant — ค่าเดียวทั้งคอลัมน์ (บั๊ก CICFlowMeter)
    const_cols = [c for c in feats if df[c].nunique(dropna=False) <= 1]
    print(f"  ค่าคงที่     {len(const_cols):>2} คอลัมน์")
    for c in const_cols:
        print(f"      {c}")

    # กลุ่ม dup — ยืนยันว่า correlation สูงจริงก่อนใช้รายชื่อจาก config
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    corr = df[num].corr().abs()
    dup_cols = []
    for c in config.DUP_COLS:
        if c not in corr.columns:
            continue
        partner = corr[c].drop(index=c).idxmax()
        v = corr.loc[c, partner]
        if v >= 0.999:
            dup_cols.append(c)
            print(f"  ซ้ำซ้อน      {c:20s} ~ {partner:20s} r={v:.4f}")
        else:
            print(f"  ข้าม (r ต่ำ) {c:20s} ~ {partner:20s} r={v:.4f}")

    drop_lists = {
        "id": [c for c in config.ID_COLS if c in df.columns],
        "port": [c for c in config.PORT_COLS if c in df.columns],
        "const": const_cols,
        "dup": dup_cols,
        "expensive": [c for c in config.EXPENSIVE_COLS if c in df.columns],
    }
    config.DROP_LISTS_JSON.write_text(
        json.dumps(drop_lists, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  [saved] {config.DROP_LISTS_JSON}")

    print("\n  จำนวนฟีเจอร์แต่ละระดับ:")
    for lv in config.FEATURE_LEVELS:
        if lv == "shap_top15":
            print(f"    {lv:14s} (สร้างใน 03)")
        else:
            print(f"    {lv:14s} {len(common.feature_columns(df, level=lv)):>3} ตัว")

    # ================================================================
    # ขั้นที่ 6 — บันทึก
    # ================================================================
    common.banner("6. บันทึก")

    feats = common.feature_columns(df)

    # เช็คว่าไม่มีคอลัมน์ที่เป็นตัวหนังสือหลงเหลือ — โมเดล tree รับได้แต่ตัวเลข
    non_numeric = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print(f"  เตือน: ยังมีคอลัมน์ที่ไม่ใช่ตัวเลข {non_numeric} — ตรวจสอบก่อนเทรน")

    df.to_parquet(config.CLEAN_PARQUET, index=False)
    print(f"  จำนวนแถวสุดท้าย: {len(df):,} แถว")      # ← เพิ่มบรรทัดนี้
    print(f"  ฟีเจอร์ที่ใช้เทรนได้: {len(feats)} ตัว")
    print(f"  [saved] {config.CLEAN_PARQUET}")

    # ตารางสรุปฟีเจอร์ไว้ดูภาพรวม (ค่าเฉลี่ย/การกระจาย) เผื่อต้องอธิบายในรายงาน
    summary = pd.DataFrame(
        {"n_unique": [df[c].nunique() for c in feats],
         "mean": [df[c].mean() for c in feats],
         "std": [df[c].std() for c in feats]},
        index=feats,
    )
    common.save_table(summary, "01_feature_summary.csv")


if __name__ == "__main__":
    # บรรทัดนี้แปลว่า "รัน main() เฉพาะตอนสั่งรันไฟล์นี้โดยตรง"
    # ถ้าไฟล์อื่น import ไฟล์นี้ main() จะไม่ทำงานเอง
    main()
