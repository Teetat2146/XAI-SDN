"""
make_report.py — รวมผลลัพธ์ทุกอันใน outputs/ เป็นรายงานเดียวที่อ่านง่าย

สร้าง RESULTS.md ที่เปิดใน VS Code แล้วกด Cmd+Shift+V จะเห็นเป็นตารางสวย ๆ

ไฟล์นี้ไม่คำนวณอะไรใหม่ แค่อ่าน CSV ที่สคริปต์ 01-05 สร้างไว้แล้วมาจัดรูปแบบ
ดังนั้นรันกี่ครั้งก็ได้ ไม่กระทบผลการทดลอง

รัน:  python make_report.py   (ต้องรัน 01-05 ให้ครบก่อน)
"""
import pandas as pd

import config

pd.set_option("display.width", 200)


def read(name, index_col=0):
    path = config.OUT_DIR / name
    return pd.read_csv(path, index_col=index_col) if path.exists() else None


def md(df, dec=4):
    """แปลง DataFrame เป็นตาราง markdown พร้อมปัดทศนิยม"""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(dec)
    return d.to_markdown()


def main():
    out = []
    add = out.append

    add("# ผลการทดลอง — XAI-Guided Two-Stage IDS (InSDN)\n")
    add("> สร้างอัตโนมัติจาก `make_report.py` — อย่าแก้ไฟล์นี้โดยตรง\n")

    add("## ลำดับของงาน\n")
    add("```")
    add("01 เตรียมข้อมูล")
    add("  → 02 เทรน stage 2 (multi-class)")
    add("      → 03 SHAP หาฟีเจอร์สำคัญของแต่ละ attack แล้วรวมเป็นชุด")
    add("          → 04 เอาชุดฟีเจอร์ไปเทรน stage 1  ← จุดที่พิสูจน์แนวคิด")
    add("          → 05 ทดสอบ attack ที่ไม่เคยเห็น")
    add("```\n")
    add("**ทำไม stage 2 มาก่อน stage 1:** เพราะต้องใช้ SHAP จาก stage 2")
    add("มาคัดฟีเจอร์ให้ stage 1 ใช้ — stage 1 จึงเบาลงได้อย่างมีหลักฐานรองรับ\n")
    add("---\n")

    # ---------- stage 2 ----------
    add("## 1. Stage 2 — Multi-class (7 ชนิด attack)\n")
    t = read("02_stage2_comparison.csv")
    if t is not None:
        add(md(t.rename(columns={"latency_ms_per_1k": "latency (ms/1k)"})) + "\n")
        add("**อ่านยังไง:** accuracy ต่างกันแค่หลักหมื่น แต่ **macro-F1 ต่างกันมาก**")
        add("— RandomForest แพ้เพราะพลาดคลาสเล็ก นี่คือเหตุผลที่ต้องใช้ macro-F1\n")

    # ---------- SHAP ----------
    add("---\n")
    add("## 2. SHAP — ฟีเจอร์สำคัญของแต่ละ attack\n")
    t = read("03_shap_importance_per_class.csv")
    if t is not None:
        add("ค่า normalize ให้แต่ละคอลัมน์รวมกันได้ 1 แล้ว จึงเทียบข้ามคลาสได้\n")
        for cls in t.columns:
            top = t[cls].nlargest(8).round(4)
            add(f"**{cls}** — " + " · ".join(f"`{f}` {v}" for f, v in top.items()) + "\n")

        counts = pd.Series(0, index=t.index)
        for cls in t.columns:
            counts[t[cls].nlargest(10).index] += 1
        shared = counts[counts > 1].sort_values(ascending=False)
        add("### ฟีเจอร์ที่ติด Top-10 ของหลายคลาส\n")
        add(shared.to_frame("ติด Top-10 กี่คลาส").to_markdown() + "\n")
        add("ตัวที่ติดหลายคลาสคือเหตุผลว่าทำไมการใช้ฟีเจอร์ชุดเดียวกันทุกคลาสถึงใช้ได้\n")

    # ---------- stage 1 ----------
    add("---\n")
    add("## 3. Stage 1 — เทรนด้วยชุดฟีเจอร์ที่ SHAP คัดมา ⭐\n")
    add("**นี่คือจุดที่แนวคิดของอาจารย์ถูกทดสอบ**")
    add("ทุกแถวตรึง recall ≥ 99% เท่ากัน และตรึงโมเดล = XGBoost")
    add("เพื่อให้ตัวแปรที่ต่างกันมีแค่ *ชุดฟีเจอร์* อย่างเดียว\n")
    t = read("04_stage1_comparison.csv", index_col=None)
    if t is not None:
        t = t.drop(columns=["model"], errors="ignore")
        t = t.rename(columns={"latency_ms_per_1k": "latency (ms/1k)",
                              "n_features": "#feat"})
        add(md(t.set_index("feature_set"), dec=6) + "\n")
        add("**อ่านยังไง:** หาแถวที่ `#feat` น้อยที่สุด โดย `recall` ไม่ตกและ `fp_rate` ไม่ขึ้น\n")

    t = read("04_stage1_by_model.csv")
    if t is not None:
        add("### ถ้าเปลี่ยนโมเดล ผลต่างไหม (recall)\n")
        add(md(t) + "\n")

    # ---------- zero-shot ----------
    add("---\n")
    add("## 4. Zero-shot — attack ที่ไม่เคยเห็น\n")
    add("เทรนด้วย OVS → เทสด้วย metasploitable  (U2R ไม่เคยอยู่ใน train เลย)\n")
    t = read("05_zero_shot_by_feature_set.csv", index_col=None)
    if t is not None:
        t = t.rename(columns={"latency_ms_per_1k": "latency (ms/1k)",
                              "n_features": "#feat"})
        add(md(t.set_index("feature_set")) + "\n")

    t = read("05_zero_shot_per_class.csv", index_col=None)
    if t is not None:
        add("### Recall รายคลาส × ชุดฟีเจอร์\n")
        add(md(t.set_index("attack_class")) + "\n")
        add("**อ่านยังไง:** แถว `U2R` คือคลาสที่ไม่เคยเทรน")
        add("ถ้า recall ไม่ตกตอนลดฟีเจอร์ = การลดฟีเจอร์ไม่ทำลายความสามารถ generalize\n")

    # ---------- ผลรอง ----------
    t = read("03_stage2_with_reduced_features.csv")
    if t is not None:
        add("---\n")
        add("## 5. ผลรอง — stage 2 เองก็ลดฟีเจอร์ได้ไหม\n")
        add("ไม่ใช่ประเด็นหลัก แต่ตอบคำถามว่าชุดฟีเจอร์ที่คัดมาใช้กับ stage 2 ได้ด้วยหรือเปล่า\n")
        add(md(t.rename(columns={"latency_ms_per_1k": "latency (ms/1k)"})) + "\n")

    # ---------- ฟีเจอร์ ----------
    t = read("01_feature_summary.csv")
    if t is not None:
        add("---\n")
        add(f"## 6. ฟีเจอร์ที่ใช้ทั้งหมด ({len(t)} ตัว)\n")
        add("<details><summary>กดเพื่อดูรายการเต็ม</summary>\n")
        add(md(t, dec=2) + "\n")
        add("</details>\n")

    path = config.ROOT / "RESULTS.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"[saved] {path}")
    print("เปิดใน VS Code แล้วกด Cmd+Shift+V เพื่อดูแบบตารางสวย ๆ")


if __name__ == "__main__":
    main()
