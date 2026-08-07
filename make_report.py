"""
make_report.py — รวมผลลัพธ์ทุกอันใน outputs/ เป็นรายงานเดียวที่อ่านง่าย

สร้าง RESULTS.md ที่เปิดใน VS Code แล้วกด Cmd+Shift+V จะเห็นเป็นตารางสวย ๆ
เอาไปวางในรายงาน/สไลด์ได้เลย

ไฟล์นี้ไม่คำนวณอะไรใหม่ แค่อ่าน CSV ที่สคริปต์ 01-05 สร้างไว้แล้วมาจัดรูปแบบ
ดังนั้นรันกี่ครั้งก็ได้ ไม่กระทบผลการทดลอง

รัน:  python make_report.py   (ต้องรัน 01-05 ให้ครบก่อน)
"""
import pandas as pd

import config

# ปิดการตัดคอลัมน์ตอน print
pd.set_option("display.width", 200)


def read(name):
    """อ่าน CSV จาก outputs/ — คืน None ถ้ายังไม่มีไฟล์ (ยังไม่ได้รันสคริปต์นั้น)"""
    path = config.OUT_DIR / name
    return pd.read_csv(path, index_col=0) if path.exists() else None


def md_table(df, floatfmt=4) -> str:
    """แปลง DataFrame เป็นตาราง markdown พร้อมปัดทศนิยม"""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(floatfmt)
    return d.to_markdown()


def main():
    out = []
    add = out.append

    add("# ผลการทดลอง — XAI-Guided Two-Stage IDS (InSDN)\n")
    add("> สร้างอัตโนมัติจาก `make_report.py` — อย่าแก้ไฟล์นี้โดยตรง")
    add("> ถ้าอยากได้ตัวเลขใหม่ ให้รัน `01`–`05` ใหม่แล้วรัน `make_report.py` อีกที\n")
    add("---\n")

    # ---------- ชั้นที่ 1 ----------
    add("## 1. ชั้นที่ 1 — Binary (Normal vs Attack)\n")
    t = read("02_binary_comparison.csv")
    if t is not None:
        t = t.rename(columns={"latency_ms_per_1k": "latency (ms/1k)"})
        # binary ได้ค่าเกือบ 1.0 หมด ต้องใช้ทศนิยม 5 ตำแหน่งถึงจะเห็นความต่าง
        add(md_table(t, floatfmt=5) + "\n")
        add("**อ่านยังไง:** ทั้ง 3 ตัวแทบแยกไม่ออกด้าน F1 → **เลือกจาก latency แทน**")
        add("ซึ่งตรงกับที่อาจารย์บอกว่าชั้นแรกต้องเบาและเร็ว\n")

    t = read("02_threshold_sweep.csv")
    if t is not None:
        add("### Threshold sweep\n")
        add(md_table(t.reset_index().set_index("threshold")) + "\n")
        add("**อ่านยังไง:** ชั้นแรกไม่ควรใช้เกณฑ์ 0.5 ตามค่าเริ่มต้น")
        add("เลือกแถวที่ `attack_recall` สูงพอ (>0.99) แล้วดูว่า `ส่งต่อชั้น2_%` เหลือเท่าไหร่")
        add("— ยิ่งน้อยยิ่งดี เพราะแปลว่าชั้นสองทำงานน้อยลง\n")

    # ---------- ชั้นที่ 2 ----------
    add("---\n")
    add("## 2. ชั้นที่ 2 — Multi-class (7 ชนิด attack)\n")
    t = read("03_multiclass_comparison.csv")
    if t is not None:
        t = t.rename(columns={"latency_ms_per_1k": "latency (ms/1k)"})
        add(md_table(t) + "\n")
        add("**อ่านยังไง:** accuracy ต่างกันแค่หลักหมื่น แต่ **macro-F1 ต่างกันมาก**")
        add("นี่คือหลักฐานว่าทำไมต้องใช้ macro-F1 — RandomForest แพ้เพราะพลาดคลาสเล็ก\n")

    # ---------- feature selection ----------
    add("---\n")
    add("## 3. เทียบวิธีรวมฟีเจอร์ (หัวใจของงาน)\n")
    t = read("04_feature_selection_comparison.csv")
    if t is not None:
        t = t.rename(columns={
            "n_features": "#feat",
            "latency_ms_per_1k": "latency (ms/1k)",
            "f1_เทียบ_full_%": "% ของ full",
        })
        t["#feat"] = t["#feat"].astype(int)
        add(md_table(t) + "\n")
        add("**อ่านยังไง:** หาแถวที่ `% ของ full` ใกล้ 100 ที่สุด โดย `#feat` ต่ำที่สุด\n")

    t = read("04_selected_features.csv")
    if t is not None:
        add("### ฟีเจอร์ที่แต่ละวิธีเลือก\n")
        rows = []
        for method, val in t.iloc[:, 0].items():
            feats = [f.strip() for f in str(val).split(",")]
            preview = ", ".join(feats[:8])
            if len(feats) > 8:
                preview += f", … (+{len(feats) - 8} ตัว)"
            rows.append({"วิธี": method, "จำนวน": len(feats), "ตัวอย่างฟีเจอร์": preview})
        add(pd.DataFrame(rows).set_index("วิธี").to_markdown() + "\n")
        add("รายชื่อเต็มอยู่ใน `outputs/04_selected_features.csv`\n")

    # ---------- SHAP ----------
    t = read("04_shap_importance_per_class.csv")
    if t is not None:
        add("### Top 10 ฟีเจอร์ของแต่ละคลาส (จาก SHAP)\n")
        add("ค่า normalize ให้แต่ละคอลัมน์รวมกันได้ 1 แล้ว จึงเทียบข้ามคลาสได้\n")
        for cls in t.columns:
            top = t[cls].nlargest(10).round(4)
            line = " · ".join(f"`{f}` {v}" for f, v in top.items())
            add(f"**{cls}** — {line}\n")

        add("### ฟีเจอร์ที่ติด Top-10 ของหลายคลาส\n")
        counts = pd.Series(0, index=t.index)
        for cls in t.columns:
            counts[t[cls].nlargest(10).index] += 1
        shared = counts[counts > 1].sort_values(ascending=False)
        add(shared.to_frame("ติด Top-10 กี่คลาส").to_markdown() + "\n")
        add("ตัวที่ติดหลายคลาสคือเหตุผลว่าทำไม `global_mean` ถึงใช้ได้ดี\n")

    # ---------- zero-shot ----------
    add("---\n")
    add("## 4. Zero-shot — attack ที่ไม่เคยเห็น\n")
    t = read("05_zero_shot_comparison.csv")
    if t is not None:
        t = t.rename(columns={"latency_ms_per_1k": "latency (ms/1k)"})
        add(md_table(t) + "\n")

    t = read("05_zero_shot_per_class.csv")
    if t is not None:
        t = t.reset_index()
        if "attack_class" in t.columns:
            t = t.set_index("attack_class")
        add("### Recall รายคลาส\n")
        add(md_table(t) + "\n")
        add("**อ่านยังไง:** แถวที่ `เคยเห็นตอนเทรน` = `ไม่ (zero-shot)` คือตัวเลขที่ตอบอาจารย์ได้ตรงที่สุด")
        add("ว่าโมเดลจับ attack รูปแบบใหม่ได้จริงไหม\n")

    # ---------- ฟีเจอร์ทั้งหมด ----------
    t = read("01_feature_summary.csv")
    if t is not None:
        add("---\n")
        add(f"## 5. ฟีเจอร์ที่ใช้ทั้งหมด ({len(t)} ตัว)\n")
        add("<details><summary>กดเพื่อดูรายการเต็ม</summary>\n")
        add(md_table(t, floatfmt=2) + "\n")
        add("</details>\n")

    path = config.ROOT / "RESULTS.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"[saved] {path}")
    print("เปิดใน VS Code แล้วกด Cmd+Shift+V เพื่อดูแบบตารางสวย ๆ")


if __name__ == "__main__":
    main()
