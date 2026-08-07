# XAI-Guided Two-Stage IDS สำหรับ SDN

ใช้ SHAP เลือกฟีเจอร์ แล้วส่งเข้า pipeline 2 ชั้น (binary filter → multi-class classifier)
บน dataset **InSDN**

---

## วิธีรัน

```bash
cd ~/Desktop/XAI-SDN

./.venv/bin/python 01_prepare_data.py       # เตรียมข้อมูล  (~30 วิ)
./.venv/bin/python 02_train_binary.py       # ชั้นที่ 1
./.venv/bin/python 03_train_multiclass.py   # ชั้นที่ 2
./.venv/bin/python 04_feature_selection.py  # SHAP + เทียบวิธีรวมฟีเจอร์  ← หัวใจของงาน
./.venv/bin/python 05_zero_shot.py          # attack ที่ไม่เคยเห็น
```

กราฟ SHAP อยู่ใน `shap_explore.ipynb` — เปิดใน VS Code แล้วเลือก kernel เป็น `.venv`
notebook ไม่เทรนอะไรใหม่ แค่โหลดโมเดลจาก `models/` มาเล่นกราฟ

ผลลัพธ์ทั้งหมดเป็น CSV ใน `outputs/` เอาไปวางในรายงานได้เลย

---

## Dataset

InSDN — อ่านจาก `~/Downloads/` แบบ read-only ไม่แตะต้นฉบับ

| ไฟล์ | แถว | เนื้อหา |
|---|---:|---|
| `Normal_data.csv` | 68,424 | traffic ปกติล้วน |
| `OVS.csv` | 138,722 | DoS, DDoS, Probe, BFA, Web-Attack, BOTNET |
| `metasploitable-2.csv` | 136,743 | DDoS, Probe, DoS, BFA, U2R |

รวม **343,889 แถว, 7 ชนิด attack + Normal**

### Class imbalance

| Attack | จำนวน |
|---|---:|
| DDoS | 121,942 |
| Probe | 98,129 |
| DoS | 53,616 |
| BFA | 1,405 |
| Web-Attack | 192 |
| BOTNET | 164 |
| **U2R** | **17** |

ต่างกัน **7,173 เท่า** → ใช้ **Macro-F1 เป็น metric หลัก ไม่ใช่ accuracy**
เพราะ accuracy จะถูกกลบด้วย DDoS/Probe จนมองไม่เห็นว่าคลาสเล็กพัง

---

## Data leakage — จัดการยังไง

### 1. ตัดคอลัมน์ identifier ออก 6 ตัว

| คอลัมน์ | เหตุผล |
|---|---|
| `Flow ID` | string ประกอบจาก IP+Port+Protocol = ยัดคำตอบให้โมเดลตรง ๆ |
| `Src IP` | เครื่องที่ยิง attack ใน testbed มี IP ตายตัว โมเดลจะจำ IP แทนพฤติกรรม |
| `Dst IP` | เครื่องเหยื่อ (metasploitable) ใช้ IP เดิมตลอด |
| `Src Port` | ephemeral port ผูกกับ session ที่ยิง |
| `Dst Port` | ใน testbed ผูกกับชนิด attack เกือบ 1:1 |
| `Timestamp` | attack ยิงเป็นช่วงเวลา โมเดลจะจำ "ช่วงนี้คือ attack" |

เก็บ `Protocol` ไว้ เพราะเป็น protocol number จริง (6/17/1) ไม่ใช่ label ที่แปลงมา

### 2. ตัดคอลัมน์ที่เป็นค่าคงที่ 12 ตัว

CICFlowMeter มีบั๊กที่คอลัมน์กลุ่ม bulk ออกมาเป็น 0 ทั้งคอลัมน์:
`Fwd/Bwd Byts/b Avg`, `Fwd/Bwd Pkts/b Avg`, `Fwd/Bwd Blk Rate Avg`,
`Fwd PSH Flags`, `Fwd URG Flags`, `CWE Flag Count`, `ECE Flag Cnt`,
`Init Fwd Win Byts`, `Fwd Seg Size Min`

ถ้าไม่ตัด SHAP จะรายงานว่า "ไม่สำคัญ" ซึ่งเป็นข้อสรุปที่ไร้ความหมาย

**เหลือฟีเจอร์ใช้งานจริง 65 ตัว** (จาก 84)

### 3. บั๊กของ dataset ที่เจอเอง

`OVS.csv` เขียน label ว่า `'DDoS '` (มีเว้นวรรคท้าย) แต่ `metasploitable-2.csv` เขียน `'DDoS'`
ถ้าไม่ `.strip()` จะกลายเป็น 8 คลาสแทนที่จะเป็น 7 และ zero-shot จะเข้าใจผิดว่า
DDoS ของ metasploitable เป็นคลาสที่ไม่เคยเห็น ทั้งที่เทรนไปแล้ว

---

## ผลลัพธ์

### ชั้นที่ 1 — Binary (Normal vs Attack)

| Model | Accuracy | F1 | Latency (ms/1k flows) |
|---|---:|---:|---:|
| RandomForest | 0.99994 | 0.99996 | 1.11 |
| XGBoost | 0.99993 | 0.99996 | 0.32 |
| DecisionTree | 0.99988 | 0.99993 | 0.18 |

ทั้งสามตัวแทบแยกไม่ออก → **เลือกจาก latency ไม่ใช่ F1** ซึ่งตรงกับที่อาจารย์บอกว่า
ชั้นแรกต้องเบาและเร็ว XGBoost เร็วกว่า RF **3.5 เท่า** โดย F1 เท่ากัน

> ⚠️ ตัวเลขนี้มาจาก random split ภายใน capture เดียวกัน จึงสูงเกินจริง
> ตัวเลขที่เชื่อถือได้จริงอยู่ในหัวข้อ zero-shot ด้านล่าง

### ชั้นที่ 2 — Multi-class (7 ชนิด attack)

| Model | Accuracy | Macro-F1 | Latency |
|---|---:|---:|---:|
| **XGBoost** | 0.9987 | **0.9563** | 0.93 |
| DecisionTree | 0.9981 | 0.9500 | 0.33 |
| RandomForest | 0.9982 | 0.8779 | 1.54 |

สังเกตว่า accuracy ต่างกันแค่ 0.0006 แต่ **Macro-F1 ต่างกันถึง 0.08** —
นี่คือหลักฐานว่าทำไมต้องใช้ Macro-F1 RandomForest แพ้เพราะพลาดคลาสเล็ก

per-class F1 ของ XGBoost: DDoS 1.000 / DoS 0.999 / Probe 0.998 / Web-Attack 0.987 /
BOTNET 0.955 / **BFA 0.897** / **U2R 0.857** (U2R มี test แค่ 3 แถว — ไม่มีนัยสำคัญ)

BFA พลาดเพราะสับสนกับ Probe (48 จาก 281 แถว) ซึ่งสมเหตุสมผล เพราะ brute-force
กับ port scan มีพฤติกรรม connection ถี่ ๆ คล้ายกัน

### หัวใจของงาน — เทียบ 4 วิธีรวมฟีเจอร์

| วิธี | #features | Macro-F1 | % ของ full | Latency |
|---|---:|---:|---:|---:|
| intersection_top15 | 2 | 0.7637 | 79.9% | 1.00 |
| **global_mean_top15** | **15** | **0.9377** | **98.1%** | **0.81** |
| union_top15 | 37 | 0.9061 | 94.8% | 0.97 |
| dynamic_k | 40 | 0.9421 | 98.5% | 0.87 |
| all_features | 65 | 0.9563 | 100% | 0.91 |

**ข้อสรุป — ตอบคำถามที่อาจารย์ตั้งไว้โดยตรง:**

1. **สมมติฐานของอาจารย์ถูก** — อาจารย์ให้ลองว่า "ถ้าใช้ฟีเจอร์เดียวกันทุกคลาส ผลจะเปลี่ยนไหม"
   คำตอบคือ `global_mean_top15` ใช้แค่ **15 ฟีเจอร์** ได้ **98.1%** ของ full-feature
   → **ไม่จำเป็นต้องแยกฟีเจอร์ต่อคลาส**

2. **Union แย่กว่าที่คิด** — ใช้ฟีเจอร์มากกว่า 2.5 เท่า (37 vs 15) แต่ F1 **ต่ำกว่า**
   (0.906 vs 0.938) ยืนยันความกังวลของอาจารย์ว่าฟีเจอร์บวมแล้วไม่ได้ดีขึ้น
   สาเหตุคือฟีเจอร์ที่สำคัญเฉพาะคลาสเล็กกลายเป็น noise สำหรับคลาสใหญ่

3. **Intersection ใช้ไม่ได้** — เหลือแค่ 2 ฟีเจอร์ที่ทุกคลาสเห็นตรงกัน F1 ร่วงเหลือ 79.9%

4. **Dynamic-K ได้ F1 สูงสุด (98.5%) แต่ไม่คุ้ม** — ใช้ 40 ฟีเจอร์เพื่อแลกกับ F1
   ที่ดีกว่า global_mean แค่ 0.4% → **global_mean_top15 คุ้มที่สุด**

Dynamic-K เลือกจำนวนฟีเจอร์ต่างกันตามคลาสจริง (DDoS ใช้แค่ 5, DoS/Probe ใช้ 21)
ซึ่งยืนยันว่าแต่ละคลาสมี "ความซับซ้อน" ไม่เท่ากัน

### Zero-shot — attack ที่ไม่เคยเห็น

เทรนบน Normal(ครึ่ง) + OVS → เทสบน Normal(อีกครึ่ง) + metasploitable
U2R มีเฉพาะใน metasploitable = คลาสที่ไม่เคยเห็นเลย

| Attack | เคยเห็น | n_test | Recall |
|---|---|---:|---:|
| BFA | ใช่ | 295 | 1.000 |
| DDoS | ใช่ | 73,529 | 0.99997 |
| Probe | ใช่ | 61,757 | 0.9994 |
| **U2R** | **ไม่ (zero-shot)** | **17** | **0.941** |
| DoS | ใช่ | 1,145 | 0.740 |

**False positive rate บน Normal: 0.05%**

ผลนี้แข็งแรงมากสำหรับเอาไปตอบอาจารย์: โมเดล flag **U2R ที่ไม่เคยเห็นเลยได้ 94.1%**
(16 จาก 17 แถว) โดย FP แค่ 0.05% → ยืนยันสมมติฐานว่าชั้นแรกบอกได้ว่า "น่าสงสัย"
แม้จะไม่รู้จักชนิด attack

**จุดที่น่าสนใจกว่า:** `DoS` ที่ **เคยเห็นตอนเทรน** กลับได้ recall แค่ 0.740 ต่ำกว่า
U2R ที่ไม่เคยเห็น แปลว่า DoS ใน metasploitable มีลักษณะต่างจาก DoS ใน OVS มาก
→ **ปัญหาไม่ใช่ "เคยเห็นหรือไม่เคยเห็น" แต่เป็นการ generalize ข้ามสภาพแวดล้อม**
อันนี้เป็นประเด็นที่เอาไปเขียน discussion ในรายงานได้ดี

---

## เตรียมตอบอาจารย์

**"ทำไม accuracy สูง 99% มั่นใจได้ยังไงว่าไม่ leak"**
ตัดคอลัมน์ identifier 6 ตัวออกแล้ว (ตารางด้านบน) และทดสอบแบบ cross-environment
คือเทรนบน OVS เทสบน metasploitable ซึ่ง IP และสภาพแวดล้อมคนละชุด — ผลที่ได้
ยังสูงอยู่ ยกเว้น DoS ที่ร่วงเหลือ 0.74 ซึ่งแสดงว่าการทดสอบนี้ท้าทายจริง ไม่ได้ตอบง่าย

**"ทำไม XGBoost/RF ไม่ใช้ Decision Tree เฉย ๆ"**
`feature_importances_` ของ ensemble เป็น Gini importance ซึ่ง bias เข้าหาฟีเจอร์ที่มี
cardinality สูง และไม่บอกทิศทางว่าค่าสูงหรือต่ำทำให้เป็น attack — SHAP แก้ได้ทั้งสองข้อ
และผลจริงคือ RandomForest ได้ Macro-F1 แค่ 0.878 แพ้ DecisionTree (0.950) ด้วยซ้ำ
เพราะพลาดคลาสเล็ก → เลือก XGBoost จากผลจริงไม่ใช่เลือกลอย ๆ

**"จะรวมฟีเจอร์ต่างคลาสยังไง"**
ทำ 4 วิธีเทียบเป็นตาราง คำตอบคือ global_mean_top15 — ใช้ฟีเจอร์เดียวกันทุกคลาส
15 ตัว ได้ 98.1% ของ full-feature ไม่ต้องแยกต่อคลาส

**"ทำไมต้องลดฟีเจอร์ ในเมื่อใช้ครบก็ดีอยู่แล้ว"**
เพราะชั้นแรกต้องรันกับ traffic ที่วิ่งเข้ามาตลอดเวลา ฟีเจอร์กลุ่ม Idle/Active statistics
ต้องรอ flow จบก่อนถึงคำนวณได้ ไม่เหมาะกับการกรอง real-time — ถ้า SHAP ชี้ว่าตัดได้
โดย F1 ไม่ตก ก็เป็นประโยชน์ที่วัดเป็นตัวเลขได้ ไม่ใช่แค่ "อธิบายได้"

**"คลาสที่มีข้อมูลน้อยมากจะทำยังไง"**
U2R มี 17 แถว เทรนไปก็ไม่มีนัยสำคัญทางสถิติ (test set เหลือ 3 แถว) จึงใช้เป็น
zero-shot class แทน ซึ่งได้ผลดีกว่า — recall 94.1% โดยไม่ต้องเทรนเลย

---

## โครงสร้างไฟล์

```
XAI-SDN/
├── config.py                 ค่าคงที่ทั้งหมด แก้ที่นี่ที่เดียว
├── common.py                 ฟังก์ชันใช้ร่วม
├── 01_prepare_data.py        รวม 3 CSV + ตัด leak + ตัด constant
├── 02_train_binary.py        ชั้นที่ 1 + threshold sweep
├── 03_train_multiclass.py    ชั้นที่ 2
├── 04_feature_selection.py   SHAP ต่อคลาส + เทียบ 4 วิธี  ← หัวใจ
├── 05_zero_shot.py           cross-environment test
├── make_report.py            รวมผลทุกอันเป็น RESULTS.md
├── shap_explore.ipynb        กราฟ SHAP + วิธีอ่านกราฟ
├── RESULTS.md                รายงานผล (สร้างอัตโนมัติ กด Cmd+Shift+V ดู)
├── data/
│   ├── raw/                  CSV ต้นฉบับ InSDN 3 ไฟล์ (สำเนา)
│   └── insdn_clean.parquet   ข้อมูลที่ clean+รวมแล้ว
├── models/                   โมเดล .pkl
├── outputs/                  ตาราง CSV ดิบ
└── figures/                  กราฟ .png
```

**โปรเจกต์ครบในตัวเอง** — ก๊อปโฟลเดอร์นี้ไปเครื่องอื่น สร้าง venv ใหม่
(`pip install -r requirements.txt`) แล้วรันได้ทันที ไม่ต้องไปหา dataset มาวางที่ไหน

---

## ยังไม่ได้ทำ

- **วัด pipeline แบบ end-to-end** — ตอนนี้ชั้น 1 กับชั้น 2 วัดแยกกัน ยังไม่ได้วัดว่า
  พอต่อกันจริง (ชั้น 2 รับเฉพาะสิ่งที่หลุดชั้น 1 มา) Macro-F1 จะตกเท่าไหร่จาก error สะสม
- **ตรวจ group leakage** — ยังไม่ได้ลอง split ตาม flow/session ภายในไฟล์เดียวกัน
- **จับเวลาการคำนวณฟีเจอร์จริง** — ตอนนี้วัดแค่ latency ของโมเดล ยังไม่ได้วัดว่า
  การคำนวณฟีเจอร์แต่ละตัวจาก Controller ใช้เวลาเท่าไหร่ ซึ่งเป็นข้ออ้างหลักของการลดฟีเจอร์
