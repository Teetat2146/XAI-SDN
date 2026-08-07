# ผลการทดลอง — XAI-Guided Two-Stage IDS (InSDN)

> สร้างอัตโนมัติจาก `make_report.py` — อย่าแก้ไฟล์นี้โดยตรง
> ถ้าอยากได้ตัวเลขใหม่ ให้รัน `01`–`05` ใหม่แล้วรัน `make_report.py` อีกที

---

## 1. ชั้นที่ 1 — Binary (Normal vs Attack)

|              |   accuracy |   precision |   recall |      f1 |   latency (ms/1k) |
|:-------------|-----------:|------------:|---------:|--------:|------------------:|
| RandomForest |    0.99994 |     0.99996 |  0.99996 | 0.99996 |           1.11774 |
| XGBoost      |    0.99993 |     0.99995 |  0.99996 | 0.99995 |           0.304   |
| DecisionTree |    0.99988 |     0.99989 |  0.99996 | 0.99993 |           0.17645 |

**อ่านยังไง:** ทั้ง 3 ตัวแทบแยกไม่ออกด้าน F1 → **เลือกจาก latency แทน**
ซึ่งตรงกับที่อาจารย์บอกว่าชั้นแรกต้องเบาและเร็ว

### Threshold sweep

|   threshold |   attack_recall |   ส่งต่อชั้น2_% |   normal_ที่หลุด_% |
|------------:|----------------:|------------:|----------------:|
|        0.05 |          1      |     80.2161 |          0.57   |
|        0.1  |          1      |     80.1506 |          0.2411 |
|        0.2  |          1      |     80.123  |          0.1023 |
|        0.3  |          1      |     80.1186 |          0.0804 |
|        0.5  |          1      |     80.1041 |          0.0219 |
|        0.7  |          0.9999 |     80.0968 |          0.0073 |
|        0.9  |          0.9996 |     80.0736 |          0.0073 |

**อ่านยังไง:** ชั้นแรกไม่ควรใช้เกณฑ์ 0.5 ตามค่าเริ่มต้น
เลือกแถวที่ `attack_recall` สูงพอ (>0.99) แล้วดูว่า `ส่งต่อชั้น2_%` เหลือเท่าไหร่
— ยิ่งน้อยยิ่งดี เพราะแปลว่าชั้นสองทำงานน้อยลง

---

## 2. ชั้นที่ 2 — Multi-class (7 ชนิด attack)

|              |   accuracy |   precision |   recall |     f1 |   latency (ms/1k) |
|:-------------|-----------:|------------:|---------:|-------:|------------------:|
| XGBoost      |     0.9987 |      0.9535 |   0.9664 | 0.9563 |            0.907  |
| DecisionTree |     0.9981 |      0.9303 |   0.9751 | 0.95   |            0.1936 |
| RandomForest |     0.9982 |      0.8884 |   0.8736 | 0.8779 |            1.5446 |

**อ่านยังไง:** accuracy ต่างกันแค่หลักหมื่น แต่ **macro-F1 ต่างกันมาก**
นี่คือหลักฐานว่าทำไมต้องใช้ macro-F1 — RandomForest แพ้เพราะพลาดคลาสเล็ก

---

## 3. เทียบวิธีรวมฟีเจอร์ (หัวใจของงาน)

|                    |   #feat |     f1 |   precision |   recall |   accuracy |   latency (ms/1k) |   % ของ full |
|:-------------------|--------:|-------:|------------:|---------:|-----------:|------------------:|-------------:|
| intersection_top15 |       2 | 0.7637 |      0.9629 |   0.6959 |     0.9713 |            0.9884 |      79.8572 |
| global_mean_top15  |      15 | 0.9377 |      0.9301 |   0.9623 |     0.9985 |            0.8838 |      98.056  |
| union_top15        |      37 | 0.9061 |      0.9041 |   0.9198 |     0.9987 |            0.8069 |      94.7504 |
| dynamic_k          |      40 | 0.9421 |      0.9327 |   0.9679 |     0.9987 |            0.8237 |      98.5189 |
| all_features       |      65 | 0.9563 |      0.9535 |   0.9664 |     0.9987 |            0.9143 |     100      |

**อ่านยังไง:** หาแถวที่ `% ของ full` ใกล้ 100 ที่สุด โดย `#feat` ต่ำที่สุด

### ฟีเจอร์ที่แต่ละวิธีเลือก

| วิธี                 |   จำนวน | ตัวอย่างฟีเจอร์                                                                                                                         |
|:-------------------|--------:|:------------------------------------------------------------------------------------------------------------------------------------|
| all_features       |      65 | Protocol, Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts, TotLen Bwd Pkts, Fwd Pkt Len Max, Fwd Pkt Len Min, … (+57 ตัว) |
| union_top15        |      37 | ACK Flag Cnt, Bwd Header Len, Bwd IAT Max, Bwd IAT Mean, Bwd IAT Min, Bwd IAT Std, Bwd IAT Tot, Bwd Pkt Len Max, … (+29 ตัว)         |
| intersection_top15 |       2 | Flow IAT Min, Init Bwd Win Byts                                                                                                     |
| global_mean_top15  |      15 | Bwd Header Len, Bwd IAT Mean, Bwd IAT Min, Bwd Pkt Len Mean, Flow Duration, Flow IAT Mean, Flow IAT Min, Flow Pkts/s, … (+7 ตัว)     |
| dynamic_k          |      40 | ACK Flag Cnt, Bwd Header Len, Bwd IAT Mean, Bwd IAT Min, Bwd IAT Std, Bwd IAT Tot, Bwd PSH Flags, Bwd Pkt Len Max, … (+32 ตัว)       |

รายชื่อเต็มอยู่ใน `outputs/04_selected_features.csv`

### Top 10 ฟีเจอร์ของแต่ละคลาส (จาก SHAP)

ค่า normalize ให้แต่ละคอลัมน์รวมกันได้ 1 แล้ว จึงเทียบข้ามคลาสได้

**BFA** — `Init Bwd Win Byts` 0.3266 · `Bwd IAT Tot` 0.0699 · `Bwd Pkt Len Max` 0.0655 · `Bwd IAT Mean` 0.0585 · `Flow Pkts/s` 0.0549 · `Flow IAT Mean` 0.0452 · `Fwd Pkt Len Mean` 0.0399 · `Flow Duration` 0.0384 · `Bwd Pkts/s` 0.0379 · `Bwd IAT Std` 0.0337

**BOTNET** — `Fwd Pkts/s` 0.2738 · `Flow IAT Mean` 0.1864 · `Flow Duration` 0.1324 · `Flow IAT Min` 0.0898 · `Bwd IAT Min` 0.06 · `Bwd Header Len` 0.0553 · `Fwd Header Len` 0.0487 · `Init Bwd Win Byts` 0.034 · `Pkt Len Mean` 0.0286 · `Flow IAT Std` 0.0238

**DDoS** — `Protocol` 0.5337 · `Bwd Header Len` 0.2436 · `Bwd IAT Min` 0.0679 · `Flow Pkts/s` 0.0307 · `Bwd IAT Mean` 0.0285 · `Flow Duration` 0.0259 · `Init Bwd Win Byts` 0.0174 · `Bwd IAT Tot` 0.0125 · `Bwd Pkt Len Std` 0.0115 · `Fwd Pkts/s` 0.0093

**DoS** — `Fwd Pkts/s` 0.2898 · `Bwd Header Len` 0.0695 · `Fwd Header Len` 0.0635 · `Init Bwd Win Byts` 0.0609 · `ACK Flag Cnt` 0.0583 · `FIN Flag Cnt` 0.0529 · `Flow IAT Min` 0.0496 · `Flow Duration` 0.0356 · `Flow IAT Std` 0.0295 · `Pkt Len Mean` 0.0279

**Probe** — `Bwd Header Len` 0.297 · `Protocol` 0.1171 · `Pkt Len Mean` 0.0863 · `Fwd Pkt Len Std` 0.0527 · `TotLen Fwd Pkts` 0.0353 · `Bwd Pkts/s` 0.0343 · `Flow IAT Min` 0.0341 · `Flow Duration` 0.0327 · `Init Bwd Win Byts` 0.0291 · `Flow IAT Max` 0.022

**U2R** — `Bwd Pkt Len Mean` 0.2227 · `TotLen Fwd Pkts` 0.1068 · `Fwd Pkts/s` 0.0774 · `Fwd Pkt Len Std` 0.0651 · `Bwd IAT Min` 0.0639 · `Flow IAT Min` 0.0529 · `Flow Byts/s` 0.0502 · `Init Bwd Win Byts` 0.0401 · `Pkt Len Std` 0.0386 · `Fwd IAT Min` 0.027

**Web-Attack** — `Init Bwd Win Byts` 0.1014 · `Fwd Pkt Len Max` 0.0985 · `Bwd Header Len` 0.0912 · `Bwd IAT Mean` 0.0826 · `Flow IAT Min` 0.0744 · `Fwd Header Len` 0.066 · `Fwd IAT Min` 0.0551 · `Tot Bwd Pkts` 0.0508 · `Flow IAT Max` 0.0471 · `Bwd IAT Min` 0.0415

### ฟีเจอร์ที่ติด Top-10 ของหลายคลาส

|                   |   ติด Top-10 กี่คลาส |
|:------------------|------------------:|
| Init Bwd Win Byts |                 7 |
| Bwd Header Len    |                 5 |
| Flow IAT Min      |                 5 |
| Flow Duration     |                 5 |
| Fwd Pkts/s        |                 4 |
| Bwd IAT Min       |                 4 |
| Bwd IAT Mean      |                 3 |
| Pkt Len Mean      |                 3 |
| Fwd Header Len    |                 3 |
| Bwd Pkts/s        |                 2 |
| Protocol          |                 2 |
| Bwd IAT Tot       |                 2 |
| Flow IAT Max      |                 2 |
| Flow IAT Std      |                 2 |
| Flow IAT Mean     |                 2 |
| Flow Pkts/s       |                 2 |
| Fwd Pkt Len Std   |                 2 |
| TotLen Fwd Pkts   |                 2 |
| Fwd IAT Min       |                 2 |

ตัวที่ติดหลายคลาสคือเหตุผลว่าทำไม `global_mean` ถึงใช้ได้ดี

---

## 4. Zero-shot — attack ที่ไม่เคยเห็น

|              |   accuracy |   precision |   recall |     f1 |   latency (ms/1k) |
|:-------------|-----------:|------------:|---------:|-------:|------------------:|
| XGBoost      |     0.9979 |      0.9999 |   0.9975 | 0.9987 |            0.2045 |
| RandomForest |     0.9974 |      0.9999 |   0.9968 | 0.9984 |            0.7398 |
| DecisionTree |     0.9925 |      0.9999 |   0.9908 | 0.9953 |            0.1329 |

### Recall รายคลาส

| attack_class   | เคยเห็นตอนเทรน   |   n_test |   recall |
|:---------------|:----------------|---------:|---------:|
| DoS            | ใช่              |     1145 |   0.7397 |
| U2R            | ไม่ (zero-shot)  |       17 |   0.9412 |
| Probe          | ใช่              |    61757 |   0.9994 |
| DDoS           | ใช่              |    73529 |   1      |
| BFA            | ใช่              |      295 |   1      |

**อ่านยังไง:** แถวที่ `เคยเห็นตอนเทรน` = `ไม่ (zero-shot)` คือตัวเลขที่ตอบอาจารย์ได้ตรงที่สุด
ว่าโมเดลจับ attack รูปแบบใหม่ได้จริงไหม

---

## 5. ฟีเจอร์ที่ใช้ทั้งหมด (65 ตัว)

<details><summary>กดเพื่อดูรายการเต็ม</summary>

|                   |   n_unique |             mean |              std |
|:------------------|-----------:|-----------------:|-----------------:|
| Protocol          |          3 |      4.96        |      4.86        |
| Flow Duration     |      85380 |      6.73717e+06 |      2.18335e+07 |
| Tot Fwd Pkts      |        569 |      6.16        |   1554.17        |
| Tot Bwd Pkts      |        699 |      6.12        |    105.86        |
| TotLen Fwd Pkts   |       5602 |    731.06        |  69652.9         |
| TotLen Bwd Pkts   |       7585 |   8335.01        | 342972           |
| Fwd Pkt Len Max   |       2132 |    115.69        |    666.52        |
| Fwd Pkt Len Min   |         76 |      4.45        |     31.29        |
| Fwd Pkt Len Mean  |       9704 |     43.71        |    267.93        |
| Fwd Pkt Len Std   |      11888 |     50.51        |    332.04        |
| Bwd Pkt Len Max   |       2988 |    409.16        |   2859.43        |
| Bwd Pkt Len Min   |         64 |      5.2         |     21.23        |
| Bwd Pkt Len Mean  |      10858 |     86.25        |    352.24        |
| Bwd Pkt Len Std   |      13313 |    113.7         |    501.17        |
| Flow Byts/s       |     117642 |  74539.6         | 662731           |
| Flow Pkts/s       |     103315 | 247306           | 565541           |
| Flow IAT Mean     |     104184 | 569523           |      2.13601e+06 |
| Flow IAT Std      |     110452 |      1.57295e+06 |      5.40199e+06 |
| Flow IAT Max      |      48508 |      4.94018e+06 |      1.64159e+07 |
| Flow IAT Min      |      18156 |  18316.9         | 902395           |
| Fwd IAT Tot       |      39965 |      2.93579e+06 |      1.64066e+07 |
| Fwd IAT Mean      |      52708 | 290420           |      2.07335e+06 |
| Fwd IAT Std       |      50639 | 461643           |      3.15685e+06 |
| Fwd IAT Max       |      39042 |      1.22979e+06 |      7.44197e+06 |
| Fwd IAT Min       |      16164 |  25047.2         | 722554           |
| Bwd IAT Tot       |      48842 |      6.46148e+06 |      2.1391e+07  |
| Bwd IAT Mean      |      70930 | 893607           |      3.3283e+06  |
| Bwd IAT Std       |      79152 |      1.9517e+06  |      6.85853e+06 |
| Bwd IAT Max       |      44259 |      4.86391e+06 |      1.63851e+07 |
| Bwd IAT Min       |      16693 |  22611.8         |      1.11783e+06 |
| Bwd PSH Flags     |          2 |      0.05        |      0.23        |
| Bwd URG Flags     |          2 |      0.01        |      0.11        |
| Fwd Header Len    |        665 |     74.87        |   1201.22        |
| Bwd Header Len    |       1057 |    125.53        |   2195.5         |
| Fwd Pkts/s        |      88034 |   2201           |  34342.2         |
| Bwd Pkts/s        |     100285 | 245105           | 563380           |
| Pkt Len Min       |         51 |      4.22        |     14.63        |
| Pkt Len Max       |       3279 |    461           |   2919.46        |
| Pkt Len Mean      |      14691 |     67.53        |    242.57        |
| Pkt Len Std       |      21336 |    114.76        |    476.03        |
| Pkt Len Var       |      20867 | 239777           |      5.50268e+06 |
| FIN Flag Cnt      |          2 |      0.14        |      0.34        |
| SYN Flag Cnt      |          2 |      0.24        |      0.43        |
| RST Flag Cnt      |          2 |      0           |      0.03        |
| PSH Flag Cnt      |          2 |      0.05        |      0.23        |
| ACK Flag Cnt      |          2 |      0.27        |      0.45        |
| URG Flag Cnt      |          2 |      0.01        |      0.11        |
| Down/Up Ratio     |         10 |      0.72        |      1.08        |
| Pkt Size Avg      |      14491 |     75.42        |    258.44        |
| Fwd Seg Size Avg  |       9704 |     43.71        |    267.93        |
| Bwd Seg Size Avg  |      10858 |     86.25        |    352.24        |
| Subflow Fwd Pkts  |        569 |      6.16        |   1554.17        |
| Subflow Fwd Byts  |       5602 |    731.08        |  69634           |
| Subflow Bwd Pkts  |        699 |      6.12        |    105.86        |
| Subflow Bwd Byts  |       7589 |   8337.21        | 343424           |
| Init Bwd Win Byts |        227 |   5751.59        |  18170.8         |
| Fwd Act Data Pkts |        289 |      1.44        |     39.47        |
| Active Mean       |      19103 |  63851           | 785272           |
| Active Std        |       8559 |  31053.5         | 503560           |
| Active Max        |      18380 | 108638           |      1.2206e+06  |
| Active Min        |      16183 |  41855.8         | 667230           |
| Idle Mean         |       5780 |      4.70817e+06 |      1.61611e+07 |
| Idle Std          |       8045 | 144361           |      1.65349e+06 |
| Idle Max          |       4179 |      4.84396e+06 |      1.64161e+07 |
| Idle Min          |       7903 |      4.58151e+06 |      1.60456e+07 |

</details>
