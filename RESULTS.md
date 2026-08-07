# ผลการทดลอง — XAI-Guided Two-Stage IDS (InSDN)

> สร้างอัตโนมัติจาก `make_report.py` — อย่าแก้ไฟล์นี้โดยตรง

## ลำดับของงาน

```
01 เตรียมข้อมูล
  → 02 เทรน stage 2 (multi-class)
      → 03 SHAP หาฟีเจอร์สำคัญของแต่ละ attack แล้วรวมเป็นชุด
          → 04 เอาชุดฟีเจอร์ไปเทรน stage 1  ← จุดที่พิสูจน์แนวคิด
          → 05 ทดสอบ attack ที่ไม่เคยเห็น
```

**ทำไม stage 2 มาก่อน stage 1:** เพราะต้องใช้ SHAP จาก stage 2
มาคัดฟีเจอร์ให้ stage 1 ใช้ — stage 1 จึงเบาลงได้อย่างมีหลักฐานรองรับ

---

## 1. Stage 2 — Multi-class (7 ชนิด attack)

|              |   accuracy |   precision |   recall |     f1 |   latency (ms/1k) |
|:-------------|-----------:|------------:|---------:|-------:|------------------:|
| XGBoost      |     0.9987 |      0.9535 |   0.9664 | 0.9563 |            0.9332 |
| DecisionTree |     0.9981 |      0.9303 |   0.9751 | 0.95   |            0.2951 |
| RandomForest |     0.9982 |      0.8884 |   0.8736 | 0.8779 |            1.5089 |

**อ่านยังไง:** accuracy ต่างกันแค่หลักหมื่น แต่ **macro-F1 ต่างกันมาก**
— RandomForest แพ้เพราะพลาดคลาสเล็ก นี่คือเหตุผลที่ต้องใช้ macro-F1

---

## 2. SHAP — ฟีเจอร์สำคัญของแต่ละ attack

ค่า normalize ให้แต่ละคอลัมน์รวมกันได้ 1 แล้ว จึงเทียบข้ามคลาสได้

**BFA** — `Init Bwd Win Byts` 0.3266 · `Bwd IAT Tot` 0.0699 · `Bwd Pkt Len Max` 0.0655 · `Bwd IAT Mean` 0.0585 · `Flow Pkts/s` 0.0549 · `Flow IAT Mean` 0.0452 · `Fwd Pkt Len Mean` 0.0399 · `Flow Duration` 0.0384

**BOTNET** — `Fwd Pkts/s` 0.2738 · `Flow IAT Mean` 0.1864 · `Flow Duration` 0.1324 · `Flow IAT Min` 0.0898 · `Bwd IAT Min` 0.06 · `Bwd Header Len` 0.0553 · `Fwd Header Len` 0.0487 · `Init Bwd Win Byts` 0.034

**DDoS** — `Protocol` 0.5337 · `Bwd Header Len` 0.2436 · `Bwd IAT Min` 0.0679 · `Flow Pkts/s` 0.0307 · `Bwd IAT Mean` 0.0285 · `Flow Duration` 0.0259 · `Init Bwd Win Byts` 0.0174 · `Bwd IAT Tot` 0.0125

**DoS** — `Fwd Pkts/s` 0.2898 · `Bwd Header Len` 0.0695 · `Fwd Header Len` 0.0635 · `Init Bwd Win Byts` 0.0609 · `ACK Flag Cnt` 0.0583 · `FIN Flag Cnt` 0.0529 · `Flow IAT Min` 0.0496 · `Flow Duration` 0.0356

**Probe** — `Bwd Header Len` 0.297 · `Protocol` 0.1171 · `Pkt Len Mean` 0.0863 · `Fwd Pkt Len Std` 0.0527 · `TotLen Fwd Pkts` 0.0353 · `Bwd Pkts/s` 0.0343 · `Flow IAT Min` 0.0341 · `Flow Duration` 0.0327

**U2R** — `Bwd Pkt Len Mean` 0.2227 · `TotLen Fwd Pkts` 0.1068 · `Fwd Pkts/s` 0.0774 · `Fwd Pkt Len Std` 0.0651 · `Bwd IAT Min` 0.0639 · `Flow IAT Min` 0.0529 · `Flow Byts/s` 0.0502 · `Init Bwd Win Byts` 0.0401

**Web-Attack** — `Init Bwd Win Byts` 0.1014 · `Fwd Pkt Len Max` 0.0985 · `Bwd Header Len` 0.0912 · `Bwd IAT Mean` 0.0826 · `Flow IAT Min` 0.0744 · `Fwd Header Len` 0.066 · `Fwd IAT Min` 0.0551 · `Tot Bwd Pkts` 0.0508

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

ตัวที่ติดหลายคลาสคือเหตุผลว่าทำไมการใช้ฟีเจอร์ชุดเดียวกันทุกคลาสถึงใช้ได้

---

## 3. Stage 1 — เทรนด้วยชุดฟีเจอร์ที่ SHAP คัดมา ⭐

**นี่คือจุดที่แนวคิดของอาจารย์ถูกทดสอบ**
ทุกแถวตรึง recall ≥ 99% เท่ากัน และตรึงโมเดล = XGBoost
เพื่อให้ตัวแปรที่ต่างกันมีแค่ *ชุดฟีเจอร์* อย่างเดียว

| feature_set        |   #feat |   recall |   fp_rate |   precision |       f1 |   ส่งต่อ_stage2_% |   latency (ms/1k) |   threshold |
|:-------------------|--------:|---------:|----------:|------------:|---------:|----------------:|------------------:|------------:|
| intersection_top15 |       2 | 0.99156  |  0.326854 |    0.924316 | 0.956758 |         85.9301 |          0.189035 |    0.208945 |
| global_mean_top15  |      15 | 0.999437 |  0.000146 |    0.999964 | 0.9997   |         80.0605 |          0.224188 |    0.99     |
| shap_from_binary   |      15 | 0.999583 |  7.3e-05  |    0.999982 | 0.999782 |         80.0707 |          0.208637 |    0.99     |
| union_top15        |      37 | 0.999492 |  7.3e-05  |    0.999982 | 0.999737 |         80.0634 |          0.212432 |    0.99     |
| dynamic_k          |      40 | 0.999601 |  0.000146 |    0.999964 | 0.999782 |         80.0736 |          0.233433 |    0.99     |
| all_features       |      65 | 0.999583 |  7.3e-05  |    0.999982 | 0.999782 |         80.0707 |          0.24198  |    0.99     |

**อ่านยังไง:** หาแถวที่ `#feat` น้อยที่สุด โดย `recall` ไม่ตกและ `fp_rate` ไม่ขึ้น

### ถ้าเปลี่ยนโมเดล ผลต่างไหม (recall)

| feature_set        |   n_features |   DecisionTree |   RandomForest |   XGBoost |
|:-------------------|-------------:|---------------:|---------------:|----------:|
| intersection_top15 |            2 |         0.9907 |         0.991  |    0.9916 |
| global_mean_top15  |           15 |         0.9998 |         0.9965 |    0.9994 |
| shap_from_binary   |           15 |         0.9999 |         0.997  |    0.9996 |
| union_top15        |           37 |         0.9999 |         0.9961 |    0.9995 |
| dynamic_k          |           40 |         0.9999 |         0.9964 |    0.9996 |
| all_features       |           65 |         0.9999 |         0.9954 |    0.9996 |

---

## 4. Zero-shot — attack ที่ไม่เคยเห็น

เทรนด้วย OVS → เทสด้วย metasploitable  (U2R ไม่เคยอยู่ใน train เลย)

| feature_set        |   #feat |   recall |   precision |     f1 |   latency (ms/1k) |
|:-------------------|--------:|---------:|------------:|-------:|------------------:|
| intersection_top15 |       2 |   0.7344 |      0.9774 | 0.8387 |            0.1641 |
| global_mean_top15  |      15 |   0.9975 |      0.9999 | 0.9987 |            0.174  |
| shap_from_binary   |      15 |   0.9981 |      0.9999 | 0.999  |            0.1563 |
| union_top15        |      37 |   0.9981 |      0.9999 | 0.999  |            0.2046 |
| dynamic_k          |      40 |   0.9979 |      0.9999 | 0.9989 |            0.1822 |
| all_features       |      65 |   0.9975 |      0.9999 | 0.9987 |            0.2437 |

### Recall รายคลาส × ชุดฟีเจอร์

| attack_class   | เคยเห็น         |   n_test |   all_features |   dynamic_k |   global_mean_top15 |   intersection_top15 |   shap_from_binary |   union_top15 |
|:---------------|:---------------|---------:|---------------:|------------:|--------------------:|---------------------:|-------------------:|--------------:|
| BFA            | ใช่             |      295 |         1      |      1      |              1      |               0.5051 |             1      |        1      |
| DDoS           | ใช่             |    73529 |         1      |      1      |              0.9995 |               0.7275 |             1      |        1      |
| DoS            | ใช่             |     1145 |         0.7397 |      0.7563 |              0.7336 |               0.0856 |             0.7843 |        0.7825 |
| Probe          | ใช่             |    61757 |         0.9994 |      0.9999 |              0.9999 |               0.7558 |             0.9999 |        0.9999 |
| U2R            | ไม่ (zero-shot) |       17 |         0.9412 |      1      |              1      |               0.5882 |             1      |        1      |

**อ่านยังไง:** แถว `U2R` คือคลาสที่ไม่เคยเทรน
ถ้า recall ไม่ตกตอนลดฟีเจอร์ = การลดฟีเจอร์ไม่ทำลายความสามารถ generalize

---

## 5. ผลรอง — stage 2 เองก็ลดฟีเจอร์ได้ไหม

ไม่ใช่ประเด็นหลัก แต่ตอบคำถามว่าชุดฟีเจอร์ที่คัดมาใช้กับ stage 2 ได้ด้วยหรือเปล่า

|                    |   n_features |     f1 |   precision |   recall |   latency (ms/1k) |   f1_เทียบ_full_% |
|:-------------------|-------------:|-------:|------------:|---------:|------------------:|-----------------:|
| intersection_top15 |            2 | 0.7637 |      0.9629 |   0.6959 |            1.1073 |          79.8572 |
| global_mean_top15  |           15 | 0.9377 |      0.9301 |   0.9623 |            0.8669 |          98.056  |
| shap_from_binary   |           15 | 0.9354 |      0.9302 |   0.9575 |            0.8679 |          97.8121 |
| union_top15        |           37 | 0.9061 |      0.9041 |   0.9198 |            0.9839 |          94.7504 |
| dynamic_k          |           40 | 0.9421 |      0.9327 |   0.9679 |            1.0255 |          98.5189 |
| all_features       |           65 | 0.9563 |      0.9535 |   0.9664 |            0.8897 |         100      |

---

## 6. ฟีเจอร์ที่ใช้ทั้งหมด (65 ตัว)

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
