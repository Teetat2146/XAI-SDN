"""ค่าคงที่ทั้งหมดของโปรเจกต์ — แก้ที่นี่ที่เดียว ทุกสคริปต์อ่านจากไฟล์นี้"""
from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).parent

# CSV ต้นฉบับของ InSDN เก็บไว้ในโปรเจกต์เลย (data/raw/) ไม่ใช่ชี้ไปที่ ~/Downloads
# เหตุผล: โปรเจกต์ต้องครบในตัวเอง ก๊อปโฟลเดอร์นี้ไปเครื่องอื่นแล้วต้องรันได้ทันที
#         ถ้าชี้ไป Downloads แล้ววันหนึ่งเผลอล้างโฟลเดอร์ ทุกอย่างพังหมด
# หมายเหตุ: ไฟล์ใน data/raw/ เป็นสำเนา — ต้นฉบับใน ~/Downloads ยังอยู่ครบ ไม่ถูกแตะ
RAW_DIR = ROOT / "data" / "raw"
RAW_FILES = {
    "Normal": RAW_DIR / "Normal_data.csv",
    "OVS": RAW_DIR / "OVS.csv",
    "metasploitable": RAW_DIR / "metasploitable-2.csv",
}
DATA_DIR = ROOT / "data"        # ข้อมูลที่ clean แล้ว
OUT_DIR = ROOT / "outputs"      # ตารางผลลัพธ์ (csv) เอาไปใส่รายงานได้เลย
FIG_DIR = ROOT / "figures"      # กราฟ
MODEL_DIR = ROOT / "models"     # โมเดลที่เทรนแล้ว (.pkl)

for _d in (DATA_DIR, OUT_DIR, FIG_DIR, MODEL_DIR):
    _d.mkdir(exist_ok=True)

CLEAN_PARQUET = DATA_DIR / "insdn_clean.parquet"

# ---- Reproducibility ----
SEED = 42
TEST_SIZE = 0.2

# ================================================================
# กลุ่มคอลัมน์ที่ "ตัดได้" — ใช้ประกอบเป็นระดับการตัดฟีเจอร์
# ================================================================
# หลักคิด: ไม่ตัดตั้งแต่ตอนเตรียมข้อมูล แต่เก็บไว้ทั้งหมดแล้วให้แต่ละการทดลอง
# เลือกเองว่าจะตัดถึงระดับไหน — จะได้พิสูจน์ *ด้วยการทดลอง* ว่าตัดแต่ละกลุ่มถูกไหม
# (ประชุม 2026-08-08: "รู้ได้ไงว่าถ้าไม่ตัดมันจะผิด" → ต้องเป็น by experiment)

# กลุ่ม 1 — identifier: จำ testbed ได้ ไม่ใช่พฤติกรรมของ traffic
# หมายเหตุ cardinality (จาก 343,889 แถว):
#   Flow ID 234,971 ค่า (68%) เกือบเป็นเลขประจำแถว
#   Src IP  122,868 ค่า (36%) สูงเพราะ DDoS ปลอม source IP
#   Dst IP    1,081 ค่า · Timestamp 1,392 ค่า
ID_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]

# กลุ่ม 2 — Port: shortcut ที่ผูกกับ testbed (BFA→22, Web-Attack→80)
PORT_COLS = ["Src Port", "Dst Port"]

# กลุ่ม 3 — ค่าคงที่ทั้งคอลัมน์ (บั๊กของ CICFlowMeter ให้ 0 หมด)
# คำนวณอัตโนมัติใน 01 แล้วบันทึกลง 01_drop_lists.json — ที่นี่เป็นค่าอ้างอิง
CONST_COLS_EXPECTED = 12

# กลุ่ม 4 — ซ้ำซ้อน (correlation = 1.0000) เก็บชื่อที่เป็นมาตรฐานกว่า ตัดอีกตัว
DUP_COLS = [
    "Subflow Fwd Pkts",   # = Tot Fwd Pkts
    "Subflow Bwd Pkts",   # = Tot Bwd Pkts
    "Subflow Fwd Byts",   # = TotLen Fwd Pkts
    "Subflow Bwd Byts",   # = TotLen Bwd Pkts
    "Fwd Seg Size Avg",   # = Fwd Pkt Len Mean
    "Bwd Seg Size Avg",   # = Bwd Pkt Len Mean
    "Bwd PSH Flags",      # = PSH Flag Cnt
    "Bwd URG Flags",      # = URG Flag Cnt
]

# กลุ่ม 5 — คำนวณแพงตอน real-time (ต้องรอ flow จบก่อนถึงคำนวณได้)
# เป็นข้ออ้างหลักของการลดฟีเจอร์ที่เขียนไว้ใน README แต่ยังไม่เคยพิสูจน์
EXPENSIVE_COLS = [f"{k} {s}" for k in ("Active", "Idle")
                  for s in ("Mean", "Std", "Max", "Min")]

# ---- ระดับการตัด: สะสมกันไปเรื่อย ๆ ----
# ระดับ n ตัดทุกอย่างของระดับ n-1 บวกกลุ่มใหม่ → รู้ว่ากลุ่มไหนมีผลแค่ไหน
FEATURE_LEVELS = ["raw", "no_id", "no_const", "no_port",
                  "no_dup", "no_expensive", "shap_top15"]

# ---- แบบการแบ่ง train/test ----
#   random    : สุ่ม 80/20 จากข้อมูลปนกันหมด — วัดประสิทธิภาพในสภาพแวดล้อมเดิม
#   ovs2meta  : train OVS → test metasploitable — วัดว่า generalize ได้ไหม
#   meta2ovs  : ทิศกลับ
# ต้องดูทั้ง 3 เพราะ random อย่างเดียวจับ leakage ไม่ได้ (โมเดลจำ IP ก็ยังทายถูก)
SPLITS = ["random", "ovs2meta", "meta2ovs"]

# B/C ใช้เฉพาะคลาสที่มีทั้งสอง testbed — BOTNET/Web-Attack มีแค่ OVS, U2R มีแค่ meta
# multi-class ทายคลาสที่ไม่มี output node ไม่ได้ ถ้าไม่ตัดออก macro-F1 จะเพี้ยน
SHARED_CLASSES = ["Normal", "DDoS", "DoS", "Probe", "BFA"]

DROP_LISTS_JSON = OUT_DIR / "01_drop_lists.json"

LABEL_COL = "Label"
NORMAL_LABEL = "Normal"

# ---- Hyperparameters ----
# follow ค่าที่งานวิจัยสาย SDN-IDS นิยมใช้ เพื่อให้เทียบผลกับ paper อื่นได้
N_ESTIMATORS = 100

# ---- SHAP ----
SHAP_SAMPLE = 5000   # subsample สำหรับคำนวณ SHAP (ทั้ง 344k แถวช้าและกินแรมเกินจำเป็น)

# ---- Feature selection ----
TOPK = 15                    # สำหรับวิธี fixed Top-K
CUMULATIVE_THRESHOLD = 0.90  # สำหรับวิธี dynamic-K

# ไฟล์เก็บชุดฟีเจอร์ที่ 03 ผลิตออกมา แล้ว 04/05 อ่านไปใช้ต่อ
# ใช้ JSON เพราะอ่านกลับมาเป็น list ได้ตรง ๆ ไม่ต้อง split string เหมือน CSV
FEATURE_SETS_JSON = OUT_DIR / "03_feature_sets.json"

# ---- เกณฑ์ของ stage 1 ----
# stage 1 เป็นตัวกรอง หน้าที่คือ "ห้ามพลาด attack" ไม่ใช่ "ทายถูกมากที่สุด"
# จึงตรึง recall ไว้ที่ค่านี้ก่อน แล้วค่อยเทียบว่าชุดฟีเจอร์ไหนให้ FP ต่ำสุด/เร็วสุด
# ถ้าวัดที่ threshold 0.5 เหมือนกันหมด จะเทียบกันไม่ยุติธรรม เพราะแต่ละชุดฟีเจอร์
# ทำให้โมเดล "เข้มงวด" ไม่เท่ากันโดยธรรมชาติ
TARGET_RECALL = 0.99
