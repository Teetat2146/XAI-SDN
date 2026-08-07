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

# ---- คอลัมน์ที่ต้องตัดออก: identifier / leakage ----
# เหตุผลรายตัวอยู่ใน README หัวข้อ "Data leakage"
LEAK_COLS = [
    "Flow ID",    # string ประกอบจาก IP+Port+Protocol = ยัดคำตอบให้โมเดลตรง ๆ
    "Src IP",     # เครื่องที่ยิง attack ใน testbed มี IP ตายตัว
    "Dst IP",     # เครื่องเหยื่อ (metasploitable) ใช้ IP เดิมตลอด
    "Src Port",   # ephemeral port ผูกกับ session ที่ยิง
    "Dst Port",   # ใน testbed ผูกกับชนิด attack เกือบ 1:1
    "Timestamp",  # attack ยิงเป็นช่วงเวลา โมเดลจะจำ "ช่วงนี้คือ attack"
]

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
