import os
import time
import csv
import evaluate
from manuscript import Pipeline
from manuscript.detectors import EAST
from manuscript.recognizers import TRBA
import re


# =============================
# CONFIG
# =============================
DATASET_ROOT = r"C:\Users\pasha\OneDrive\Рабочий стол\Dataset of handwritten school essays in Russian\Dataset of handwritten school essays in Russian\handwritten_essay"
TEST_DIR = os.path.join(DATASET_ROOT, "train")  # можешь поменять на "test"
OUTPUT_CSV = "ocr_results.csv"


# =============================
# OCR PIPELINE
# =============================
pipeline = Pipeline(
    detector=EAST(device="cuda"),
    recognizer=TRBA(device="cuda", weights="trba_lite_g1"),
)

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


# =============================
# TEXT NORMALIZATION
# =============================
def normalize_text(t: str) -> str:
    t = t.replace("\n", " ").replace("\t", " ")
    t = re.sub(r"\s+", " ", t)
    t = t.lower()
    return t.strip()


# =============================
# PROCESS ONE SAMPLE
# =============================
def process_sample(sample_path):

    # ---------- GT ----------
    gt_path = os.path.join(sample_path, "text.txt")
    if not os.path.exists(gt_path):
        print(f"WARNING: no GT file at {gt_path}")
        return None

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_text = f.read().strip()

    # ---------- PAGES ----------
    pages = sorted([f for f in os.listdir(sample_path) if f.endswith(".png")])
    if not pages:
        print(f"WARNING: no PNG pages in {sample_path}")
        return None

    pred_full_text = ""

    for page_name in pages:
        page_path = os.path.join(sample_path, page_name)
        print(f"   → OCR {page_path}")

        result = pipeline.predict(page_path)
        pred_page = pipeline.get_text(
            pipeline.correct_with_llm(
                result["page"], api_url="https://demo.ai.sfu-kras.ru/v1"
            )
        )
        pred_full_text += pred_page + "\n"

    # ---------- NORMALIZE ----------
    pred_norm = normalize_text(pred_full_text)
    gt_norm = normalize_text(gt_text)

    # ---------- METRICS ----------
    cer = cer_metric.compute(predictions=[pred_norm], references=[gt_norm])
    wer = wer_metric.compute(predictions=[pred_norm], references=[gt_norm])

    return pred_full_text, gt_text, cer, wer


# =============================
# MAIN
# =============================
rows = []
sample_ids = sorted(os.listdir(TEST_DIR), key=lambda x: int(x))

for sid in sample_ids:
    sample_path = os.path.join(TEST_DIR, sid)
    if not os.path.isdir(sample_path):
        continue

    print(f"\n=== SAMPLE {sid} ===")

    result = process_sample(sample_path)
    if result is None:
        continue

    pred_text, gt_text, cer, wer = result

    rows.append([sid, cer, wer, pred_text, gt_text])

    print(f"  CER = {cer:.4f}, WER = {wer:.4f}")


# =============================
# SUMMARY
# =============================
mean_cer = sum(r[1] for r in rows) / len(rows)
mean_wer = sum(r[2] for r in rows) / len(rows)

print("\n========== SUMMARY METRICS ==========")
print(f"Mean CER: {mean_cer:.4f}")
print(f"Mean WER: {mean_wer:.4f}")


# =============================
# SAVE CSV
# =============================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_id", "cer", "wer", "pred_text", "gt_text"])
    writer.writerows(rows)

print(f"\nCSV saved to: {OUTPUT_CSV}")
