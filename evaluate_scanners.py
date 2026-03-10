"""
evaluate_scanners.py - Compare YOLO+PaddleOCR vs Qwen2-VL

Metrics: Accuracy, CER, Levenshtein, Inference Time
CSV output: one row per image per model
"""

import os
import re
import csv
import json
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from difflib import SequenceMatcher

# ── Point these at your dataset ───────────────────────────────────────────────
TEST_IMAGES_DIR   = "evaluation_dataset/synthetic/images"
GROUND_TRUTH_FILE = "evaluation_dataset/synthetic/annotations.json"

# To test on original real images swap to:
# TEST_IMAGES_DIR   = "D:/WORK/License_verification/test/images"
# GROUND_TRUTH_FILE = "evaluation_dataset/ground_truth/annotations.json"

CSV_OUTPUT_FILE  = "evaluation_dataset/results/evaluation_results.csv"
JSON_OUTPUT_FILE = "evaluation_dataset/results/evaluation_results.json"

ALL_FIELDS = ["name", "dl_number", "issue_date", "expiry_date"]

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# =========================
# Metrics
# =========================
def levenshtein_distance(s1, s2):
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    d = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        d[i][0] = i
    for j in range(len(s2) + 1):
        d[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(s1)][len(s2)]


def character_error_rate(gt, pred):
    if not gt:
        return 1.0 if pred else 0.0
    return levenshtein_distance(gt, pred or "") / len(gt)


def similarity_ratio(s1, s2):
    return SequenceMatcher(None, s1 or "", s2 or "").ratio()


def normalize_dl_number(dl):
    if not dl:
        return ""
    return re.sub(r'\s+', '', dl).upper().strip()


def normalize_date(d):
    if not d:
        return ""
    return d.strip().replace('/', '-').upper()


def normalize_name(n):
    if not n:
        return ""
    return n.strip().upper()


def normalize_field(field, value):
    if field == "dl_number":
        return normalize_dl_number(value)
    elif field in ("issue_date", "expiry_date"):
        return normalize_date(value)
    else:
        return normalize_name(value)


# =========================
# Load Ground Truth
# =========================
def load_ground_truth():
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"❌ Ground truth not found: {GROUND_TRUTH_FILE}")
        return None

    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        annotated = {item['image']: item['fields'] for item in data
                     if item['fields'].get('name') or item['fields'].get('dl_number')}
    else:
        annotated = {k: v for k, v in data.items()
                     if v.get('name') or v.get('dl_number')}

    print(f"✅ Loaded {len(annotated)} annotated samples")
    return annotated


# =========================
# Scanners
# =========================
def run_yolo_paddleocr_scanner(image_path):
    import sys
    sys.path.insert(0, 'D:/WORK/License_verification')
    from dl_camera_app import (
        detect_fields, extract_all_text, find_issue_date,
        find_dl_from_full_text, yolo_model, ocr
    )

    start_time = time.time()
    img = cv2.imread(image_path)

    if img is None:
        return {"name": None, "dl_number": None, "issue_date": None,
                "expiry_date": None, "inference_time": 0, "error": "Failed to load"}

    try:
        yolo_fields = detect_fields(img)
        name      = yolo_fields.get('name', {}).get('text') if 'name' in yolo_fields else None
        dl_number = yolo_fields.get('dl_no', {}).get('text') if 'dl_no' in yolo_fields else None
        dl_number = normalize_dl_number(dl_number) if dl_number else None

        full_texts = extract_all_text(img)

        if not dl_number:
            raw = find_dl_from_full_text(full_texts)
            dl_number = normalize_dl_number(raw) if raw else None

        issue_date = find_issue_date(full_texts)

        # Expiry: latest DD-MM-YYYY date in full OCR text
        expiry_date = None
        all_text    = ' '.join(full_texts) if isinstance(full_texts, list) else str(full_texts)
        dates_found = re.findall(r'\d{2}[-/]\d{2}[-/]\d{4}', all_text)
        if dates_found:
            parsed = []
            for d in dates_found:
                d_clean = d.replace('/', '-')
                try:
                    parsed.append((datetime.strptime(d_clean, '%d-%m-%Y'), d_clean))
                except ValueError:
                    pass
            if parsed:
                parsed.sort(key=lambda x: x[0])
                expiry_date = parsed[-1][1]
                if issue_date and expiry_date == normalize_date(issue_date) and len(parsed) > 1:
                    expiry_date = parsed[-2][1]

        return {
            "name": name, "dl_number": dl_number,
            "issue_date": issue_date, "expiry_date": expiry_date,
            "inference_time": time.time() - start_time
        }

    except Exception as e:
        return {"name": None, "dl_number": None, "issue_date": None,
                "expiry_date": None, "inference_time": time.time() - start_time,
                "error": str(e)}


def run_qwen2vl_scanner(image_path):
    import sys
    sys.path.insert(0, 'D:/WORK/License_verification')
    from dl_scanner_qwen2vl import extract_dl_info, parse_front_output, model, processor

    start_time = time.time()
    try:
        pil_img = Image.open(image_path)
        output  = extract_dl_info(pil_img, side="front")
        data    = parse_front_output(output)

        if data.get('dl_number'):
            data['dl_number'] = normalize_dl_number(data['dl_number'])

        return {
            "name":           data.get('name'),
            "dl_number":      data.get('dl_number'),
            "issue_date":     data.get('issue_date'),
            "expiry_date":    data.get('expiry_date'),
            "inference_time": time.time() - start_time
        }
    except Exception as e:
        return {"name": None, "dl_number": None, "issue_date": None,
                "expiry_date": None, "inference_time": time.time() - start_time,
                "error": str(e)}


# =========================
# CSV Schema
# =========================
def get_csv_headers():
    headers = ["image_file", "model", "source_type"]
    for field in ALL_FIELDS:
        headers += [
            f"gt_{field}",
            f"pred_{field}",
            f"{field}_match",       # 1 = exact match, 0 = wrong, "" = no GT
            f"{field}_cer",         # Character Error Rate
            f"{field}_similarity",  # 0.0-1.0 string similarity
        ]
    headers += ["fields_correct", "fields_total", "accuracy_pct",
                "inference_time_s", "error"]
    return headers


def build_csv_row(img_file, model_name, gt, prediction, source_type="unknown"):
    row = {
        "image_file":      img_file,
        "model":           model_name,
        "source_type":     source_type,
        "error":           prediction.get("error", ""),
        "inference_time_s": round(prediction.get("inference_time", 0), 3),
    }

    fields_correct = 0
    fields_with_gt = 0

    for field in ALL_FIELDS:
        gt_raw   = gt.get(field, "") or ""
        pred_raw = prediction.get(field) or ""

        gt_norm   = normalize_field(field, gt_raw)
        pred_norm = normalize_field(field, pred_raw)

        match = int(gt_norm == pred_norm) if gt_norm else ""
        cer   = round(character_error_rate(gt_norm, pred_norm), 4) if gt_norm else ""
        sim   = round(similarity_ratio(gt_norm, pred_norm), 4) if gt_norm else ""

        row[f"gt_{field}"]         = gt_norm
        row[f"pred_{field}"]       = pred_norm
        row[f"{field}_match"]      = match
        row[f"{field}_cer"]        = cer
        row[f"{field}_similarity"] = sim

        if gt_norm:
            fields_with_gt += 1
            if match == 1:
                fields_correct += 1

    row["fields_correct"] = fields_correct
    row["fields_total"]   = fields_with_gt
    row["accuracy_pct"]   = round(fields_correct / fields_with_gt * 100, 1) if fields_with_gt else 0

    return row


# =========================
# Evaluate all images
# =========================
def evaluate_all(ground_truth, scanners):
    os.makedirs(os.path.dirname(CSV_OUTPUT_FILE), exist_ok=True)
    headers = get_csv_headers()

    all_rows      = []
    summary       = {name: {f: {"correct": 0, "cer": [], "sim": []} for f in ALL_FIELDS}
                     for name, _ in scanners}
    summary_times = {name: [] for name, _ in scanners}

    total = len(ground_truth)
    done  = 0

    with open(CSV_OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for img_file, gt in ground_truth.items():
            img_path    = os.path.join(TEST_IMAGES_DIR, img_file)
            source_type = gt.get("source", "unknown")
            done += 1

            if not os.path.exists(img_path):
                print(f"  ⚠️  [{done}/{total}] Not found: {img_file}")
                continue

            if done % 50 == 0 or done <= 3:
                print(f"  🔄 [{done}/{total}] Processing {img_file}")

            for model_name, scanner_func in scanners:
                prediction = scanner_func(img_path)
                row        = build_csv_row(img_file, model_name, gt, prediction, source_type)
                writer.writerow(row)
                all_rows.append(row)

                # Accumulate stats
                summary_times[model_name].append(prediction.get("inference_time", 0))
                for field in ALL_FIELDS:
                    gt_norm   = normalize_field(field, gt.get(field, "") or "")
                    pred_norm = normalize_field(field, prediction.get(field) or "")
                    if not gt_norm:
                        continue
                    if gt_norm == pred_norm:
                        summary[model_name][field]["correct"] += 1
                    summary[model_name][field]["cer"].append(
                        character_error_rate(gt_norm, pred_norm))
                    summary[model_name][field]["sim"].append(
                        similarity_ratio(gt_norm, pred_norm))

    print(f"\n✅ CSV written → {CSV_OUTPUT_FILE}  ({len(all_rows)} rows)")
    return all_rows, summary, summary_times


# =========================
# Print summary table
# =========================
def print_summary(summary, summary_times, total_images):
    print("\n" + "="*72)
    print("   FINAL EVALUATION SUMMARY")
    print("="*72)

    model_names = list(summary.keys())

    col_w = 28
    print(f"\n{'Field':<18}", end="")
    for m in model_names:
        print(f"  {m[:col_w-2]:<{col_w}}", end="")
    print()
    print("-" * (18 + (col_w + 2) * len(model_names)))

    for field in ALL_FIELDS:
        print(f"{field.upper():<18}", end="")
        for m in model_names:
            d       = summary[m][field]
            n       = len(d["cer"])
            correct = d["correct"]
            pct     = correct / n * 100 if n else 0
            avg_cer = np.mean(d["cer"]) if d["cer"] else 0
            cell    = f"{correct}/{n} ({pct:.0f}%) CER:{avg_cer:.3f}"
            print(f"  {cell:<{col_w}}", end="")
        print()

    print(f"\n{'Avg Time':<18}", end="")
    for m in model_names:
        t = np.mean(summary_times[m]) if summary_times[m] else 0
        print(f"  {t:.3f}s{'':<{col_w-7}}", end="")
    print()

    print("\n" + "="*72)

    totals = {m: sum(summary[m][f]["correct"] for f in ALL_FIELDS)
              for m in model_names}
    winner = max(totals, key=totals.get)

    print(f"\n🏆 WINNER: {winner}")
    for m in model_names:
        avg_t = np.mean(summary_times[m]) if summary_times[m] else 0
        print(f"   {m:<20}: {totals[m]} correct fields | {avg_t:.3f}s avg")
    print("="*72)

    return totals


def save_json_summary(summary, summary_times, totals):
    output = {
        "timestamp":   datetime.now().isoformat(),
        "csv_file":    CSV_OUTPUT_FILE,
        "per_model":   {},
    }
    for m in summary:
        output["per_model"][m] = {
            "total_correct_fields": totals[m],
            "avg_inference_time":   round(np.mean(summary_times[m]), 3) if summary_times[m] else 0,
            "fields": {}
        }
        for field in ALL_FIELDS:
            d = summary[m][field]
            n = len(d["cer"])
            output["per_model"][m]["fields"][field] = {
                "correct":  d["correct"],
                "n":        n,
                "accuracy": round(d["correct"] / n * 100, 1) if n else 0,
                "avg_cer":  round(np.mean(d["cer"]), 4) if d["cer"] else None,
                "avg_sim":  round(np.mean(d["sim"]), 4) if d["sim"] else None,
            }

    os.makedirs(os.path.dirname(JSON_OUTPUT_FILE), exist_ok=True)
    with open(JSON_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"✅ JSON summary → {JSON_OUTPUT_FILE}")


# =========================
# Main
# =========================
def main():
    print("="*60)
    print("   DL SCANNER EVALUATION")
    print("="*60)

    print("\n🔧 Loading models...")

    import sys
    sys.path.insert(0, 'D:/WORK/License_verification')

    scanners = []

    try:
        from dl_camera_app import load_models
        load_models()
        print("✅ YOLO + PaddleOCR loaded")
        scanners.append(("YOLO+PaddleOCR", run_yolo_paddleocr_scanner))
    except Exception as e:
        print(f"❌ YOLO+PaddleOCR: {e}")

    try:
        from dl_scanner_qwen2vl import load_model
        load_model()
        print("✅ Qwen2-VL loaded")
        scanners.append(("Qwen2-VL-INT4", run_qwen2vl_scanner))
    except Exception as e:
        print(f"❌ Qwen2-VL: {e}")

    if not scanners:
        print("❌ No models loaded. Exiting.")
        return

    ground_truth = load_ground_truth()
    if not ground_truth:
        return

    print(f"\n📊 {len(ground_truth)} images × {len(scanners)} models "
          f"= {len(ground_truth) * len(scanners)} CSV rows\n")

    all_rows, summary, summary_times = evaluate_all(ground_truth, scanners)

    totals = print_summary(summary, summary_times, len(ground_truth))
    save_json_summary(summary, summary_times, totals)

    print(f"\n📁 Results:")
    print(f"   CSV  → {CSV_OUTPUT_FILE}")
    print(f"   JSON → {JSON_OUTPUT_FILE}")


if __name__ == "__main__":
    main()