import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import cv2
import re
import torch
import json
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from pymongo import MongoClient
from PIL import Image
import numpy as np

model = None
processor = None

# =========================
# Load Qwen2-VL Model
# =========================
def load_model():
    global model, processor

    print("Loading Qwen2-VL-2B-Instruct with INT4 Quantization...")
    print("  (First run will take longer, then cached)")

    # INT4 Quantization for 3x smaller, 2x faster
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # NormalFloat4 - best for LLMs
        bnb_4bit_use_double_quant=True  # Nested quantization for extra compression
    )

    # Load model with quantization
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        quantization_config=quantization_config,
        device_map="cuda",
        torch_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    adapter_path = os.environ.get("QWEN2VL_LORA_ADAPTER")
    if adapter_path and os.path.isdir(adapter_path):
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"Loaded LoRA adapter: {adapter_path}")
        except Exception as exc:
            print(f"Warning: failed to load LoRA adapter ({adapter_path}): {exc}")

    # Check VRAM usage
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ Qwen2-VL loaded (INT4 quantized)")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {vram_used:.2f}GB / {vram_total:.2f}GB ({vram_used/vram_total*100:.1f}%)")


# =========================
# Extract Fields using VLM
# =========================
def extract_dl_info(image_pil, side="front"):
    """Use Qwen2-VL to extract information from license image."""

    if side == "front":
        prompt = """Look at this Indian driving license (front side).
Extract the following information:
1. Full name (the person's name, usually below the DL number)
2. DL number (format: 2 letters followed by 13-17 digits, example: TS10820200000403)
3. Issue date (labeled as "Issued On:" or "DOI", format: DD-MM-YYYY or DD/MM/YYYY)

CRITICAL RULES:
- The DL number is printed in RED color at the top
- The person's name is directly below the DL number (NOT the father's name which appears after)
- DO NOT include expiry date - it's on the back side only
- If you cannot find a field, do NOT make it up or guess

Return ONLY in this exact format:
Name: [full name]
DL Number: [number]
Issue Date: [date]"""

    else:  # back
        prompt = """Look at this Indian driving license (back side).
Extract the expiry date/validity date.
It's usually labeled as "Date of Validity" or "Valid Till" or appears after "Non-Transport".

Return ONLY:
Expiry Date: [date in DD-MM-YYYY format]"""

    # Prepare message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Generate
    print(f"  🤖 Processing {side} with Qwen2-VL...")

    import time
    start_time = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,  # Low temperature for factual extraction
            do_sample=False   # Deterministic output
        )

    inference_time = time.time() - start_time
    print(f"     ⏱️  Inference time: {inference_time:.2f}s")

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


# =========================
# Parse VLM Output
# =========================
def parse_front_output(text):
    """Parse front side VLM output."""
    data = {"name": None, "dl_number": None, "issue_date": None}

    # Extract name
    name_match = re.search(r'Name:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if name_match:
        data["name"] = name_match.group(1).strip().upper()

    # Extract DL number
    dl_match = re.search(r'DL Number:\s*([A-Z]{2}\d{13,17})', text, re.IGNORECASE)
    if dl_match:
        data["dl_number"] = dl_match.group(1)

    # Extract issue date
    date_match = re.search(r'Issue Date:\s*(\d{2}[-/]\d{2}[-/]\d{4})', text, re.IGNORECASE)
    if date_match:
        date_str = date_match.group(1).replace('/', '-')
        data["issue_date"] = date_str

    # IGNORE expiry date on front - it's hallucinated!
    # Front side does NOT have expiry date visible

    return data


def parse_back_output(text):
    """Parse back side VLM output."""
    # Extract expiry date
    date_match = re.search(r'Expiry Date:\s*(\d{2}[-/]\d{2}[-/]\d{4})', text, re.IGNORECASE)
    if date_match:
        return date_match.group(1).replace('/', '-')

    # Fallback: find any date
    date_match = re.search(r'(\d{2}[-/]\d{2}[-/]\d{4})', text)
    if date_match:
        return date_match.group(1).replace('/', '-')

    return None


# =========================
# LoRA Fine-tuning (QLoRA)
# =========================
class Qwen2VLDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Expect: {"messages":[{"role":"user","content":[{"type":"image","image":...},{"type":"text","text":...}]},{"role":"assistant","content":[{"type":"text","text":...}]}]}
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Qwen2VLDataCollator:
    """Data collator for Qwen2-VL fine-tuning."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        """Collate batch of samples for Qwen2-VL training."""
        from qwen_vl_utils import process_vision_info

        texts = []
        image_inputs = []
        video_inputs = []

        for item in batch:
            messages = item["messages"]

            # Apply chat template (add_generation_prompt=False for training!)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False  # ← CHANGED: False for training
            )
            texts.append(text)

            # Process vision info
            imgs, vids = process_vision_info(messages)

            # Handle None values
            if imgs:
                image_inputs.extend(imgs)
            if vids:
                video_inputs.extend(vids)

        # Tokenize and prepare inputs
        inputs = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        # ← CRITICAL FIX: Add labels for training
        labels = inputs["input_ids"].clone()

        # Mask padding tokens (-100 = ignored in loss)
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100

        inputs["labels"] = labels

        return inputs


def train_lora(
    train_jsonl,
    val_jsonl=None,
    output_dir="runs/qwen2vl_lora",
    epochs=2,
    batch_size=1,
    grad_accum=8,
    lr=2e-4,
    save_steps=500,
):
    from transformers import TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    global model, processor

    print("Loading base model (QLoRA)...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        quantization_config=quantization_config,
        device_map="cuda",
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_dataset = Qwen2VLDataset(train_jsonl)
    eval_dataset = Qwen2VLDataset(val_jsonl) if val_jsonl else None
    data_collator = Qwen2VLDataCollator(processor)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=20,
        save_steps=save_steps,
        save_total_limit=2,
        fp16=True,
        eval_strategy="no",
        eval_steps=save_steps if eval_dataset else None,
        report_to="none",
        remove_unused_columns=False,  # Important for vision models
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Done. Adapter saved to: {output_dir}")


def check_expiry(expiry_str):
    """Check if license is valid."""
    try:
        dt = datetime.strptime(expiry_str, "%d-%m-%Y")
    except:
        try:
            dt = datetime.strptime(expiry_str, "%d/%m/%Y")
        except:
            return None, "Invalid date format"

    today = datetime.today()
    if dt < today:
        days = (today - dt).days
        return False, f"EXPIRED — {days} days ago ({dt.strftime('%d %b %Y')})"
    days = (dt - today).days
    return True, f"VALID — expires {dt.strftime('%d %b %Y')} ({days} days left)"


# =========================
# Camera Capture
# =========================
def capture(instruction):
    """Capture image from DroidCam."""
    droidcam_ip = "http://192.168.0.105:4747/video"
    cap = cv2.VideoCapture(droidcam_ip)

    if not cap.isOpened():
        print("⚠️  DroidCam not available, using default camera")
        cap = cv2.VideoCapture(0)

    print(f"\n  {instruction}")
    print("  SPACE = capture   Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        bw = int(w * 0.82)
        bh = int(bw / 1.585)
        x1 = (w - bw) // 2
        y1 = (h - bh) // 2

        display = frame.copy()
        cv2.rectangle(display, (x1, y1), (x1+bw, y1+bh), (0, 255, 0), 2)
        cv2.putText(display, instruction, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
        cv2.putText(display, "Fit license in box — SPACE to capture",
                    (20, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
        cv2.imshow("DL Scanner (Qwen2-VL)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            captured = frame.copy()
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Quit")

    cap.release()
    cv2.destroyAllWindows()

    # Convert to PIL for Qwen2-VL
    rgb = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    return pil_image


# =========================
# Main Scanning Logic
# =========================
def scan_front():
    """Scan front of license."""
    print("\n" + "="*60)
    print("SCANNING FRONT")
    print("="*60)

    frame = capture("Show FRONT of Driving License")

    # Extract info using Qwen2-VL
    output = extract_dl_info(frame, side="front")
    print(f"\n  📄 VLM Output:\n{output}\n")

    # Parse output
    data = parse_front_output(output)

    print(f"  ✔ Name       : {data['name'] or 'NOT FOUND'}")
    print(f"  ✔ DL Number  : {data['dl_number'] or 'NOT FOUND'}")
    print(f"  ✔ Issue Date : {data['issue_date'] or 'NOT FOUND'}")

    return data


def scan_back():
    """Scan back of license."""
    print("\n" + "="*60)
    print("SCANNING BACK")
    print("="*60)

    frame = capture("Show BACK of Driving License")

    # Extract expiry date
    output = extract_dl_info(frame, side="back")
    print(f"\n  📄 VLM Output:\n{output}\n")

    expiry_date = parse_back_output(output)

    print(f"  ✔ Expiry Date: {expiry_date or 'NOT FOUND'}")

    return expiry_date


def main():
    print("="*60)
    print("   INDIAN DL SCANNER (Qwen2-VL INT4)")
    print("="*60)

    # Load model
    load_model()

    # Connect to MongoDB (moved here so network issues don't block startup)
    MONGO_URI = os.environ.get("MONGO_URI",
                               "mongodb+srv://corazortechnology:A0Qfk2PbjOMKN32Z@cluster0.drxzj5r.mongodb.net/")

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        collection = client["licenseDB"]["licenses"]
        # Test connection
        client.server_info()
        print("✅ MongoDB connected")
    except Exception as e:
        print(f"⚠️  MongoDB connection failed: {e}")
        print("   (Will still scan, just won't save to DB)")
        collection = None

    # Scan front
    front_data = scan_front()

    # Scan back
    expiry_date = scan_back()

    # Final result
    print("\n" + "="*60)
    print("   FINAL RESULT")
    print("="*60)
    print(f"  Name       : {front_data['name'] or 'NOT FOUND'}")
    print(f"  DL Number  : {front_data['dl_number'] or 'NOT FOUND'}")
    print(f"  Issue Date : {front_data['issue_date'] or 'NOT FOUND'}")
    print(f"  Expiry Date: {expiry_date or 'NOT FOUND'}")

    verdict = None
    if expiry_date:
        verdict = check_expiry(expiry_date)
        print(f"  Status     : {verdict[1]}")

    print("="*60)

    # Save to MongoDB
    if collection is not None:
        doc = {
            "name": front_data['name'],
            "dl_number": front_data['dl_number'],
            "issue_date": front_data['issue_date'],
            "expiry_date": expiry_date,
            "valid": verdict[0] if verdict else None,
            "verdict": verdict[1] if verdict else None,
            "scanner_version": "Qwen2-VL-INT4",
            "created_at": datetime.utcnow(),
        }
        res = collection.insert_one(doc)
        print(f"\n  ✅ Saved to MongoDB (id: {res.inserted_id})")
    else:
        print("\n  ⚠️  Skipped MongoDB save (no connection)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2-VL DL Scanner / Fine-tuning")
    parser.add_argument("--train", action="store_true", help="Run LoRA fine-tuning")
    parser.add_argument("--train-jsonl", default="evaluation_dataset/synthetic_30k_finetune/train.jsonl")
    parser.add_argument("--val-jsonl", default="evaluation_dataset/synthetic_30k_finetune/val.jsonl")
    parser.add_argument("--output-dir", default="runs/qwen2vl_lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=500)
    args = parser.parse_args()

    if args.train:
        train_lora(
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            save_steps=args.save_steps,
        )
    else:
        main()
