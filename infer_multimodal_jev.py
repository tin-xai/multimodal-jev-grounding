#!/usr/bin/env python3
"""Run multimodal JEV inference and print option probabilities plus a box."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from train_multimodal_jev import JEVBoxModel, MARKERS, marker_token_ids


def load_record(path: Path, record_id: str | None):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if record_id is None or row.get("id") == record_id:
            return row
    raise ValueError(f"No record found in {path} for id={record_id!r}")


def make_prompt(question: str, options: list[str]) -> str:
    option_text = " ".join(f"{i}: {name.replace('_', ' ')}" for i, name in enumerate(options))
    return f"{question} {option_text} Answer with one digit: 0, 1, 2, or 3. Answer:"


def prepare_inputs(processor, image, prompt):
    conversations = [[{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]]
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    else:
        text = [prompt]
    return processor(text=text, images=[image], return_tensors="pt", padding=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument("--box-checkpoint", required=True, help="epoch-N-box.pt file")
    parser.add_argument("--record", type=Path, help="JSON or JSONL CUB record")
    parser.add_argument("--record-id", help="Record id when --record is JSONL")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--question", default="Which bird species is shown?")
    parser.add_argument("--options", nargs=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.record:
        row = load_record(args.record, args.record_id)
        image_path = Path(row["image"])
        question = row.get("question", "Which bird species is shown?")
        options = row["options"]
    else:
        if args.image is None or args.options is None:
            parser.error("Provide --record or --image plus four --options")
        image_path, question, options = args.image, args.question, args.options
    if len(options) != 4:
        parser.error("This checkpoint expects exactly four options")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    base = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype)
    from peft import PeftModel
    base = PeftModel.from_pretrained(base, args.adapter)
    model = JEVBoxModel(base, marker_token_ids(processor, tokenizer)).to(device)
    checkpoint = torch.load(args.box_checkpoint, map_location="cpu")
    model.box_head.load_state_dict(checkpoint["box_head"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = prepare_inputs(processor, image, make_prompt(question, options))
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    with torch.inference_mode():
        option_logits, boxes = model(**inputs)
    probabilities = option_logits.float().softmax(-1)[0].cpu().tolist()
    normalized_box = boxes[0].float().cpu().tolist()
    width, height = image.size
    pixel_box = [
        normalized_box[0] * width,
        normalized_box[1] * height,
        normalized_box[2] * width,
        normalized_box[3] * height,
    ]
    result = {
        "image": str(image_path),
        "question": question,
        "options": options,
        "choice": options[int(torch.tensor(probabilities).argmax())],
        "probabilities": dict(zip(options, probabilities)),
        "box_normalized_xyxy": normalized_box,
        "box_pixel_xyxy": pixel_box,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
