#!/usr/bin/env python3
"""Render a short annotated MP4 demo from a trained multimodal JEV model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor

from train_multimodal_jev import JEVBoxModel, marker_token_ids


def prompt_for(question, options):
    choices = " ".join(f"{i}: {x.replace('_', ' ')}" for i, x in enumerate(options))
    return f"{question} {choices} Answer with one digit: 0, 1, 2, or 3. Answer:"


def xyxy(box, width, height):
    return [int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--box-checkpoint", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("cub_jev_demo.mp4"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seconds-per-frame", type=float, default=3.0)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/cub_jev_demo"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()][:args.count]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    base = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype)
    from peft import PeftModel
    base = PeftModel.from_pretrained(base, args.adapter)
    model = JEVBoxModel(base, marker_token_ids(processor, processor.tokenizer)).to(device).eval()
    checkpoint = torch.load(args.box_checkpoint, map_location="cpu")
    model.box_head.load_state_dict(checkpoint["box_head"])

    args.work_dir.mkdir(parents=True, exist_ok=True)
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    font_path = next((path for path in font_candidates if Path(path).exists()), None)
    font = ImageFont.truetype(font_path, 28) if font_path else ImageFont.load_default()
    small_font = ImageFont.truetype(font_path, 22) if font_path else font
    title_font = ImageFont.truetype(font_path, 36) if font_path else font
    inference_seconds = []
    for index, row in enumerate(rows):
        image = Image.open(row["image"]).convert("RGB")
        conversations = [[{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_for("Which bird species is shown?", row["options"])},
        ]}]]
        text = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt", padding=True)
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        start = time.perf_counter()
        with torch.inference_mode():
            logits, boxes = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds.append(time.perf_counter() - start)
        probabilities = logits.float().softmax(-1)[0].cpu().tolist()
        predicted_box = boxes[0].float().cpu().tolist()
        predicted_index = int(torch.tensor(probabilities).argmax())

        canvas = Image.new("RGB", (1600, 900), "#101522")
        image.thumbnail((1080, 820))
        canvas.paste(image, (30 + (1080 - image.width) // 2, 40))
        draw = ImageDraw.Draw(canvas)
        image_x = 30 + (1080 - image.width) // 2
        image_y = 40
        draw.rectangle((image_x, image_y, image_x + image.width, image_y + image.height), outline="#ffffff", width=2)
        scale_x, scale_y = image.width, image.height
        pred = xyxy(predicted_box, scale_x, scale_y)
        gt = xyxy(row["box"], scale_x, scale_y)
        pred = [pred[0] + image_x, pred[1] + image_y, pred[2] + image_x, pred[3] + image_y]
        gt = [gt[0] + image_x, gt[1] + image_y, gt[2] + image_x, gt[3] + image_y]
        draw.rectangle(gt, outline="#46e37b", width=3)
        draw.rectangle(pred, outline="#ff4d5e", width=4)
        draw.text((pred[0] + 5, max(image_y, pred[1] - 14)), "predicted", fill="#ff4d5e", font=font)
        draw.text((gt[0] + 5, max(image_y, gt[1] - 28)), "ground truth", fill="#46e37b", font=font)

        panel_x = 1170
        draw.text((panel_x, 50), "MULTIMODAL JEV", fill="#ffffff", font=title_font)
        draw.text((panel_x, 105), "Question: Which bird species?", fill="#b9c4d6", font=small_font)
        draw.text((panel_x, 155), f"Prediction: {row['options'][predicted_index].replace('_', ' ')}", fill="#ffffff", font=font)
        draw.text((panel_x, 200), f"Ground truth: {row['class_name'].split('.', 1)[-1].replace('_', ' ')}", fill="#46e37b", font=font)
        draw.text((panel_x, 265), "Option probabilities", fill="#ffffff", font=font)
        for choice_index, (option, probability) in enumerate(zip(row["options"], probabilities)):
            y = 320 + choice_index * 92
            label = option.replace('_', ' ')
            draw.text((panel_x, y), f"{choice_index}: {label[:24]}", fill="#dce4f2", font=small_font)
            draw.rectangle((panel_x, y + 34, panel_x + 320, y + 55), fill="#26344d")
            draw.rectangle((panel_x, y + 34, panel_x + int(320 * probability), y + 55), fill="#5aa9ff" if choice_index != predicted_index else "#ffbd59")
            draw.text((panel_x + 335, y + 28), f"{probability:.3f}", fill="#ffffff", font=small_font)
        draw.text((panel_x, 720), "Red: predicted box", fill="#ff4d5e", font=small_font)
        draw.text((panel_x, 760), "Green: ground-truth box", fill="#46e37b", font=small_font)
        draw.text((panel_x, 825), f"CUB test image {index + 1}/{len(rows)}", fill="#7f8da6", font=small_font)
        canvas.save(args.work_dir / f"frame-{index:04d}.png")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None and os.environ.get("CONDA_PREFIX"):
        candidate = Path(os.environ["CONDA_PREFIX"]) / "bin" / "ffmpeg"
        if candidate.exists():
            ffmpeg = str(candidate)
    if ffmpeg is None:
        # Some older conda installs keep package executables under pkgs/.
        candidates = sorted(Path.home().glob("anaconda3/pkgs/ffmpeg-*/bin/ffmpeg"))
        if candidates:
            ffmpeg = str(candidates[-1])
    if ffmpeg is not None:
        try:
            subprocess.run([
                ffmpeg, "-y", "-loglevel", "error", "-framerate", str(1 / args.seconds_per_frame),
                "-i", str(args.work_dir / "frame-%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-vf", "format=yuv420p", str(args.output),
            ], check=True)
        except subprocess.CalledProcessError:
            ffmpeg = None
    if ffmpeg is None:
        import cv2
        frames = [Image.open(args.work_dir / f"frame-{i:04d}.png").convert("RGB") for i in range(len(rows))]
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 1 / args.seconds_per_frame, frames[0].size)
        for frame in frames:
            writer.write(cv2.cvtColor(__import__("numpy").array(frame), cv2.COLOR_RGB2BGR))
        writer.release()
    print(f"{args.output} | avg_inference_seconds_per_image={sum(inference_seconds) / len(inference_seconds):.3f}")


if __name__ == "__main__":
    main()
