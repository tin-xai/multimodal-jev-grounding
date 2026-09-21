#!/usr/bin/env python3
"""Train a multimodal JEV-style option scorer plus one box regression head.

This is an intentionally small reference trainer. It expects a Transformers
vision-language model whose processor accepts image + text and whose forward
pass can return hidden states. Qwen2.5-VL and Gemma 3-family models are
reasonable starting points, subject to the installed Transformers version.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup


MARKERS = ["0", "1", "2", "3"]


class CubDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        option_text = " ".join(f"{i}: {name.replace('_', ' ')}" for i, name in enumerate(row["options"]))
        prompt = f"Which bird species is shown? {option_text} Answer with one digit: 0, 1, 2, or 3. Answer:"
        return {
            "image": Image.open(row["image"]).convert("RGB"),
            "prompt": prompt,
            "answer_index": row["answer_index"],
            "box": torch.tensor(row["box"], dtype=torch.float32),
        }


class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        conversations = [
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": x["prompt"]},
            ]}]
            for x in examples
        ]
        if hasattr(self.processor, "apply_chat_template"):
            prompts = self.processor.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=True
            )
        else:
            prompts = [x["prompt"] for x in examples]
        images = [x["image"] for x in examples]
        batch = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        batch["answer_index"] = torch.tensor([x["answer_index"] for x in examples], dtype=torch.long)
        batch["box_target"] = torch.stack([x["box"] for x in examples])
        return batch


class JEVBoxModel(torch.nn.Module):
    def __init__(self, base, marker_ids):
        super().__init__()
        self.base = base
        self.marker_ids = marker_ids
        hidden = base.config.hidden_size
        self.box_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 4),
        )

    def forward(self, **batch):
        output = self.base(**batch, output_hidden_states=True, return_dict=True)
        lengths = batch["attention_mask"].sum(dim=1) - 1
        last = output.logits[torch.arange(output.logits.size(0), device=output.logits.device), lengths]
        option_logits = last[:, self.marker_ids]
        hidden = output.hidden_states[-1][torch.arange(output.hidden_states[-1].size(0), device=output.logits.device), lengths]
        hidden = hidden.to(self.box_head[0].weight.dtype)
        raw_boxes = self.box_head(hidden).sigmoid()
        left_top = torch.minimum(raw_boxes[:, :2], raw_boxes[:, 2:])
        right_bottom = torch.maximum(raw_boxes[:, :2], raw_boxes[:, 2:])
        boxes = torch.cat([left_top, right_bottom], dim=-1)
        return option_logits, boxes


def marker_token_ids(processor, tokenizer):
    ids = []
    for marker in MARKERS:
        encoded = tokenizer(marker, add_special_tokens=False).input_ids
        if len(encoded) != 1:
            raise ValueError(f"{marker} must be one token for direct JEV scoring; got {encoded}")
        ids.append(encoded[0])
    return ids


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def evaluate(model, loader, device, box_weight):
    model.eval()
    totals = {"loss": 0.0, "choice": 0, "count": 0, "iou": 0.0}
    with torch.no_grad():
        for batch in loader:
            targets = batch.pop("answer_index").to(device)
            box_target = batch.pop("box_target").to(device)
            logits, boxes = model(**move_batch(batch, device))
            loss = F.cross_entropy(logits, targets) + box_weight * F.smooth_l1_loss(boxes, box_target)
            pred = logits.argmax(-1)
            intersection = torch.minimum(boxes[:, 2:], box_target[:, 2:]) - torch.maximum(boxes[:, :2], box_target[:, :2])
            intersection = intersection.clamp_min(0).prod(-1)
            area_a = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
            area_b = (box_target[:, 2:] - box_target[:, :2]).clamp_min(0).prod(-1)
            iou = intersection / (area_a + area_b - intersection + 1e-6)
            n = len(targets)
            totals["loss"] += loss.item() * n
            totals["choice"] += (pred == targets).sum().item()
            totals["iou"] += iou.sum().item()
            totals["count"] += n
    return {"loss": totals["loss"] / totals["count"], "choice_accuracy": totals["choice"] / totals["count"], "mean_iou": totals["iou"] / totals["count"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--train", type=Path, default=Path("data/cub_records/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/cub_records/validation.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/cub-jev"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--box-weight", type=float, default=5.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--lora", action="store_true", help="Train LoRA adapters in the multimodal base")
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    is_main = rank == 0
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    base = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype)
    ids = marker_token_ids(processor, tokenizer)
    if args.lora:
        from peft import LoraConfig, get_peft_model
        base = get_peft_model(
            base,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=2 * args.lora_rank,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    model = JEVBoxModel(base, ids).to(device)
    if args.freeze_base and not args.lora:
        for parameter in model.base.parameters():
            parameter.requires_grad = False
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    train_dataset = CubDataset(args.train)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler, collate_fn=Collator(processor))
    val_loader = DataLoader(CubDataset(args.validation), batch_size=args.batch_size, shuffle=False, collate_fn=Collator(processor))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = max(1, (len(train_loader) * args.epochs) // args.gradient_accumulation)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, steps // 10), steps)
    args.output.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            targets = batch.pop("answer_index").to(device)
            box_target = batch.pop("box_target").to(device)
            logits, boxes = model(**move_batch(batch, device))
            loss = (F.cross_entropy(logits, targets) + args.box_weight * F.smooth_l1_loss(boxes, box_target)) / args.gradient_accumulation
            loss.backward()
            if (step + 1) % args.gradient_accumulation == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
        if distributed:
            dist.barrier()
        if is_main:
            saved_model = model.module if distributed else model
            metrics = evaluate(saved_model, val_loader, device, args.box_weight)
            print(json.dumps({"epoch": epoch + 1, **metrics}), flush=True)
            if args.lora:
                saved_model.base.save_pretrained(args.output / f"epoch-{epoch + 1}-adapter")
                torch.save({"box_head": saved_model.box_head.state_dict(), "marker_ids": ids, "args": vars(args)}, args.output / f"epoch-{epoch + 1}-box.pt")
            else:
                torch.save({"model": saved_model.state_dict(), "marker_ids": ids, "args": vars(args)}, args.output / f"epoch-{epoch + 1}.pt")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
