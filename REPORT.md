# Multimodal JEV with Species Probabilities and Bounding Boxes

## Short assessment

The model reached **96.49% four-way choice accuracy** and **0.535 mean IoU** on validation after three epochs.

An IoU of 0.535 is a useful prototype result: the model is generally localizing the bird, but the boxes are not yet tight enough to call this strong grounding performance. A stronger next target would be roughly 0.65+ mIoU, together with AP50, AP75, and the percentage of examples above IoU thresholds. Classification is already strong, although it is measured on four candidate species rather than all 200 classes.

## Model and task

The system uses Qwen2.5-VL-3B-Instruct. Each example contains an image and a question such as “Which bird species is shown?” Four candidate species are included in the prompt and represented by numeric labels `0`, `1`, `2`, and `3`, so JEV can read their next-token logits.

The model has two outputs:

1. A four-way option distribution from the logits of tokens `0`–`3`.
2. A four-number bounding-box prediction for normalized `xyxy` coordinates.

The base model is adapted with LoRA. A trainable box head is attached to the final multimodal hidden representation. Its sigmoid output is canonicalized so `x1 <= x2` and `y1 <= y2`.

## CUB processing

CUB-200-2011 contains 11,788 bird images across 200 species, with one bird bounding box per image. The original box format is:

```text
x, y, width, height
```

The formatter converts it to normalized corners:

```text
x1 = x / image_width
y1 = y / image_height
x2 = (x + width) / image_width
y2 = (y + height) / image_height
```

Each JSONL record stores an image path, four candidate species, the correct option index, and a normalized box:

```json
{
  "image": "/path/to/image.jpg",
  "options": ["species_a", "species_b", "species_c", "species_d"],
  "answer_index": 2,
  "box": [0.12, 0.08, 0.77, 0.99]
}
```

The official CUB test split was preserved. The official training split was divided into:

```text
training:   5,395 images
validation:   599 images
test:       5,794 images
```

For every image, the correct species is combined with three randomly sampled distractors. This validates the JEV mechanism, but is easier than evaluating all 200 species.

The formatter is `prepare_cub.py` and writes:

```text
data/cub_records/train.jsonl
data/cub_records/validation.jsonl
data/cub_records/test.jsonl
```

## Loss function

Let `z` be the four selected option logits. The option distribution is:

```text
p_options = softmax(z)
L_option = CrossEntropy(z, correct_option)
```

The box head predicts normalized coordinates `b_hat` and the target is `b`:

```text
L_box = SmoothL1(b_hat, b)
L_total = L_option + 5.0 * L_box
```

The weight `5.0` is the current experiment setting, not a tuned optimum. The box corners are ordered before the loss and inference output.

LoRA is applied to the multimodal model’s attention and MLP projection layers. The box head remains fully trainable. Training uses DistributedDataParallel across eight A100 40 GB GPUs, with one image per GPU and gradient accumulation of eight.

## Training

The completed `gpu5` run used:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 train_multimodal_jev.py \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --train data/cub_records/train.jsonl \
  --validation data/cub_records/validation.jsonl \
  --output runs/cub-jev-8gpu-v2 \
  --lora --epochs 3 --batch-size 1 \
  --gradient-accumulation 8 --lr 1e-5 --box-weight 5.0
```

The resulting files are:

```text
runs/cub-jev-8gpu-v2/epoch-3-adapter/
runs/cub-jev-8gpu-v2/epoch-3-box.pt
```

## Results

Epoch-3 validation results:

```text
choice_accuracy: 0.9649
mean_iou:        0.5347
```

Choice accuracy is four-way accuracy; it does not mean 96.49% accuracy over all 200 CUB species.

Mean IoU is:

```text
IoU = intersection_area / union_area
```

Classification is currently stronger than localization. This is expected because the box branch is a small regression head over the final multimodal representation, not a dedicated detection decoder.

## Inference

`infer_multimodal_jev.py` loads the base model, LoRA adapter, and box checkpoint. It then loads the image, builds the multimodal prompt, runs one forward pass, applies softmax over the four selected logits, and returns normalized plus pixel-space `xyxy` coordinates.

Example:

```bash
python infer_multimodal_jev.py \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter runs/cub-jev-8gpu-v2/epoch-3-adapter \
  --box-checkpoint runs/cub-jev-8gpu-v2/epoch-3-box.pt \
  --record data/cub_records/test.jsonl \
  --record-id cub_00001
```

The output includes the selected option, probabilities, normalized box, and pixel box. The inference result for `cub_00001` selected `black_footed_albatross` with probability `0.999484` and achieved approximately `0.55 IoU` against that image’s ground-truth box.

## Demo video

`make_demo_video.py` renders annotated frames and combines them into an MP4. The demo displays option probabilities, the predicted box in red, and the ground-truth box in green.

## Recommended next steps

1. Evaluate all 200 species, not only four sampled options.
2. Report AP50, AP75, and the percentage of boxes with IoU >= 0.5.
3. Add GIoU or CIoU loss to Smooth L1.
4. Use a visual-token pooling strategy instead of only the final text position.
5. Add more box-focused examples and stronger image augmentation.
6. Add explicit grounding questions such as “Where is the bird?”
7. Compare against a dedicated detector or grounding model.
