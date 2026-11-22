# TrimRes-YOLO

Code, configurations, trained weights and raw results for

> **TrimRes-YOLO: A two-scale, low-rank nano detector for tiny objects in high-resolution UAV imagery**
> Dang Thanh Tuan, Le Phu-Nguyen, Nguyen Thien Thanh, Phan Anh Minh. Manuscript submitted to *PLOS ONE*.

TrimRes-YOLO starts from COCO-pretrained YOLO26n and asks how much of a nano detector can be pruned or factorized at a
fixed 1280×1280 input. It detects at strides 8 and 16 only, downsamples through low-rank convolutions initialized by
truncated SVD, and rebuilds the stride-32 stage with low-rank projections, hybrid depthwise/partial-linear-attention
bottlenecks and ReLU-kernel linear attention that reuses the pretrained attention projections. A re-parameterized neck
block adds a training-only branch that is folded away at export.

## Main results (VisDrone2019-DET, 1280×1280, three seeds, mean ± s.d.)

| Model | Params (M) | GFLOPs | Val AP | Test-dev AP | T4 PyTorch FP16 e2e (ms) | Orin Nano TensorRT FP16 e2e / fwd (ms) |
|---|---|---|---|---|---|---|
| YOLO26n, same recipe | 2.39 | 22.76 | 30.51 ± 0.11 | 23.57 ± 0.17 | 18.61 | 25.53 / 11.96 |
| YOLOv10n, same recipe | 2.28 | 27.14 | 30.50 ± 0.38 | 23.69 ± 0.11 | 16.59 | 25.44 / 11.96 |
| **TrimRes-YOLO** | **1.30** | **15.94** | 29.61 ± 0.16 | 22.88 ± 0.18 | 19.33 | **23.31 / 9.80** |

AP is COCO-style mAP@0.50:0.95 with the VisDrone ignore regions filtered (see *Evaluation protocol*). TrimRes-YOLO is a
cheaper operating point (46% fewer parameters, 30% fewer FLOPs), 0.9 AP below the same-recipe YOLO26n control. Whether
the FLOP reduction lowers latency depends on the runtime: it is slower in eager PyTorch on a T4 and faster with
TensorRT on a Jetson Orin Nano. The leave-one-out ablation (AB1–AB8, three seeds each) is in `configs/ablations/` and
`results/runs/`.

## Repository layout

```
bench/                 training, evaluation and latency code (Python package; module paths match the released checkpoints)
  pra.py               model builder, COCO weight inheritance, SVD-initialized low-rank stride-2 convolutions
  access.py            TrimRes block changes: low-rank deep convs, HybridBottleneck, linear-attention C2PSA, RepConv
  train.py, main.py    fine-tuning entry point (python -m bench.main --config ...)
  data.py              VisDrone2019-DET download and conversion (COCO json + ignore regions + YOLO labels)
  predict.py, evaluate.py   multi-label inference and the ignore-aware COCO evaluator
  measure.py           interleaved latency rounds (PyTorch and TensorRT) and re-scoring of saved checkpoints
models/                model YAMLs (TrimRes-YOLO and the ablation graphs that differ from it)
configs/               one config per experiment: trimres, baselines/, ablations/ab1..ab8, latency, re-scoring
results/               per-run evaluation results and training curves for every number in the paper (see results/README.md)
scripts/setup_jetson.sh   Jetson Orin Nano environment (PyTorch + TensorRT, MAXN_SUPER, fixed clocks)
```

## Installation

Python 3.10–3.12 with a CUDA build of PyTorch, then:

```bash
pip install -r requirements.txt
```

All models were trained with Ultralytics 8.4.192; other versions may change the results. The first run downloads
`yolo26n.pt` (COCO weights) and VisDrone2019-DET (from the Ultralytics asset mirror) and converts it under `data_root`
(default `/tmp/visdrone/data`; change it in the config).

## Training

One seed per process, one GPU each (the paper used one NVIDIA L4 per seed, about 4 h for 100 epochs):

```bash
python -m bench.main --config configs/trimres.yaml --seed 0            # TrimRes-YOLO
python -m bench.main --config configs/baselines/yolo26n.yaml --seed 0  # same-recipe YOLO26n control
python -m bench.main --config configs/baselines/yolov10n.yaml --seed 0
python -m bench.main --config configs/ablations/ab3.yaml --seed 0      # AB1 ... AB8
```

Outputs go to `$OUTPUT_DIR` (default `./outputs`): Ultralytics training files, detection files, and
`result_<tag>.json` with validation and test-dev metrics. Each training run prints its parameter count and the
retained SVD energy of every factorized layer. Deterministic mode fixes the seeds, but runs on different machines are
not bit-identical; repeating a seed changed AP by 0.11–0.27 points in our runs.

## Released weights and re-scoring

The GitHub release `v1.0` holds the three TrimRes-YOLO checkpoints (`trimres_s0.pt`, `trimres_s1.pt`,
`trimres_s2.pt`, best validation fitness, about 3 MB each) and their validation detection files. To re-score them:

```bash
mkdir -p weights && cd weights
gh release download v1.0 --repo tuandang304/TrimRes-YOLO --pattern 'trimres_s*.pt'   # or download from the release page
cd .. && python -m bench.main --config configs/evaluate_released.yaml
```

The checkpoints are standard Ultralytics checkpoints whose custom modules live in `bench/`, so load them from the
repository root (`from ultralytics import YOLO; YOLO("weights/trimres_s0.pt")`).

## Evaluation protocol

Inference uses confidence 0.001, NMS IoU 0.7, multi-label assignment and at most 1000 detections per image. Before COCO
matching, every ground-truth box and prediction whose intersection with a VisDrone ignore region (categories 0 and 11)
exceeds 50% of its own area is removed, following the official VisDrone devkit. COCOeval then scores up to 500
detections per image. Scale bins: tiny < 16², small < 32² (COCO definition, includes tiny), medium 32²–96², large ≥ 96².

## Latency

```bash
python -m bench.main --config configs/latency_t4.yaml           # NVIDIA T4, eager PyTorch FP16
bash scripts/setup_jetson.sh && python -m bench.main --config configs/latency_orin_nano.yaml   # Jetson Orin Nano
```

Batch 1, 1280 input, five interleaved rounds with the model order shuffled per round; each model is reloaded and warmed
up for 30 iterations per round. End to end = pre-processing + forward + NMS at confidence 0.25 over 200 validation
images; forward only = a fixed 1×3×1280×1280 FP16 tensor timed with CUDA events. On the Orin Nano, TensorRT FP16 engines
are built on the device. The configs expect checkpoints in `weights/<tag>.pt`.

## Results

`results/` contains, for every trained run behind the paper's tables, the evaluation result and the per-epoch training
curve, plus the separate evaluation and latency sessions. See [results/README.md](results/README.md).

## License

AGPL-3.0, the license of Ultralytics, on which this code builds. VisDrone2019-DET is subject to its own terms of use
(https://github.com/VisDrone/VisDrone-Dataset).

## Citation

If you use this code, please cite the paper (citation details will be added on publication):

```bibtex
@article{trimresyolo,
  title  = {TrimRes-YOLO: A two-scale, low-rank nano detector for tiny objects in high-resolution UAV imagery},
  author = {Dang, Thanh Tuan and Le, Phu-Nguyen and Nguyen, Thien Thanh and Phan, Anh Minh},
  note   = {Manuscript submitted to PLOS ONE},
  year   = {2026}
}
```
