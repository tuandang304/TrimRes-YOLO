# Results behind the paper

Run tags inside the files use the development name of the model: `access1` = TrimRes-YOLO, `ab1`…`ab8` = the
leave-one-out arms, `yolo26n_l4` / `yolo26n_1280` / `yolo26n_sr` = the same-recipe YOLO26n control (seed 0),
`pub_*` = public checkpoints.

## `runs/<model>/seed<k>/`

One folder per trained run (100 epochs, 1280×1280, one NVIDIA L4).

- `eval_result.json`: ignore-aware (`ignore_aware`) and standard (`std`) COCO metrics on the validation split
  (`valid@1280`), and test-dev (`test@1280`) where the run evaluated it, plus parameters and GFLOPs.
- `train_results.csv`: the per-epoch Ultralytics log (losses, validator mAP, precision, recall). The per-epoch validator
  does not filter ignore regions, so its mAP is lower than the ignore-aware value.

| Folder | Paper row |
|---|---|
| `trimres` | TrimRes-YOLO |
| `yolo26n_control` | YOLO26n, same recipe |
| `yolov10n` | YOLOv10n, same recipe |
| `ab1_keep_stride64_pool` … `ab8_one_to_one_head` | AB1 … AB8 |

Notes, matching the table footnotes of the paper:

- `yolo26n_control/seed0`: the training host scored this checkpoint with an older inference setting (single-label
  NMS, 500 detections); `eval_result.json` is the re-score with the unified evaluator (30.58 AP), which the paper uses.
- `yolo26n_control/seed1`, `seed2`, `ab1…/seed0`, `ab6…/seed1`, `ab8…/seed0` and `seed1` are retrainings with the
  same configuration whose checkpoints were kept; the paper reports these runs.
- `ab3…/seed0`, `ab4…/seed0`, `ab5…/seed0` have no `eval_result.json`: their validation metrics come from
  `evaluation_sessions/valid_public_checkpoints_and_ablation_seed0.json` (tags `ab3_c3k2`, `ab4_softmax`, `ab5_norep`).

## `evaluation_sessions/`

Separate scoring sessions of saved checkpoints with the same evaluator (`multi_md1000` = the paper's setting):

- `valid_public_checkpoints_and_ablation_seed0.json`: validation scores of the public YOLO26n, YOLO11n, YOLOv10n and
  RF-DETR-Nano checkpoints, and of the seed-0 checkpoints of AB3, AB4 and AB5.
- `valid_trimres_per_class.json`: validation per-class AP of the three TrimRes-YOLO checkpoints (Table 8 of the paper);
  their overall AP here agrees with the training-host scores within 0.01.
- `testdev_trimres_and_ablation_seed0.json`: test-dev scores, including per-class AP, of the three TrimRes-YOLO
  checkpoints.

## `latency/`

- `t4_rounds.json`: NVIDIA T4, eager PyTorch FP16, five interleaved rounds per model (`e2e_round_ms`, `fwd_round_ms`).
- `orin_nano_rounds.jsonl`, `orin_nano_summary.csv`: Jetson Orin Nano 8 GB (Super, MAXN_SUPER), eager PyTorch FP16 and
  TensorRT FP16 rounds, and peak memory.
