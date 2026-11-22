"""TrimRes-YOLO model builder: YOLO26n-derived graph, explicit COCO weight inheritance, SVD-initialized low-rank convs.

An architecture is configured as `train.arch`:
  yaml:          repo-relative model YAML (models/*.yaml)
  head_c3:       width of the Detect classification branch, an int or one per level
  head_k:        kernel of the depthwise convs in the classification branch, one per level (default 3)
  stride_mult:   letterbox/grid multiple for the whole model (64 when the deep stage runs at stride 64, AB1)
  lowrank:       {layers, ratio} stride-2 convs replaced by LowRankConv with rank ratio * c_out (SVD init from COCO)
  inherit:       {ranges: [[lo, hi, offset], ...], detect_levels: [source level for each new Detect level]}
                 maps yolo26n.pt tensors model.{i} -> model.{i + offset} for lo <= i <= hi, and Detect level l <- level src.
  access:        block changes applied after inheritance (bench/access.py)
Every inherited tensor must match in shape; the count of transferred tensors is printed.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

try:
    from ultralytics.nn.modules.conv import Conv as _Conv
except Exception:  # pragma: no cover - ultralytics always present in runs
    _Conv = nn.Module
try:
    from ultralytics.nn.tasks import DetectionModel as _DetectionModel
except Exception:  # pragma: no cover - ultralytics always present in runs
    _DetectionModel = nn.Module









class LowRankConv(nn.Module):
    """3x3 stride-2 conv factorised as 3x3-s2 (c1->r, no BN) then Conv-BN-SiLU 1x1 (r->c2); init from SVD."""

    def __init__(self, c1: int, c2: int, r: int):
        super().__init__()
        self.a = nn.Conv2d(c1, r, 3, 2, 1, bias=False)
        self.b = _Conv(r, c2, 1, 1)

    def forward(self, x):
        return self.b(self.a(x))

    @torch.no_grad()
    def init_from(self, conv) -> float:
        """Copy the best rank-r approximation of conv.conv.weight; returns retained energy fraction."""
        W = conv.conv.weight.detach().float()
        c2, c1 = W.shape[:2]
        U, S, Vh = torch.linalg.svd(W.reshape(c2, -1), full_matrices=False)
        r = min(self.a.out_channels, S.numel())
        sq = S[:r].sqrt()
        a = (sq[:, None] * Vh[:r]).reshape(r, c1, 3, 3)
        b = (U[:, :r] * sq[None]).reshape(c2, r, 1, 1)
        if self.a.weight.shape == a.shape and self.b.conv.weight.shape == b.shape:
            self.a.weight.copy_(a)
            self.b.conv.weight.copy_(b)
            self.b.bn.load_state_dict(conv.bn.state_dict())
        else:  # widened layer: copy the COCO factors into the overlapping slice, keep the fresh init elsewhere
            self.a.weight[:r, :c1].copy_(a[:, :self.a.weight.shape[1]])
            self.b.conv.weight[:c2, :r].copy_(b[:self.b.conv.weight.shape[0]])
            for k, v in conv.bn.state_dict().items():
                t = getattr(self.b.bn, k)
                if t.ndim:
                    t[:v.shape[0]].copy_(v[:t.shape[0]])
        return float((S[:r] ** 2).sum() / (S ** 2).sum())


def surgery(model, arch: dict) -> None:
    from ultralytics.nn.modules.conv import Conv, DWConv
    seq = model.model
    if arch.get("lowrank"):  # random init here; the trainer re-inits from COCO via SVD (svd_init)
        for i in arch["lowrank"]["layers"]:
            old = seq[i]
            assert isinstance(old, Conv) and old.conv.stride == (2, 2), (i, old)
            c1, c2 = old.conv.in_channels, old.conv.out_channels
            new = LowRankConv(c1, c2, max(8, int(round(arch["lowrank"]["ratio"] * c2 / 8)) * 8))
            new.i, new.f, new.type = old.i, old.f, "bench.pra.LowRankConv"
            new.np = sum(p.numel() for p in new.parameters())
            seq[i] = new
    det = seq[-1]
    if arch.get("head_c3"):
        ch = [s[0][0].conv.in_channels for s in det.cv3]
        c3s = arch["head_c3"] if isinstance(arch["head_c3"], list) else [arch["head_c3"]] * len(ch)  # int or per level
        lite = nn.ModuleList(
            nn.Sequential(nn.Sequential(DWConv(x, x, k), Conv(x, c3, 1)),
                          nn.Sequential(DWConv(c3, c3, k), Conv(c3, c3, 1)),
                          nn.Conv2d(c3, det.nc, 1)) for x, c3, k in zip(ch, c3s, arch.get("head_k", [3] * len(ch))))
        det.cv3 = lite
        if getattr(det, "one2one_cv3", None) is not None:
            import copy
            det.one2one_cv3 = copy.deepcopy(lite)
    if hasattr(det, "bias_init"):
        det.bias_init()
    if arch.get("stride_mult"):
        # model.stride drives letterbox / rect-batch padding; Detect keeps its own per-level strides.
        model.stride = torch.tensor([float(arch["stride_mult"])])


def inherit(model, arch: dict, src: str = "yolo26n.pt") -> tuple[int, int]:
    """Copy COCO tensors from yolo26n.pt into `model` using the arch's index map. Returns (copied, total)."""
    from ultralytics.nn.tasks import load_checkpoint
    w, _ = load_checkpoint(src)
    sd_src = w.float().state_dict()
    inh = arch.get("inherit", {})
    off, max_src = inh.get("offset", 0), inh.get("max_src", 22)
    det_idx = len(model.model) - 1
    src_det = len(w.model) - 1
    levels = inh.get("detect_levels")
    mapped = {}
    for k, v in sd_src.items():
        mm = re.match(r"model\.(\d+)\.(.*)", k)
        if not mm:
            continue
        i, rest = int(mm.group(1)), mm.group(2)
        ranges = inh.get("ranges")  # [[lo, hi, offset], ...] for architectures with inserted layers
        if ranges:
            hit = [o for lo, hi, o in ranges if lo <= i <= hi]
            if hit:
                mapped[f"model.{i + hit[0]}.{rest}"] = v
        elif i <= max_src:
            mapped[f"model.{i + off}.{rest}"] = v
        if i == src_det and levels is not None:
            lm = re.match(r"((?:one2one_)?cv[23])\.(\d+)\.(.*)", rest)
            if lm:
                for new_l, src_l in enumerate(levels):
                    if src_l == int(lm.group(2)):
                        mapped[f"model.{det_idx}.{lm.group(1)}.{new_l}.{lm.group(3)}"] = v
    sd = model.state_dict()
    ok = {k: v for k, v in mapped.items() if k in sd and sd[k].shape == v.shape}
    n_partial = 0
    if inh.get("partial"):  # widened layers: copy the overlapping channels, keep the fresh init for the rest
        for k, v in mapped.items():
            if k in sd and k not in ok and sd[k].ndim == v.ndim and not k.startswith(f"model.{det_idx}."):
                t = sd[k].clone()
                cut = tuple(slice(0, min(a, b)) for a, b in zip(t.shape, v.shape))
                t[cut] = v[cut]
                ok[k] = t
                n_partial += 1
        print(f"[pra] partial copy of {n_partial} widened tensors", flush=True)
    model.load_state_dict(ok, strict=False)
    print(f"[pra] inherited {len(ok)}/{len(sd)} tensors from {src} "
          f"(mapped {len(mapped)}, shape-mismatch {sum(1 for k in mapped if k in sd and sd[k].shape != mapped[k].shape)})",
          flush=True)
    return len(ok), len(sd)


def svd_init(model, arch: dict) -> None:
    """Initialise every LowRankConv from the SVD of the corresponding COCO stride-2 conv."""
    if not arch.get("lowrank"):
        return
    from ultralytics.nn.tasks import load_checkpoint
    w, _ = load_checkpoint(arch.get("init", "yolo26n.pt"))
    src = w.float()
    inh = arch.get("inherit", {})
    for i in arch["lowrank"]["layers"]:
        lr = model.model[i]
        si = i
        for lo, hi, o in inh.get("ranges", []) or []:
            if lo <= i - o <= hi:
                si = i - o
        if not inh.get("ranges"):
            si = i - inh.get("offset", 0)
        energy = lr.init_from(src.model[si])
        print(f"[pra] layer {i}: SVD rank {lr.a.out_channels}/{lr.b.conv.out_channels} retains {energy:.3f} energy", flush=True)



def build(arch: dict, nc: int = 10, verbose: bool = False):
    from ultralytics.nn.tasks import DetectionModel
    m = DetectionModel(str(ROOT / arch["yaml"]), nc=nc, verbose=verbose)
    surgery(m, arch)
    return m


def make_trainer(arch: dict):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class TrimResTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = build(arch, nc=self.data["nc"])
            if arch.get("init", "yolo26n.pt"):
                inherit(model, arch, arch.get("init", "yolo26n.pt"))
                svd_init(model, arch)
            if arch.get("access"):  # block changes built from the inherited tensors
                from . import access
                access.apply(model, arch["access"])
            n_par = sum(p.numel() for p in model.parameters())
            print(f"[pra] arch={arch} params={n_par / 1e6:.3f}M", flush=True)
            return model

    return TrimResTrainer
