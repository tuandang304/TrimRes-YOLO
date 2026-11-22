"""TrimRes-YOLO block changes, applied after the COCO-pretrained YOLO26n tensors are loaded.

Config: arch["access"] = list of {op: ..., layers: [...], ...}. Each op builds its block from the pretrained weights it
replaces, so the modified network starts close to the pretrained function.

ops
- lowrank_conv  3x3 Conv -> LowRankConv of rank `rank` (3x3 conv to r channels, then 1x1 Conv-BN-act), truncated-SVD init.
- lowrank_pw    a 1x1 Conv attribute (e.g. SPPF.cv2) -> rank-`rank` pair of 1x1 convs, truncated-SVD init.
- hybrid_c3k    in the C3k2 at `layers`, inner C3k bottleneck `idx` becomes a HybridBottleneck: local mixer + partial
                single-head linear attention + ConvFFN, each a zero-init residual branch (identity at step 0).
                `replace: true` drops the pretrained bottleneck (local mixer = zero-init depthwise 5x5).
- linattn_psa   every PSABlock attention in the C2PSA at `layers` uses ReLU-kernel linear attention with the
                inherited qkv / pe / proj weights (softmax replaced, weights kept).
- repconv       every 3x3 conv inside the inner bottlenecks of the C3k2 at `layers` -> RepConv (3x3 inherited +
                1x1 branch with zero BN scale and shift), folded into one 3x3 conv by model.fuse() at export.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Attention, Bottleneck, C2PSA, C3k, C3k2
from ultralytics.nn.modules.conv import Conv, RepConv


def _zero_bn(conv: Conv) -> Conv:
    nn.init.zeros_(conv.bn.weight)
    return conv


def _linear_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q,k: (B, h, d, N); v: (B, h, e, N) -> (B, h, e, N). ReLU kernel, computed in fp32."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v = F.relu(q.float()) + 1e-6, F.relu(k.float()) + 1e-6, v.float()
        kv = torch.einsum("bhdn,bhen->bhde", k, v)
        z = 1.0 / (torch.einsum("bhdn,bhd->bhn", q, k.sum(-1)) + 1e-6)
        return torch.einsum("bhdn,bhde->bhen", q, kv) * z.unsqueeze(2)


class LinearAttention(Attention):
    """Ultralytics Attention with softmax replaced by ReLU-kernel linear attention; keeps qkv/pe/proj (inherited)."""

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q, k, v = self.qkv(x).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)
        out = _linear_attn(q, k, v).to(v.dtype).reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(out)



class PartialLinAttn(nn.Module):
    """Single-head linear attention on the first c/r channels (SHViT-style partial channels); zero-init output."""

    def __init__(self, c: int, ratio: int = 4, d: int = 16):
        super().__init__()
        self.cp = max(8, c // ratio)
        self.d = d
        self.qkv = Conv(self.cp, 2 * d + self.cp, 1, act=False)
        self.proj = _zero_bn(Conv(c, c, 1, act=False))

    def forward(self, x):
        B, C, H, W = x.shape
        xa, xr = x[:, : self.cp], x[:, self.cp:]
        q, k, v = self.qkv(xa).flatten(2).split([self.d, self.d, self.cp], dim=1)
        a = _linear_attn(q[:, None], k[:, None], v[:, None])[:, 0].to(x.dtype).reshape(B, self.cp, H, W)
        return self.proj(torch.cat([a, xr], 1))


class ConvFFN(nn.Module):
    """1x1 expand -> DW 3x3 -> 1x1 project (zero-init)."""

    def __init__(self, c: int, e: float = 2.0):
        super().__init__()
        h = int(c * e)
        self.fc1 = Conv(c, h, 1)
        self.dw = Conv(h, h, 3, g=h)
        self.fc2 = _zero_bn(Conv(h, c, 1, act=False))

    def forward(self, x):
        return self.fc2(self.dw(self.fc1(x)))


class HybridBottleneck(nn.Module):
    """MetaFormer-style hybrid: x + local(x); x + attn(x); x + ffn(x). local = pretrained bottleneck or zero-init DW 5x5."""

    def __init__(self, c: int, local: nn.Module | None, ratio: int = 4, d: int = 16, e: float = 2.0):
        super().__init__()
        self.local = local if local is not None else _zero_bn(Conv(c, c, 5, g=c, act=False))
        self.keep_bottleneck = local is not None
        self.attn = PartialLinAttn(c, ratio, d)
        self.ffn = ConvFFN(c, e)

    def forward(self, x):
        x = self.local(x) if self.keep_bottleneck else x + self.local(x)  # a pretrained Bottleneck already adds its shortcut
        x = x + self.attn(x)
        return x + self.ffn(x)



def to_repconv(conv: Conv) -> RepConv:
    """3x3 Conv -> RepConv with the conv as branch 1 (inherited) and a zero-init 1x1 branch; act kept."""
    c1, c2 = conv.conv.in_channels, conv.conv.out_channels
    s, g = conv.conv.stride[0], conv.conv.groups
    r = RepConv(c1, c2, 3, s, 1, g=g, act=conv.act)
    r.conv1.load_state_dict(conv.state_dict())
    nn.init.zeros_(r.conv2.bn.weight)
    nn.init.zeros_(r.conv2.bn.bias)
    return r.to(conv.conv.weight.device)



def _inner_c3k(m: nn.Module) -> C3k:
    assert isinstance(m, C3k2) and isinstance(m.m[0], C3k), type(m)
    return m.m[0]


def apply(model, specs: list) -> None:
    seq = model.model
    old_bn = {id(b) for b in model.modules() if isinstance(b, nn.BatchNorm2d)}
    for sp in specs:
        op = sp["op"]
        for li in sp["layers"]:
            m = seq[li]
            if op == "linattn_psa":
                assert isinstance(m, C2PSA), (li, type(m))
                for blk in m.m:
                    blk.attn.__class__ = LinearAttention
            elif op == "hybrid_c3k":
                ck = _inner_c3k(m)
                idx = sp.get("idx", 1)
                b = ck.m[idx]
                c = b.cv2.conv.out_channels
                local = None if sp.get("replace") else b
                ck.m[idx] = HybridBottleneck(c, local, sp.get("ratio", 4), sp.get("d", 16), sp.get("e", 2.0)).to(b.cv1.conv.weight.device)
            elif op == "repconv":
                if isinstance(m, Conv):  # a stand-alone 3x3 conv layer (e.g. the bottom-up downsampler)
                    r = to_repconv(m)
                    r.i, r.f, r.type, r.np = m.i, m.f, "ultralytics.nn.modules.conv.RepConv", sum(p.numel() for p in r.parameters())
                    seq[li] = r
                else:  # every 3x3 conv inside the inner bottlenecks of a C3k2
                    for b in _inner_c3k(m).m:
                        for name in ("cv1", "cv2"):
                            cv = getattr(b, name)
                            if isinstance(cv, Conv) and cv.conv.kernel_size == (3, 3):
                                setattr(b, name, to_repconv(cv))
            elif op == "lowrank_conv":  # 3x3 Conv layer -> LowRankConv of rank r, SVD of the inherited kernel
                from .pra import LowRankConv
                new = LowRankConv(m.conv.in_channels, m.conv.out_channels, sp["rank"]).to(m.conv.weight.device)
                if m.conv.stride != (1, 1):
                    new.a.stride = m.conv.stride
                e = new.init_from(m)
                new.i, new.f, new.type = m.i, m.f, "bench.pra.LowRankConv"
                seq[li] = new
                print(f"[access] layer {li}: SVD rank {sp['rank']} retains {e:.3f} energy", flush=True)
            elif op == "lowrank_pw":  # a 1x1 Conv attribute (e.g. SPPF.cv2) -> rank-r pair, SVD of the inherited weight
                cv = getattr(m, sp["attr"])
                W = cv.conv.weight.detach().float()[:, :, 0, 0]
                U, S_, Vh = torch.linalg.svd(W, full_matrices=False)
                r = sp["rank"]; s = S_[:r].sqrt()
                a1 = nn.Conv2d(W.shape[1], r, 1, bias=False); b1 = Conv(r, W.shape[0], 1, act=cv.act)
                with torch.no_grad():
                    a1.weight.copy_((s[:, None] * Vh[:r])[:, :, None, None]); b1.conv.weight.copy_((U[:, :r] * s[None])[:, :, None, None])
                b1.bn.load_state_dict(cv.bn.state_dict())
                setattr(m, sp["attr"], nn.Sequential(a1, b1).to(W.device))
                print(f"[access] layer {li}.{sp['attr']}: SVD rank {r} retains {float((S_[:r]**2).sum()/(S_**2).sum()):.3f} energy", flush=True)
            else:
                raise ValueError(op)
            if hasattr(seq[li], "np"):
                seq[li].np = sum(p.numel() for p in seq[li].parameters())
        print(f"[access] {op} at layers {sp['layers']}", flush=True)
    for mod in model.modules():  # only BNs created here get Ultralytics' settings (eps 1e-3, momentum 0.03); existing BNs keep theirs
        if isinstance(mod, nn.BatchNorm2d) and id(mod) not in old_bn:
            mod.eps, mod.momentum = 1e-3, 0.03
