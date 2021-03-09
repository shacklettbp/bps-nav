import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegNetXBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_width, stride=1):
        super().__init__()

        downsample = []
        if stride != 1 or in_channels != out_channels:
            if stride != 1:
                downsample.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

            downsample += [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(out_channels // group_width, out_channels),
            ]

        self.downsample = nn.Sequential(*downsample)

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(out_channels // group_width, out_channels),
            nn.ReLU(True),
            nn.Conv2d(
                out_channels,
                out_channels,
                groups=out_channels // group_width,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False,
            ),
            nn.GroupNorm(out_channels // group_width, out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(out_channels // group_width, out_channels),
        )

        self.relu = nn.ReLU(True)

    def _combine(self, x, skip):
        return self.relu(x + skip)

    def forward(self, x):
        skip = self.downsample(x)
        x = self.convs(x)

        return self._combine(x, skip)


class SE(nn.Module):
    def __init__(self, planes, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(planes, int(planes / r)),
            nn.ReLU(True),
            nn.Linear(int(planes / r), planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.squeeze(x)
        x = x.view(b, c)
        x = self.excite(x)

        return x.view(b, c, 1, 1)


class RegNetYBlock(RegNetXBlock):
    def __init__(self, in_channels, out_channels, group_width, stride=1):
        super().__init__(in_channels, out_channels, group_width, stride=stride)

        self.se = SE(out_channels)

    def _combine(self, x, skip):
        return self.relu(x * self.se(x) + skip)


class SpaceToDepth(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(N, C * 4, H // 2, W // 2)


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, gs):
    """Adjusts the compatibility of widths and groups."""
    gs = [min(g, w) for g, w in zip(gs, ws)]
    ws = [quantize_float(w, g) for w, g in zip(ws, gs)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class RegNet(nn.Module):
    def __init__(
        self,
        in_channels,
        initial_channels,
        stage_widths,
        stage_depths,
        stage_group_widths,
        use_se,
    ):
        super().__init__()
        self.current_channels = initial_channels

        self.stem = nn.Sequential(
            SpaceToDepth(),
            nn.Conv2d(
                in_channels * 4, self.current_channels, kernel_size=1, bias=False,
            ),
            nn.GroupNorm(
                self.current_channels // stage_group_widths[0], self.current_channels
            ),
            nn.ReLU(True),
        )

        block = RegNetYBlock if use_se else RegNetXBlock
        self.stages = nn.ModuleList()

        for stage_idx in range(len(stage_depths)):
            stage_blocks = []
            width = stage_widths[stage_idx]
            group_width = stage_group_widths[stage_idx]
            for block_idx in range(stage_depths[stage_idx]):
                stride = 2 if block_idx == 0 else 1
                stage_blocks.append(
                    block(
                        self.current_channels,
                        width,
                        group_width=group_width,
                        stride=stride,
                    )
                )

                self.current_channels = width

            self.stages.append(nn.Sequential(*stage_blocks))

        self.final_channels = self.current_channels
        self.compression_stages = len(stage_depths) + 1
        self.final_spatial_compress = 1.0 / (2 ** self.compression_stages)

    def forward(self, x):
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        return x


def regnet(in_channels, w_a, w_0, w_m, d, gw, use_se=False):
    ws, *_ = generate_regnet(w_a, w_0, w_m, d)
    s_ws, s_ds = get_stages_from_blocks(ws, ws)
    s_gw = [gw for _ in s_ws]
    s_ws, s_gw = adjust_ws_gs_comp(s_ws, s_gw)
    w_0 = int(gw * round(w_0 / gw))
    return RegNet(in_channels, w_0, s_ws, s_ds, s_gw, use_se=use_se)


def regnetx_200mf(in_channels, *args, **kwargs):
    d = 13
    w_0 = 24
    w_a = 36
    w_m = 2.5
    return regnet(in_channels, w_a, w_0, w_m, d, gw=8)


def regnetx_400mf(in_channels, *args, **kwargs):
    d = 22
    w_0 = 24
    w_a = 24.48
    w_m = 2.54
    return regnet(in_channels, w_a, w_0, w_m, d, gw=16)


def regnetx_600mf(in_channels, *args, **kwargs):
    d = 16
    w_0 = 48
    w_a = 36.97
    w_m = 2.24
    return regnet(in_channels, w_a, w_0, w_m, d, gw=24)


def regnetx_800mf(in_channels, *args, **kwargs):
    d = 16
    w_0 = 56
    w_a = 35.73
    w_m = 2.28
    return regnet(in_channels, w_a, w_0, w_m, d, gw=16)


def regnety_200mf(in_channels, *args, **kwargs):
    d = 13
    w_0 = 24
    w_a = 36.44
    w_m = 2.49
    return regnet(in_channels, w_a, w_0, w_m, d, gw=8, use_se=True)


def regnety_400mf(in_channels, *args, **kwargs):
    d = 16
    w_0 = 48
    w_a = 27.89
    w_m = 2.09
    return regnet(in_channels, w_a, w_0, w_m, d, gw=16, use_se=True)


def regnety_600mf(in_channels, *args, **kwargs):
    d = 15
    w_0 = 48
    w_a = 32.54
    w_m = 2.32
    return regnet(in_channels, w_a, w_0, w_m, d, gw=16, use_se=True)


def regnety_800mf(in_channels, *args, **kwargs):
    d = 14
    w_0 = 56
    w_a = 38.84
    w_m = 2.4
    return regnet(in_channels, w_a, w_0, w_m, d, gw=16, use_se=True)


def regnety_3200mf(in_channels, *args, **kwargs):
    d = 21
    w_0 = 80
    w_a = 42.63
    w_m = 2.66
    return regnet(in_channels, w_a, w_0, w_m, d, gw=24, use_se=True)


def regnety_6400mf(in_channels, *args, **kwargs):
    d = 25
    w_0 = 112
    w_a = 33.22
    w_m = 2.27
    return regnet(in_channels, w_a, w_0, w_m, d, gw=72, use_se=True)
