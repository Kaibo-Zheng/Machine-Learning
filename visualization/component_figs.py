"""Generate report method figures in a clean academic diagram style.

The report consumes PNG files, so the diagrams are drawn directly with Pillow.
The visual language intentionally mirrors ``doc/2.svg``: white background,
thin orthogonal arrows, low-saturation component blocks, compact labels, and
plain panel headings.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


OUT = Path("doc")
SCALE = 2

BG = "#ffffff"
INK = "#111827"
MUTED = "#667386"
ARROW = "#111827"
GRID = "#d7dee7"

TRAIN_FILL, TRAIN_STROKE = "#fff4c8", "#c4a854"
NOTR_FILL, NOTR_STROKE = "#d9efeb", "#9bbdb8"
NEUT_FILL, NEUT_STROKE = "#f3f6f9", "#aeb9c6"
LOSS_FILL, LOSS_STROKE = "#f8e4e8", "#c47b87"
GROUP_FILL, GROUP_STROKE = "#fffdf0", "#d8c06a"

CLASS_COLORS = [
    "#5f8fd3", "#f0a15a", "#7fbf73", "#8b72c6", "#df6667",
    "#5ebdb5", "#f2c94c", "#9a8176", "#a6b34e", "#ce96b8",
]
LOGIT_COLORS = [
    "#717b86", "#b9bec5", "#89919b", "#d9dde2", "#6f7883",
    "#eef0f2", "#9ca4ad", "#c8cdd3", "#0b5ea8", "#aeb4bd",
]


def _font_path(*names: str) -> str | None:
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return str(path)
    return None


FONT_REG = _font_path("arial.ttf", "segoeui.ttf")
FONT_BOLD = _font_path("arialbd.ttf", "segoeuib.ttf") or FONT_REG
FONT_ITALIC = _font_path("ariali.ttf", "segoeuii.ttf") or FONT_REG


def font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_ITALIC if italic else FONT_REG
    if path:
        return ImageFont.truetype(path, size * SCALE)
    return ImageFont.load_default()


def sc(v: float) -> int:
    return int(round(v * SCALE))


def box(x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
    return sc(x), sc(y), sc(x + w), sc(y + h)


class Figure:
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.im = Image.new("RGB", (w * SCALE, h * SCALE), BG)
        self.d = ImageDraw.Draw(self.im)

    def save(self, name: str, pad: int = 34, aliases: Sequence[str] = ()) -> None:
        OUT.mkdir(exist_ok=True)
        diff = ImageChops.difference(self.im, Image.new("RGB", self.im.size, BG))
        bbox = diff.getbbox()
        im = self.im.crop(bbox) if bbox else self.im
        canvas = Image.new("RGB", (im.width + pad * 2, im.height + pad * 2), BG)
        canvas.paste(im, (pad, pad))
        path = OUT / name
        canvas.save(path, optimize=True)
        print(f"saved {path}")
        for alias in aliases:
            alias_path = OUT / alias
            canvas.save(alias_path, optimize=True)
            print(f"saved {alias_path}")

    def text(
        self,
        xy: tuple[float, float],
        text: str,
        size: int = 16,
        fill: str = INK,
        bold: bool = False,
        italic: bool = False,
        anchor: str = "mm",
        align: str = "center",
        spacing: int = 4,
    ) -> None:
        self.d.multiline_text(
            (sc(xy[0]), sc(xy[1])),
            text,
            fill=fill,
            font=font(size, bold, italic),
            anchor=anchor,
            align=align,
            spacing=sc(spacing),
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        outline: str,
        width: int = 2,
        radius: int = 4,
    ) -> None:
        self.d.rounded_rectangle(
            box(x, y, w, h),
            radius=sc(radius),
            fill=fill,
            outline=outline,
            width=sc(width),
        )

    def dashed_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = GROUP_FILL,
        outline: str = GROUP_STROKE,
        width: int = 2,
        dash: int = 10,
        gap: int = 8,
        radius: int = 4,
    ) -> None:
        self.rect(x, y, w, h, fill, fill, width=1, radius=radius)
        pts = [
            ((x + radius, y), (x + w - radius, y)),
            ((x + w, y + radius), (x + w, y + h - radius)),
            ((x + w - radius, y + h), (x + radius, y + h)),
            ((x, y + h - radius), (x, y + radius)),
        ]
        for start, end in pts:
            self._dashed_line(start, end, outline, width, dash, gap)

    def block(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        kind: str = "train",
        size: int = 17,
        bold: bool = False,
    ) -> None:
        fill, stroke = {
            "train": (TRAIN_FILL, TRAIN_STROKE),
            "notr": (NOTR_FILL, NOTR_STROKE),
            "neut": (NEUT_FILL, NEUT_STROKE),
            "loss": (LOSS_FILL, LOSS_STROKE),
            "group": (GROUP_FILL, GROUP_STROKE),
        }[kind]
        self.rect(x, y, w, h, fill, stroke, width=2, radius=4)
        self.text((x + w / 2, y + h / 2), text, size=size, bold=bold)

    def panel(self, x: float, y: float, w: float, h: float, title: str = "") -> None:
        self.rect(x, y, w, h, "#fbfcfe", "#dce3eb", width=1, radius=8)
        if title:
            self.text((x + 18, y + 18), title, size=16, bold=True, anchor="lm")

    def step_badge(self, x: float, y: float, text: str, fill: str = "#eef2f7") -> None:
        self.d.ellipse(box(x, y, 26, 26), fill=fill, outline="#aeb9c6", width=sc(1))
        self.text((x + 13, y + 13), text, size=12, bold=True)

    def line(self, pts: Sequence[tuple[float, float]], fill: str = ARROW, width: int = 2) -> None:
        self.d.line([(sc(x), sc(y)) for x, y in pts], fill=fill, width=sc(width), joint="curve")

    def _dashed_line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        fill: str = ARROW,
        width: int = 2,
        dash: int = 10,
        gap: int = 8,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        pos = 0.0
        while pos < length:
            seg = min(dash, length - pos)
            a = (x1 + ux * pos, y1 + uy * pos)
            b = (x1 + ux * (pos + seg), y1 + uy * (pos + seg))
            self.line([a, b], fill, width)
            pos += dash + gap

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        fill: str = ARROW,
        width: int = 2,
        head: int = 10,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        self.line([start, end], fill, width)
        ang = math.atan2(y2 - y1, x2 - x1)
        a1 = ang + math.pi * 0.82
        a2 = ang - math.pi * 0.82
        p1 = (x2 + head * math.cos(a1), y2 + head * math.sin(a1))
        p2 = (x2 + head * math.cos(a2), y2 + head * math.sin(a2))
        self.d.polygon([(sc(x2), sc(y2)), (sc(p1[0]), sc(p1[1])), (sc(p2[0]), sc(p2[1]))], fill=fill)

    def poly_arrow(
        self,
        pts: Sequence[tuple[float, float]],
        fill: str = ARROW,
        width: int = 2,
        head: int = 10,
    ) -> None:
        if len(pts) < 2:
            return
        self.line(pts, fill, width)
        x1, y1 = pts[-2]
        x2, y2 = pts[-1]
        ang = math.atan2(y2 - y1, x2 - x1)
        a1 = ang + math.pi * 0.82
        a2 = ang - math.pi * 0.82
        p1 = (x2 + head * math.cos(a1), y2 + head * math.sin(a1))
        p2 = (x2 + head * math.cos(a2), y2 + head * math.sin(a2))
        self.d.polygon([(sc(x2), sc(y2)), (sc(p1[0]), sc(p1[1])), (sc(p2[0]), sc(p2[1]))], fill=fill)

    def matrix(self, x: float, y: float, cols: int, rows: int, cell: int = 14) -> None:
        for r in range(rows):
            for c in range(cols):
                col = CLASS_COLORS[(r * cols + c) % len(CLASS_COLORS)]
                self.d.rectangle(box(x + c * cell, y + r * cell, cell, cell), fill=col)
        self.d.rectangle(box(x, y, cols * cell, rows * cell), outline=GRID, width=sc(1))

    def vector(
        self,
        x: float,
        y: float,
        n: int,
        cell_w: int = 13,
        cell_h: int = 13,
        palette: Sequence[str] = CLASS_COLORS,
        hi: int | None = None,
    ) -> None:
        for i in range(n):
            col = palette[i % len(palette)]
            if hi is not None and i == hi:
                col = "#0b5ea8"
            self.d.rectangle(box(x, y + i * cell_h, cell_w, cell_h), fill=col)
        self.d.rectangle(box(x, y, cell_w, n * cell_h), outline="#9aa6b2", width=sc(1))

    def feature_stack(self, x: float, y: float, w: float, h: float, title: str, sub: str) -> None:
        for off in (12, 6, 0):
            self.rect(x + off, y - off, w, h, TRAIN_FILL, TRAIN_STROKE, width=2, radius=4)
        cx = x + w / 2 + 12
        self.text((cx, y + h / 2 - 10), title, size=16, bold=True)
        self.text((cx, y + h / 2 + 24), sub, size=13, fill=MUTED)

    def digit(self, x: float, y: float, w: float, h: float, img: Image.Image, frame: bool = True) -> None:
        if frame:
            self.rect(x, y, w, h, "#ffffff", "#a9b4c1", width=1, radius=2)
            pad = max(3, int(min(w, h) * 0.07))
        else:
            pad = 0
        thumb = img.resize((sc(w - 2 * pad), sc(h - 2 * pad)), Image.Resampling.BICUBIC)
        self.im.paste(thumb.convert("RGB"), (sc(x + pad), sc(y + pad)))
        if frame:
            self.d.rounded_rectangle(box(x, y, w, h), radius=sc(2), outline="#a9b4c1", width=sc(1))

    def label_box(self, x: float, y: float, w: float, h: float, text: str, size: int = 14) -> None:
        self.block(x, y, w, h, text, "neut", size=size)

    def legend(self, x: float, y: float, items: Sequence[tuple[str, str]]) -> None:
        xx = x
        for label, kind in items:
            fill, stroke = {
                "train": (TRAIN_FILL, TRAIN_STROKE),
                "notr": (NOTR_FILL, NOTR_STROKE),
                "neut": (NEUT_FILL, NEUT_STROKE),
                "loss": (LOSS_FILL, LOSS_STROKE),
            }[kind]
            self.rect(xx, y - 10, 28, 20, fill, stroke, width=2, radius=3)
            self.text((xx + 38, y), label, size=14, fill=INK, anchor="lm")
            xx += 156


def load_digits() -> dict[int, Image.Image]:
    from core.config import ANCHOR_INDICES
    from core.data import load_datasets, materialize

    _, train_set, _, _ = load_datasets()
    x_train, y_train = materialize(train_set)
    out: dict[int, Image.Image] = {}
    for idx in ANCHOR_INDICES:
        arr = x_train[idx, 0].numpy()
        img = Image.fromarray(np.uint8(255 - arr * 255), mode="L").convert("RGB")
        out[int(y_train[idx])] = img
    return out


def digit_row(f: Figure, x: float, y: float, imgs: dict[int, Image.Image], size: int = 24) -> None:
    for i in range(10):
        f.digit(x + i * size, y, size, size, imgs[i], frame=True)


def _trim_image(im: Image.Image) -> Image.Image:
    rgb = im.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, BG))
    bbox = diff.getbbox()
    return rgb.crop(bbox) if bbox else rgb


def _scale_to_width(im: Image.Image, width: int) -> Image.Image:
    if im.width == width:
        return im
    height = round(im.height * width / im.width)
    return im.resize((width, height), Image.Resampling.LANCZOS)


def _save_composite(name: str, panels: Sequence[tuple[str, str, int]], aliases: Sequence[str] = ()) -> None:
    OUT.mkdir(exist_ok=True)
    margin, gap = 52, 48
    loaded: list[tuple[str, Image.Image]] = []
    for title, filename, width in panels:
        im = _scale_to_width(_trim_image(Image.open(OUT / filename)), width)
        loaded.append((title, im))

    canvas_w = max(im.width for _, im in loaded) + margin * 2
    canvas_h = margin
    for _, im in loaded:
        canvas_h += 48 + im.height + gap
    canvas_h += margin - gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)
    y = margin
    for title, im in loaded:
        x = (canvas_w - im.width) // 2
        draw.text((x, y), title, fill=INK, font=font(20, bold=True), anchor="la")
        y += 48
        canvas.paste(im, (x, y))
        y += im.height + gap

    path = OUT / name
    canvas.save(path, optimize=True)
    print(f"saved {path}")
    for alias in aliases:
        alias_path = OUT / alias
        canvas.save(alias_path, optimize=True)
        print(f"saved {alias_path}")


def fig_pipeline(imgs: dict[int, Image.Image]) -> None:
    f = Figure(1320, 300)

    def emb_bar(
        x: float,
        y: float,
        label: str,
        colors: Sequence[str] = (),
        size: int = 16,
        italic: bool = True,
        label_y: float | None = None,
    ) -> None:
        palette = colors or ["#b090d0", "#80b0e0", "#f08090", "#b0d0f0", "#6090d0", "#c0b0e0"]
        cell = 18
        for i, color in enumerate(palette[:6]):
            f.d.rectangle(box(x, y + i * cell, cell, cell), fill=color)
        f.d.rectangle(box(x, y, cell, 6 * cell), outline="#dce3ec", width=sc(1))
        if label:
            f.text((x + cell / 2, label_y or y + 6 * cell + 24), label, size=size, fill=INK, italic=italic)

    y = 104
    main_label_y = 182
    x0, y0, s = 28, y - 38, 40
    for i in range(6):
        f.digit(x0 + (i % 3) * s, y0 + (i // 3) * s, 36, 36, imgs[i])
    f.text((x0 + 58, main_label_y), "unlabeled", size=15, fill=INK)

    f.arrow((158, y), (194, y))
    f.block(214, y - 27, 124, 54, "Encoder", "train", 17, True)

    f.arrow((338, y), (374, y))
    emb_bar(396, y - 54, "h", label_y=main_label_y)

    f.arrow((432, y), (468, y))
    f.block(488, y - 27, 102, 54, "LGC", "notr", 18, True)

    bottom_label_y = 262
    icon_y = 220
    train_col, anchor_col, non_col = 326, 542, 800

    f.rect(train_col - 14, icon_y - 10, 28, 20, TRAIN_FILL, TRAIN_STROKE, width=2, radius=3)
    f.text((train_col, bottom_label_y), "Trainable", size=14, fill=INK)

    anchor_size = 18
    digit_row(f, anchor_col - 5 * anchor_size, icon_y - 9, imgs, anchor_size)
    f.text((anchor_col, bottom_label_y), "10 anchors", size=13, fill=INK)
    f.arrow((anchor_col, icon_y - 22), (anchor_col, y + 27 + 9))

    f.rect(non_col - 14, icon_y - 10, 28, 20, NOTR_FILL, NOTR_STROKE, width=2, radius=3)
    f.text((non_col, bottom_label_y), "Non-trainable", size=14, fill=INK)

    f.arrow((590, y), (626, y))
    emb_bar(648, y - 54, "pseudo labels", size=14, italic=False, label_y=main_label_y)

    f.arrow((684, y), (720, y))
    f.block(740, y - 27, 102, 54, "CNN", "train", 17, True)

    f.arrow((842, y), (878, y))
    f.block(898, y - 25, 104, 50, "Softmax", "notr", 15, True)

    f.arrow((1002, y), (1038, y))
    emb_bar(1060, y - 54, "", LOGIT_COLORS)
    f.text((1100, y), "98.5%\ntest acc", size=17, bold=True, anchor="lm")

    f.save("fig_pipeline.png", pad=24, aliases=["1.png"])


def fig_ae(imgs: dict[int, Image.Image]) -> None:
    f = Figure(1060, 300)
    digit = imgs[4]
    recon = digit.filter(ImageFilter.GaussianBlur(radius=1.2))

    y = 146
    f.digit(44, y - 40, 80, 80, digit)
    f.text((84, y + 74), "input", size=13, fill=MUTED)
    f.arrow((144, y), (214, y))
    f.block(236, y - 30, 118, 60, "Encoder", "train", 17, True)
    f.arrow((374, y), (468, y))
    f.vector(500, y - 72, 9, 16, 15, CLASS_COLORS)
    f.text((518, y + 86), "h (128-d)", size=14, fill=TRAIN_STROKE, italic=True, anchor="lm")
    f.arrow((552, y), (650, y))
    f.block(672, y - 30, 118, 60, "Decoder", "train", 17, True)
    f.arrow((810, y), (882, y))
    f.digit(904, y - 40, 80, 80, recon)
    f.text((944, y + 74), "reconstruction", size=13, fill=MUTED)

    f.block(438, 28, 150, 40, "MSE loss", "loss", 16, True)
    f.poly_arrow([(944, 98), (944, 48), (596, 48)], fill=LOSS_STROKE)
    f.poly_arrow([(430, 48), (84, 48), (84, 98)], fill=LOSS_STROKE)
    f.legend(440, 282, [("Trainable", "train")])
    f.save("fig_ae.png")


def fig_simclr(imgs: dict[int, Image.Image]) -> None:
    f = Figure(1180, 390)
    base = imgs[4]
    view_i = base.rotate(11, resample=Image.Resampling.BILINEAR, fillcolor="white")
    view_j = Image.new("RGB", base.size, "white")
    view_j.paste(base, (2, -2))

    f.digit(40, 160, 70, 70, base)
    f.text((75, 256), "image x", size=13, fill=MUTED)
    f.arrow((130, 195), (190, 195))
    f.block(212, 170, 104, 50, "Augment", "neut", 16, True)
    f.text((264, 244), "no flip", size=12, fill=MUTED, italic=True)

    f.block(610, 18, 170, 36, "use h, not z", "neut", 14)

    rows = [(80, view_i, "i"), (240, view_j, "j")]
    for y, img, sub in rows:
        mid_y = y + 35
        f.poly_arrow([(316, 195), (360, 195), (360, mid_y), (408, mid_y)])
        f.digit(430, y, 70, 70, img)
        f.text((465, y + 92), f"view {sub}", size=13, italic=True)
        f.arrow((518, mid_y), (574, mid_y))
        f.block(596, y + 10, 108, 50, "Encoder", "train", 15, True)
        f.arrow((724, mid_y), (780, mid_y))
        f.vector(802, y + 3, 6, 13, 12, CLASS_COLORS)
        f.text((810, y + 92), f"h{sub}", size=14, fill=TRAIN_STROKE, italic=True)
        f.arrow((834, mid_y), (888, mid_y))
        f.block(910, y + 10, 96, 50, "Projection", "train", 14, True)
        f.arrow((1026, mid_y), (1068, mid_y))
        f.vector(1088, y + 10, 4, 13, 12, CLASS_COLORS)
        f.text((1096, y + 92), f"z{sub}", size=14, italic=True)

    f.block(1030, 166, 98, 58, "NT-Xent", "loss", 16, True)
    f.poly_arrow([(1100, 125), (1138, 125), (1138, 178), (1128, 178)], fill=LOSS_STROKE)
    f.poly_arrow([(1100, 285), (1138, 285), (1138, 212), (1128, 212)], fill=LOSS_STROKE)
    f.text((995, 358), "pull + / push -", size=12, fill=LOSS_STROKE, italic=True)
    f.legend(44, 360, [("Trainable", "train"), ("Loss", "loss")])
    f.save("fig_simclr.png")


def _blobs(seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([
        [-5.2, 2.8], [-2.8, 5.2], [0.6, 3.8], [4.4, 4.6], [6.0, 1.1],
        [3.0, -1.8], [-0.8, -3.3], [-4.7, -2.0], [-1.2, 0.6], [2.5, 1.3],
    ])
    pts, labels = [], []
    for c, ctr in enumerate(centers):
        theta = 0.35 * c
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        scale = np.diag([0.9 + 0.08 * (c % 3), 0.38 + 0.06 * (c % 4)])
        cloud = rng.normal(0, 1, size=(72, 2)) @ scale @ rot.T + ctr
        pts.append(cloud)
        labels.append(np.full(len(cloud), c))
    p = np.vstack(pts)
    y = np.concatenate(labels)
    anchors = [np.where(y == c)[0][np.argmin(np.linalg.norm(p[y == c] - centers[c], axis=1))] for c in range(10)]
    return p, y, np.array(anchors)


def _map_point(p: np.ndarray, ox: float, oy: float, sx: float, sy: float) -> tuple[float, float]:
    return ox + p[0] * sx, oy + (10.2 - p[1]) * sy


def fig_propagation(_: dict[int, Image.Image]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import ScalarMappable

    p, labels, anchors = _blobs()
    anchor_dist = np.stack([np.linalg.norm(p - p[a], axis=1) for a in anchors], axis=1).min(axis=1)

    masks = [
        anchor_dist <= np.quantile(anchor_dist, 0.86),
        anchor_dist <= np.quantile(anchor_dist, 0.32),
    ]
    for mask in masks:
        mask[anchors] = True

    cmap = plt.get_cmap("tab10")
    norm = BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), squeeze=False)
    axes = axes[0]
    titles = ["Local-Global Consistency", "Nearest-Prototype"]
    captions = ["coverage ≈ 86%", "coverage ≈ 32%"]

    for ax, title, caption, mask in zip(axes, titles, captions, masks):
        ax.scatter(p[~mask, 0], p[~mask, 1], c="#9aa0a6", s=12, alpha=0.35, linewidths=0)
        ax.scatter(p[mask, 0], p[mask, 1], c=labels[mask], cmap=cmap, norm=norm, s=13, alpha=0.72, linewidths=0)
        ax.scatter(
            p[anchors, 0],
            p[anchors, 1],
            c=labels[anchors],
            cmap=cmap,
            norm=norm,
            s=260,
            marker="*",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.03,
            0.04,
            caption,
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="0.65", boxstyle="round,pad=0.25", alpha=0.9),
        )

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=list(axes), ticks=range(10), fraction=0.03, pad=0.015)
    cbar.set_label("assigned pseudo digit")
    OUT.mkdir(exist_ok=True)
    for name in ("fig_propagation.png", "3.png"):
        path = OUT / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)


def fig_selftrain(_: dict[int, Image.Image]) -> None:
    f = Figure(1180, 430)
    f.block(86, 42, 220, 50, "Propagation base", "notr", 17, True)

    f.block(86, 188, 132, 56, "Train\nclassifier", "train", 15, True)
    f.block(432, 188, 132, 56, "Re-score\nunlabeled", "train", 15, True)
    f.block(748, 188, 136, 56, "Confidence\nfilter", "neut", 15, True)
    f.block(432, 322, 132, 54, "Union", "neut", 17, True)

    f.poly_arrow([(152, 92), (152, 188)])
    f.arrow((218, 216), (432, 216))
    f.text((322, 196), "round r", size=14, fill=MUTED, italic=True)
    f.arrow((564, 216), (748, 216))

    for i in range(10):
        color = f"#{195 - i * 13:02x}{222 - i * 8:02x}{242 - i * 2:02x}"
        f.d.rectangle(box(762 + i * 12, 276, 12, 17), fill=color)
    f.text((824, 306), "confidence scores", size=12, fill=MUTED)

    f.poly_arrow([(816, 244), (816, 349), (564, 349)])
    f.poly_arrow([(432, 349), (152, 349), (152, 244)])
    f.text((76, 300), "next round", size=14, fill=MUTED, italic=True)
    f.block(442, 394, 112, 26, "fixed base", "neut", 13)
    f.legend(650, 404, [("Trainable", "train"), ("Non-trainable", "notr"), ("Set update", "neut")])
    f.save("fig_selftrain.png")


def fig_cnn(imgs: dict[int, Image.Image]) -> None:
    f = Figure(1550, 320)
    f.digit(42, 112, 78, 78, imgs[4])
    f.text((81, 218), "input\n1 x 32 x 32", size=13, fill=MUTED)
    f.arrow((140, 151), (218, 151))

    stages = [
        (240, 98, 126, 106, "Conv 1", "32 ch\n16 x 16"),
        (446, 106, 118, 90, "Conv 2", "64 ch\n8 x 8"),
        (644, 118, 104, 66, "Conv 3", "128 ch\n4 x 4"),
    ]
    for x, y, w, h, title, sub in stages:
        f.feature_stack(x, y, w, h, title, sub)

    f.arrow((380, 151), (438, 151))
    f.arrow((578, 151), (638, 151))
    f.arrow((760, 151), (842, 151))
    f.vector(870, 70, 11, 14, 15, CLASS_COLORS)
    f.text((900, 262), "flatten\n2048", size=13, fill=MUTED, anchor="lm", align="left")
    f.arrow((916, 151), (994, 151))
    f.block(1018, 118, 92, 66, "FC 256", "train", 16)
    f.arrow((1130, 151), (1194, 151))
    f.block(1218, 126, 72, 50, "FC 10", "train", 16, True)
    f.arrow((1310, 151), (1352, 151))
    f.block(1374, 126, 96, 50, "Softmax", "notr", 15, True)
    f.arrow((1470, 151), (1512, 151))
    f.vector(1526, 92, 9, 10, 10, LOGIT_COLORS, hi=8)
    f.legend(610, 298, [("Trainable", "train")])
    f.save("fig_cnn.png")


def fig_representation() -> None:
    imgs = load_digits()
    f = Figure(1760, 660)

    def small_label(x: float, y: float, text: str) -> None:
        f.text((x, y), text, size=16, fill=INK)

    def block(x: float, y: float, text: str, kind: str = "train", w: float = 160, h: float = 56) -> None:
        f.block(x, y, w, h, text, kind, 18, False)

    def vec(x: float, y: float, label: str, cell: int = 18, label_pos: str = "below") -> None:
        colors = ["#b090d0", "#80b0e0", "#f08090", "#b0d0f0", "#6090d0", "#c0b0e0"]
        for i, color in enumerate(colors):
            f.d.rectangle(box(x, y + i * cell, cell, cell), fill=color)
        f.d.rectangle(box(x, y, cell, len(colors) * cell), outline="#dce3ec", width=sc(1))
        label_y = y - 17 if label_pos == "above" else y + len(colors) * cell + 26
        f.text((x + cell / 2, label_y), label, size=16, fill=INK, italic=True)

    def loss_block(x: float, y: float, text: str, w: float = 170, h: float = 58) -> None:
        f.block(x, y, w, h, text, "loss", 18, False)

    # Panel a: autoencoder representation learning.
    y = 178
    f.digit(82, y - 50, 100, 100, imgs[4])
    small_label(132, y + 84, "input image")
    f.arrow((204, y), (308, y))
    block(328, y - 28, "Encoder")
    f.arrow((488, y), (598, y))
    vec(632, y - 54, "h")
    f.arrow((674, y), (798, y))
    block(818, y - 28, "Decoder")
    f.arrow((978, y), (1090, y))
    f.digit(1110, y - 50, 100, 100, imgs[4].filter(ImageFilter.GaussianBlur(radius=1.2)))
    small_label(1160, y + 84, "reconstruction")
    f.arrow((1232, y), (1418, y))
    loss_block(1440, y - 29, "MSE loss", 180, 58)

    f.poly_arrow([(132, y - 54), (132, 88), (1530, 88), (1530, y - 29)], fill=LOSS_STROKE)

    # Panel b: contrastive representation learning.
    base = imgs[4]
    view_i = base.rotate(10, resample=Image.Resampling.BILINEAR, fillcolor="white")
    view_j = Image.new("RGB", base.size, "white")
    view_j.paste(base, (2, -2))

    row_i, row_j = 382, 512
    fork_y = (row_i + row_j) / 2
    f.digit(82, fork_y - 46, 92, 92, base)
    small_label(128, fork_y + 78, "image x")
    f.arrow((190, fork_y), (288, fork_y))
    block(308, fork_y - 28, "Augment", "notr", 154, 56)

    f.poly_arrow([(462, fork_y), (512, fork_y), (512, row_i), (564, row_i)])
    f.poly_arrow([(462, fork_y), (512, fork_y), (512, row_j), (564, row_j)])

    for row, img, sub in [(row_i, view_i, "i"), (row_j, view_j, "j")]:
        label_pos = "above" if sub == "i" else "below"
        f.digit(584, row - 43, 86, 86, img)
        f.text((627, row + 70), f"view {sub}", size=15, fill=INK, italic=True)
        f.arrow((690, row), (760, row))
        block(780, row - 28, "Encoder")
        f.arrow((940, row), (1014, row))
        vec(1046, row - 54, f"h_{sub}", label_pos=label_pos)
        f.arrow((1088, row), (1168, row))
        block(1188, row - 28, "Projection\nhead", w=164, h=56)
        f.arrow((1352, row), (1414, row))
        vec(1444, row - 54, f"z_{sub}", label_pos=label_pos)

    loss_block(1558, fork_y - 34, "NT-Xent", 136, 68)
    f.poly_arrow([(1474, row_i), (1534, row_i), (1534, fork_y - 12), (1558, fork_y - 12)], fill=LOSS_STROKE)
    f.poly_arrow([(1474, row_j), (1534, row_j), (1534, fork_y + 12), (1558, fork_y + 12)], fill=LOSS_STROKE)
    f.legend(600, 630, [("Trainable", "train"), ("Non-trainable", "notr"), ("Loss", "loss")])

    # Draw panel labels last so they sit on top without forcing layout gaps.
    f.text((24, 38), "a", size=30, bold=True, anchor="lm")
    f.text((86, 38), "Autoencoder", size=24, bold=True, anchor="lm")
    f.text((24, 300), "b", size=30, bold=True, anchor="lm")
    f.text((86, 300), "SimCLR", size=24, bold=True, anchor="lm")
    f.save("fig_representation.png", aliases=["2.png"])


def fig_classifier_selftrain() -> None:
    imgs = load_digits()
    f = Figure(1450, 440)

    def conv_block(x: float, y: float, w: float, h: float, title: str, sub: str) -> None:
        for off in (6, 3, 0):
            f.rect(x + off, y - off, w, h, TRAIN_FILL, TRAIN_STROKE, width=2, radius=4)
        f.text((x + w / 2 + 6, y + h / 2 - 8), title, size=13, bold=True)
        f.text((x + w / 2 + 6, y + h / 2 + 16), sub, size=10, fill=INK)

    def confidence_bar(x: float, y: float) -> None:
        for i in range(10):
            color = f"#{198 - i * 12:02x}{224 - i * 7:02x}{243 - i * 2:02x}"
            f.d.rectangle(box(x + i * 12, y, 12, 16), fill=color)

    mid = 108
    f.digit(72, mid - 34, 68, 68, imgs[4])
    f.text((106, mid + 60), "input", size=13, fill=INK)
    f.arrow((158, mid), (230, mid))
    conv_block(250, mid - 28, 100, 56, "Conv 1", "32 ch")
    f.arrow((366, mid), (424, mid))
    conv_block(444, mid - 28, 100, 56, "Conv 2", "64 ch")
    f.arrow((560, mid), (618, mid))
    conv_block(638, mid - 28, 100, 56, "Conv 3", "128 ch")
    f.arrow((754, mid), (784, mid))
    f.vector(806, mid - 54, 6, 18, 18, CLASS_COLORS)
    f.arrow((842, mid), (900, mid))
    f.block(920, mid - 22, 92, 44, "Flatten", "neut", 13, True)
    f.arrow((1012, mid), (1058, mid))
    f.block(1078, mid - 23, 82, 46, "FC 256", "train", 13, True)
    f.arrow((1160, mid), (1200, mid))
    f.block(1220, mid - 19, 66, 38, "FC 10", "train", 13, True)
    f.arrow((1286, mid), (1322, mid))
    f.block(1342, mid - 21, 92, 42, "Softmax", "notr", 13, True)

    loop_y = 292
    f.block(82, loop_y - 24, 150, 48, "Propagation\nbase", "notr", 13, True)
    f.block(336, loop_y - 28, 132, 56, "Train\nclassifier", "train", 13, True)
    f.block(622, loop_y - 28, 150, 56, "Re-score\nunlabeled", "train", 13, True)
    f.block(934, loop_y - 28, 150, 56, "Confidence\nfilter", "neut", 13, True)
    f.block(1226, loop_y - 24, 132, 48, "Update\nset", "neut", 13, True)

    f.arrow((232, loop_y), (336, loop_y))
    f.arrow((468, loop_y), (622, loop_y))
    f.arrow((772, loop_y), (934, loop_y))
    f.arrow((1084, loop_y), (1226, loop_y))
    f.poly_arrow([(1358, loop_y), (1390, loop_y), (1390, loop_y + 64), (402, loop_y + 64), (402, loop_y + 28)])

    f.legend(500, 398, [("Trainable", "train"), ("Non-trainable", "notr"), ("Set update", "neut")])

    f.text((24, 34), "a", size=30, bold=True, anchor="lm")
    f.text((86, 34), "CNN classifier", size=24, bold=True, anchor="lm")
    f.text((24, 222), "b", size=30, bold=True, anchor="lm")
    f.text((86, 222), "Self-training loop", size=24, bold=True, anchor="lm")

    f.save("fig_classifier_selftrain.png", aliases=["4.png"])


def main() -> None:
    imgs = load_digits()
    fig_pipeline(imgs)
    fig_ae(imgs)
    fig_simclr(imgs)
    fig_propagation(imgs)
    fig_selftrain(imgs)
    fig_cnn(imgs)
    fig_representation()
    fig_classifier_selftrain()
    print("method figures generated in doc/")


if __name__ == "__main__":
    main()
