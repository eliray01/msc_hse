"""
Build small GIFs from rollout PNG sequences (e.g. plots_delta / plots_full).

Uses palette quantization and optional resizing so outputs stay small on disk.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from PIL import Image

STEP_PATTERN = re.compile(r"^step_(\d+)", re.IGNORECASE)


def _step_sort_key(path: Path) -> tuple[int, str]:
    m = STEP_PATTERN.match(path.stem)
    if m:
        return (int(m.group(1)), path.name)
    return (10**9, path.name)


def list_step_pngs(folder: Path) -> list[Path]:
    """PNG paths in `folder` matching step_*.png, ordered by step index."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    paths = sorted(folder.glob("step_*.png"), key=_step_sort_key)
    return paths


def iter_leaf_sequence_dirs(roots: Iterable[Path | str]) -> Iterator[Path]:
    """
    Yield directories under each root that directly contain at least one step_*.png.
    """
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(root, topdown=True):
            p = Path(dirpath)
            if any(fn.lower().endswith(".png") and fn.lower().startswith("step_") for fn in filenames):
                yield p


def _load_rgb_resized(path: Path, max_edge: int | None) -> Image.Image:
    im = Image.open(path)
    im = im.convert("RGB")
    if max_edge is not None and max_edge > 0:
        w, h = im.size
        m = max(w, h)
        if m > max_edge:
            scale = max_edge / m
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    return im


def pngs_to_gif(
    png_paths: Sequence[Path | str],
    out_path: Path | str,
    *,
    fps: float = 8.0,
    max_edge: int | None = 480,
    palette_colors: int = 64,
    frame_stride: int = 1,
    optimize: bool = True,
    overwrite: bool = True,
) -> Path | None:
    """
    Encode ordered PNGs into one GIF.

    Parameters
    ----------
    max_edge
        Longer image side in pixels after resize; None to keep original size.
    palette_colors
        Number of colors in the global palette (smaller => smaller files, more banding).
    frame_stride
        Use every n-th frame (e.g. 2 halves frame count and file size).
    """
    paths = [Path(p) for p in png_paths]
    if frame_stride > 1:
        paths = paths[::frame_stride]
    if len(paths) < 2:
        return None

    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))

    first_rgb = _load_rgb_resized(paths[0], max_edge)
    p0 = first_rgb.quantize(
        colors=min(palette_colors, 256),
        method=Image.Quantize.MEDIANCUT,
    )
    del first_rgb

    rest: list[Image.Image] = []
    for p in paths[1:]:
        rgb = _load_rgb_resized(p, max_edge)
        rest.append(rgb.quantize(palette=p0))
        del rgb

    p0.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=optimize,
        disposal=2,
    )
    for im in rest:
        im.close()
    p0.close()
    return out_path


def make_gifs_from_plot_roots(
    roots: Sequence[Path | str],
    out_root: Path | str,
    *,
    fps: float = 8.0,
    max_edge: int | None = 480,
    palette_colors: int = 64,
    frame_stride: int = 1,
    gif_name: str = "sequence.gif",
    overwrite: bool = True,
) -> list[Path]:
    """
    For each directory under ``roots`` that contains step_*.png, write one GIF under
    ``out_root``, mirroring the relative path from that root.

    Returns paths of written GIFs (skipped / too-few-frames dirs omitted).
    """
    written: list[Path] = []
    out_root = Path(out_root)

    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        for seq_dir in iter_leaf_sequence_dirs([root]):
            pngs = list_step_pngs(seq_dir)
            if len(pngs) < 2:
                continue
            rel = seq_dir.relative_to(root)
            dest = out_root / root.name / rel / gif_name
            out = pngs_to_gif(
                pngs,
                dest,
                fps=fps,
                max_edge=max_edge,
                palette_colors=palette_colors,
                frame_stride=frame_stride,
                overwrite=overwrite,
            )
            if out is not None:
                written.append(out)
    return written


def _default_roots(repo: Path) -> list[Path]:
    return [repo / "plots_delta", repo / "plots_full"]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="GIFs from plots_delta / plots_full rollout PNGs.")
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=None,
        help="Plot roots (default: plots_delta and plots_full next to repo root)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo / "plot_gifs_out",
        help="Output directory (mirrors root name + relative path)",
    )
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument(
        "--max-edge",
        type=int,
        default=480,
        help="Max longer side in pixels; 0 = no resize",
    )
    parser.add_argument("--colors", type=int, default=64, help="Palette size (smaller => smaller GIF)")
    parser.add_argument("--stride", type=int, default=1, help="Use every n-th frame")
    args = parser.parse_args()

    roots = args.roots if args.roots else _default_roots(repo)
    max_edge = None if args.max_edge <= 0 else args.max_edge

    outs = make_gifs_from_plot_roots(
        roots,
        args.out,
        fps=args.fps,
        max_edge=max_edge,
        palette_colors=args.colors,
        frame_stride=args.stride,
    )
    print(f"Wrote {len(outs)} GIFs under {args.out.resolve()}")


if __name__ == "__main__":
    main()
