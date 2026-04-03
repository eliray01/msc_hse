"""
Autoregressive rollout for turbulent_radiative_layer_2D using CNextU-Net / UNetConvNext.

- **HuggingFace** ``polymathic-ai/UNetConvNext-turbulent_radiative_layer_2D``: always *full*
  prediction mode (normalised full frame → normalised full frame).
- **Your checkpoints** (``CNextUNet`` from ``train_cnextunet_full_frame.py``):
  use ``prediction_mode="full"`` (same convention).

Uses WellDataset (test split) for Z-score statistics. Trajectories come from the test HDF5
(``traj_id``, ``t_start``).

Notebook (pretrained HuggingFace):
    from autoregressive_cnextunet import predict_autoregressive_cnextunet, save_rollout_plots

    out = predict_autoregressive_cnextunet(base_path="./data", t_roll=10)
    save_rollout_plots(out, field="density", out_dir="./plots")

Notebook (local checkpoint):
    out = predict_autoregressive_cnextunet(
        model_pt_path="models_trained/unet_full_default/best_cnextunet_by_valid_rollout_vrmse.pt",
        t_roll=10,
    )

Notebook (auto-pick checkpoint in directory):
    from autoregressive_cnextunet import LOCAL_TRAINED_DIR, predict_autoregressive_cnextunet

    out = predict_autoregressive_cnextunet(
        trained_model_dir=str(LOCAL_TRAINED_DIR),
        t_roll=10,
    )

CLI:
    python autoregressive_cnextunet.py
    python autoregressive_cnextunet.py --model-pt models_trained/unet_full_default/best_cnextunet_by_valid_rollout_vrmse.pt
    python autoregressive_cnextunet.py --trained-model-dir models_trained/unet_full_default
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any, Literal, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from training_scripts.train_cnextunet_full_frame import CNextUNet
from the_well.benchmark.models import UNetConvNext as WellUNetConvNext
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization

FIELD_NAMES = ["density", "pressure", "velocity_x", "velocity_y"]
N_STEPS_INPUT = 4
DATASET_NAME = "turbulent_radiative_layer_2D"
PRETRAINED_REPO = "polymathic-ai/UNetConvNext-turbulent_radiative_layer_2D"

_REPO_ROOT = Path(__file__).resolve().parent
LOCAL_TRAINED_DIR = _REPO_ROOT / "models_trained" / "unet_full_default"

PredictionMode = Literal["delta", "full"]
CheckpointPrefer = Literal["rollout", "1step"]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _as_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return pick_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def load_trajectory_from_hdf5(path: str, traj_id: int) -> torch.Tensor:
    """Returns traj (T, H, W, F) with channels [density, pressure, vx, vy]."""
    with h5py.File(path, "r") as f:
        density = f["t0_fields"]["density"][:]
        pressure = f["t0_fields"]["pressure"][:]
        velocity = f["t1_fields"]["velocity"][:]

    dens = density[traj_id]
    pres = pressure[traj_id]
    vx = velocity[traj_id][..., 0]
    vy = velocity[traj_id][..., 1]

    traj = np.stack([dens, pres, vx, vy], axis=-1)
    return torch.tensor(traj, dtype=torch.float32)


def default_test_hdf5(base_path: str) -> str | None:
    pattern = os.path.join(
        base_path, "datasets", DATASET_NAME, "data", "test", "*.hdf5",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[0]


def resolve_test_hdf5_path(base_path: str, hdf5_path: str | None) -> str:
    p = (hdf5_path or "").strip()
    if not p:
        p = default_test_hdf5(base_path) or ""
    if not p or not os.path.isfile(p):
        raise FileNotFoundError(
            "Pass hdf5_path= to a test HDF5 file, or place *.hdf5 under "
            f"{base_path}/datasets/{DATASET_NAME}/data/test/"
        )
    return p


def load_test_normalization(
    base_path: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Z-score stats from WellDataset test split. Returns (mean_full, std_full, mean_delta, std_delta, n_fields)."""
    dset = WellDataset(
        well_base_path=f"{base_path}/datasets",
        well_dataset_name=DATASET_NAME,
        well_split_name="test",
        n_steps_input=N_STEPS_INPUT,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )
    f = dset.metadata.n_fields
    norm = dset.norm
    mean_full = norm.flattened_means["variable"].to(device)
    std_full = norm.flattened_stds["variable"].to(device)
    mean_d = norm.flattened_means_delta["variable"].to(device)
    std_d = norm.flattened_stds_delta["variable"].to(device)
    return mean_full, std_full, mean_d, std_d, f


def load_cnextunet_from_checkpoint(
    checkpoint_path: str,
    n_fields: int,
    device: torch.device,
    *,
    stages: int = 4,
    blocks_per_stage: int = 2,
    blocks_at_neck: int = 1,
    init_features: int = 42,
    kernel_size: int = 7,
) -> torch.nn.Module:
    """Load ``CNextUNet`` weights saved via ``torch.save(state_dict, ...)``."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device).eval()

    state_dict = None
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt

    if state_dict is None:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model = CNextUNet(
        dim_in=N_STEPS_INPUT * n_fields,
        dim_out=1 * n_fields,
        n_spatial_dims=2,
        stages=stages,
        blocks_per_stage=blocks_per_stage,
        blocks_at_neck=blocks_at_neck,
        init_features=init_features,
        kernel_size=kernel_size,
        gradient_checkpointing=False,
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def load_pretrained_well_cnextunet(
    device: torch.device,
    repo_id: str = PRETRAINED_REPO,
) -> torch.nn.Module:
    return WellUNetConvNext.from_pretrained(repo_id).to(device).eval()


def resolve_checkpoint_in_dir(
    directory: str | Path,
    prefer: CheckpointPrefer = "rollout",
) -> str:
    """Pick a default ``.pt`` inside a CNextU-Net training output folder."""
    d = Path(directory).expanduser().resolve()
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    rollout_first = [
        d / "best_cnextunet_by_valid_rollout_vrmse.pt",
        d / "best_cnextunet_by_valid_1step_vrmse.pt",
    ]
    step_first = [rollout_first[1], rollout_first[0]]

    order = rollout_first if prefer == "rollout" else step_first
    for p in order:
        if p.is_file():
            return str(p)

    raise FileNotFoundError(
        f"No matching checkpoint in {d} for prefer={prefer!r}. "
        f"Expected one of: {[str(x.name) for x in rollout_first]}"
    )


def _coalesce_model_pt_path(
    model_pt_path: str | None,
    checkpoint_path: str | None,
) -> str | None:
    a = (model_pt_path or "").strip() or None
    b = (checkpoint_path or "").strip() or None
    if a and b and os.path.abspath(a) != os.path.abspath(b):
        raise ValueError(
            f"model_pt_path and checkpoint_path disagree: {a!r} vs {b!r}. Pass only one."
        )
    return a or b


def resolve_checkpoint_path(
    model_pt_path: str | None,
    trained_model_dir: str | Path | None,
    prefer: CheckpointPrefer,
    *,
    checkpoint_path: str | None = None,
) -> str | None:
    """Returns a ``.pt`` path for a local checkpoint, or ``None`` to use HuggingFace pretrained."""
    cp = _coalesce_model_pt_path(model_pt_path, checkpoint_path)
    if cp:
        if not os.path.isfile(cp):
            raise FileNotFoundError(cp)
        return cp
    td = trained_model_dir
    if td is not None and str(td).strip():
        return resolve_checkpoint_in_dir(td, prefer=prefer)
    return None


def normalized_context_from_trajectory(
    traj: torch.Tensor,
    t_start: int,
    mean_full: torch.Tensor,
    std_full: torch.Tensor,
    n_steps_input: int = N_STEPS_INPUT,
) -> torch.Tensor:
    """Build x_init_norm (1, Ti, H, W, F) from raw traj on the same device as stats."""
    sl = traj[t_start : t_start + n_steps_input]
    return ((sl - mean_full.view(1, 1, 1, -1)) / std_full.view(1, 1, 1, -1)).unsqueeze(0)


def autoregressive_rollout_full_frame(
    model: torch.nn.Module,
    x_init_norm: torch.Tensor,
    t_roll: int,
    mean_full: torch.Tensor,
    std_full: torch.Tensor,
) -> torch.Tensor:
    """
    Model output: **normalised full** next frame.

    x_init_norm: (1, Ti=4, H, W, F) normalised full states.
    Returns predictions in **raw** physical units: (t_roll, H, W, F).
    """
    x_roll = x_init_norm.clone()
    preds: list[torch.Tensor] = []
    f = mean_full.shape[0]

    with torch.no_grad():
        for _ in range(t_roll):
            x_in = rearrange(x_roll, "B Ti H W F -> B (Ti F) H W")
            pred = model(x_in)
            pred = rearrange(pred, "B (To F) H W -> B To H W F", To=1, F=f)
            pred_full_norm = pred[:, 0]

            pred_full_raw = pred_full_norm * std_full + mean_full
            preds.append(pred_full_raw.squeeze(0))

            x_roll = torch.cat([x_roll[:, 1:], pred_full_norm.unsqueeze(1)], dim=1)

    return torch.stack(preds, dim=0)


def autoregressive_rollout_delta(
    model: torch.nn.Module,
    x_init_norm: torch.Tensor,
    t_roll: int,
    mean_full: torch.Tensor,
    std_full: torch.Tensor,
) -> torch.Tensor:
    """
    Model output: delta in normalised space (last_frame + delta = next_frame).

    x_init_norm: (1, Ti=4, H, W, F) normalised full states.
    Returns predictions in **raw** physical units: (t_roll, H, W, F).
    """
    x_roll = x_init_norm.clone()
    preds: list[torch.Tensor] = []
    f = mean_full.shape[0]

    with torch.no_grad():
        for _ in range(t_roll):
            x_in = rearrange(x_roll, "B Ti H W F -> B (Ti F) H W")
            pred = model(x_in)
            delta = rearrange(pred, "B (To F) H W -> B To H W F", To=1, F=f)[:, 0]

            pred_full_norm = x_roll[:, -1] + delta
            pred_full_raw = pred_full_norm * std_full + mean_full
            preds.append(pred_full_raw.squeeze(0))

            x_roll = torch.cat([x_roll[:, 1:], pred_full_norm.unsqueeze(1)], dim=1)

    return torch.stack(preds, dim=0)


def predict_autoregressive_cnextunet(
    *,
    base_path: str = "./data",
    hdf5_path: str | None = None,
    traj_id: int = 0,
    t_start: int = 0,
    t_roll: int = 10,
    device: torch.device | str | None = None,
    model: torch.nn.Module | None = None,
    repo_id: str = PRETRAINED_REPO,
    prediction_mode: PredictionMode = "full",
    model_pt_path: str | None = None,
    checkpoint_path: str | None = None,
    trained_model_dir: str | Path | None = None,
    checkpoint_prefer: CheckpointPrefer = "rollout",
    stages: int = 4,
    blocks_per_stage: int = 2,
    blocks_at_neck: int = 1,
    init_features: int = 42,
    kernel_size: int = 7,
) -> dict[str, Any]:
    """
    Load test normalisation, trajectory from HDF5, run autoregressive rollout with CNextU-Net.

    **Model selection** (first match wins):

    1. If ``model`` is passed, it is used (you must set ``prediction_mode`` correctly).
    2. Else if ``model_pt_path`` / ``checkpoint_path`` / ``trained_model_dir`` is set,
       load local ``CNextUNet`` from disk (architecture params must match training).
    3. Else load HuggingFace ``UNetConvNext`` (``prediction_mode`` must be ``"delta"``).

    Returns a dict with:
        pred_raw, traj, t0, t_indices, model, device, normalisation tensors, n_fields,
        hdf5_path, traj_id, t_start, t_roll,
        prediction_mode, model_backend, model_pt_path, plot_tag,
        architecture params (stages, blocks_per_stage, …).
    """
    dev = _as_device(device)
    path = resolve_test_hdf5_path(base_path, hdf5_path)
    mean_full, std_full, mean_d, std_d, n_fields = load_test_normalization(base_path, dev)

    ckpt_resolved = resolve_checkpoint_path(
        model_pt_path,
        trained_model_dir,
        checkpoint_prefer,
        checkpoint_path=checkpoint_path,
    )

    if model is not None:
        m = model
        if ckpt_resolved is not None:
            raise ValueError(
                "Pass either model= or model_pt_path=/checkpoint_path=/trained_model_dir=, not both."
            )
        mode = prediction_mode
        backend = "user_supplied"
        ckpt_used: str | None = None
    elif ckpt_resolved is not None:
        m = load_cnextunet_from_checkpoint(
            ckpt_resolved,
            n_fields,
            dev,
            stages=stages,
            blocks_per_stage=blocks_per_stage,
            blocks_at_neck=blocks_at_neck,
            init_features=init_features,
            kernel_size=kernel_size,
        )
        mode = prediction_mode
        backend = "cnextunet_checkpoint"
        ckpt_used = ckpt_resolved
    else:
        if prediction_mode != "full":
            raise ValueError(
                'HuggingFace pretrained UNetConvNext only supports prediction_mode="full". '
                'Pass prediction_mode="full" or omit it (default).'
            )
        m = load_pretrained_well_cnextunet(dev, repo_id=repo_id)
        mode = "full"
        backend = "pretrained_well"
        ckpt_used = None

    traj = load_trajectory_from_hdf5(path, traj_id).to(dev)
    if traj.shape[-1] != n_fields:
        raise ValueError(f"Trajectory channels {traj.shape[-1]} != dataset n_fields {n_fields}")

    t_end = t_start + N_STEPS_INPUT + t_roll
    if t_start < 0 or t_end > traj.shape[0]:
        raise ValueError(
            f"Need 0 <= t_start and t_start + {N_STEPS_INPUT} + t_roll <= T; "
            f"got t_start={t_start}, t_roll={t_roll}, T={traj.shape[0]}"
        )

    x_init_norm = normalized_context_from_trajectory(traj, t_start, mean_full, std_full)
    if mode == "full":
        pred_raw = autoregressive_rollout_full_frame(
            m, x_init_norm, t_roll, mean_full, std_full
        )
    else:
        pred_raw = autoregressive_rollout_delta(
            m, x_init_norm, t_roll, mean_full, std_full
        )

    t0 = t_start + N_STEPS_INPUT
    if backend == "pretrained_well":
        plot_tag = "pretrained_full"
    elif backend == "cnextunet_checkpoint":
        stem = Path(ckpt_used).stem if ckpt_used else "checkpoint"
        plot_tag = f"cnextunet_{mode}_{stem}"
    else:
        plot_tag = f"cnextunet_custom_{mode}"

    return {
        "pred_raw": pred_raw,
        "traj": traj,
        "t0": t0,
        "t_indices": [t0 + s for s in range(t_roll)],
        "model": m,
        "device": dev,
        "mean_full": mean_full,
        "std_full": std_full,
        "mean_delta": mean_d,
        "std_delta": std_d,
        "n_fields": n_fields,
        "hdf5_path": path,
        "traj_id": traj_id,
        "t_start": t_start,
        "t_roll": t_roll,
        "prediction_mode": mode,
        "model_backend": backend,
        "model_pt_path": ckpt_used,
        "checkpoint_path": ckpt_used,
        "plot_tag": plot_tag,
        "stages": stages,
        "blocks_per_stage": blocks_per_stage,
        "blocks_at_neck": blocks_at_neck,
        "init_features": init_features,
        "kernel_size": kernel_size,
    }


def save_rollout_plots(
    rollout: dict[str, Any],
    *,
    field: str = "density",
    out_dir: str,
    cmap: str = "turbo",
    dpi: int = 150,
    show_progress: bool = True,
) -> str:
    """
    Save one PNG per step: GT vs prediction for ``field``.
    ``rollout`` must be the dict returned by :func:`predict_autoregressive_cnextunet`.
    Returns the directory where figures were written.
    """
    if field not in FIELD_NAMES:
        raise ValueError(f"field must be one of {FIELD_NAMES}, got {field!r}")

    traj = rollout["traj"]
    pred_raw = rollout["pred_raw"]
    t0 = rollout["t0"]
    t_roll = rollout["t_roll"]
    traj_id = rollout["traj_id"]
    plot_tag = rollout.get("plot_tag", "rollout")

    field_idx = FIELD_NAMES.index(field)
    traj_cpu = traj.detach().cpu()
    pred_cpu = pred_raw.detach().cpu()

    gt_all = traj_cpu[t0 : t0 + t_roll, :, :, field_idx].numpy()
    vmin, vmax = np.nanpercentile(gt_all, [1, 99])

    out_sub = os.path.join(out_dir, f"traj{traj_id:03d}_{field}_{plot_tag}")
    os.makedirs(out_sub, exist_ok=True)

    it = range(t_roll)
    if show_progress:
        it = tqdm(it, desc="save plots")

    for step in it:
        t_idx = t0 + step
        gt = traj_cpu[t_idx, :, :, field_idx].numpy()
        pr = pred_cpu[step, :, :, field_idx].numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        im0 = axes[0].imshow(gt, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(f"GT t={t_idx}")
        axes[1].imshow(pr, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Pred step {step} (t={t_idx})")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im0, ax=axes, fraction=0.046, pad=0.04)
        fig.savefig(os.path.join(out_sub, f"step_{step:02d}_t{t_idx:03d}.png"), dpi=dpi)
        plt.close(fig)

    return out_sub


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autoregressive rollout: CNextU-Net / UNetConvNext (delta or full-frame)."
    )
    p.add_argument("--base-path", type=str, default="./data")
    p.add_argument("--hdf5-path", type=str, default="")
    p.add_argument("--traj-id", type=int, default=0)
    p.add_argument("--t-start", type=int, default=0)
    p.add_argument("--t-roll", type=int, default=10, dest="t_roll")
    p.add_argument(
        "--prediction-mode", type=str, default="full", choices=["delta", "full"],
        help='Prediction target: "full" (default, used by both pretrained and local checkpoints).',
    )
    p.add_argument(
        "--model-pt", "--checkpoint",
        type=str, default="", dest="model_pt", metavar="PATH",
        help="Path to CNextU-Net weights (.pt). Overrides --trained-model-dir.",
    )
    p.add_argument(
        "--trained-model-dir", type=str, default="",
        help="Directory with best_cnextunet_by_valid_*.pt.",
    )
    p.add_argument(
        "--checkpoint-prefer", type=str, default="rollout",
        choices=["rollout", "1step"], dest="checkpoint_prefer",
    )
    p.add_argument("--field", type=str, default="density", choices=FIELD_NAMES)
    p.add_argument("--out-dir", type=str, default="./autoregressive_plots_cnextunet")
    p.add_argument("--cmap", type=str, default="turbo")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--blocks-per-stage", type=int, default=2, dest="blocks_per_stage")
    p.add_argument("--blocks-at-neck", type=int, default=1, dest="blocks_at_neck")
    p.add_argument("--init-features", type=int, default=42, dest="init_features")
    p.add_argument("--kernel-size", type=int, default=7, dest="kernel_size")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else pick_device()

    hdf5_path = args.hdf5_path.strip() or None
    model_pt = args.model_pt.strip() or None
    tdir = args.trained_model_dir.strip() or None

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"device={device}")
    print(
        f"hdf5={resolve_test_hdf5_path(args.base_path, hdf5_path)} "
        f"traj_id={args.traj_id} t_start={args.t_start} t_roll={args.t_roll} "
        f"field={args.field} prediction_mode={args.prediction_mode}"
    )
    if model_pt:
        print(f"model_pt={model_pt}")
    elif tdir:
        print(f"trained_model_dir={tdir}")

    rollout = predict_autoregressive_cnextunet(
        base_path=args.base_path,
        hdf5_path=hdf5_path,
        traj_id=args.traj_id,
        t_start=args.t_start,
        t_roll=args.t_roll,
        device=device,
        prediction_mode=args.prediction_mode,
        model_pt_path=model_pt,
        trained_model_dir=tdir,
        checkpoint_prefer=args.checkpoint_prefer,
        stages=args.stages,
        blocks_per_stage=args.blocks_per_stage,
        blocks_at_neck=args.blocks_at_neck,
        init_features=args.init_features,
        kernel_size=args.kernel_size,
    )

    print(f"model_backend={rollout['model_backend']} plot_tag={rollout['plot_tag']}")
    print("rollout… done")
    out_sub = save_rollout_plots(
        rollout,
        field=args.field,
        out_dir=args.out_dir,
        cmap=args.cmap,
        show_progress=True,
    )
    print(f"Saved {args.t_roll} figures under {out_sub}")


if __name__ == "__main__":
    main()
