"""
Autoregressive rollout for turbulent_radiative_layer_2D.

- **HuggingFace** ``polymathic-ai/FNO-turbulent_radiative_layer_2D``: always *delta* targets
  (normalized Δ → reconstruct full frame), same as the benchmark.
- **Your checkpoints** (``neuralop`` FNO from ``train_fno_delta.py`` / ``train_fno_full_frame.py``):
  use ``prediction_mode="delta"`` or ``prediction_mode="full"``.

Uses WellDataset (test split) for Z-score statistics. Trajectories come from the test HDF5
(``traj_id``, ``t_start``).

Notebook (pretrained):
    from autoregressive_pretrained_fno import predict_autoregressive_fno, save_rollout_plots

    out = predict_autoregressive_fno(base_path="./data", hdf5_path=".../test/....hdf5", t_roll=10)
    save_rollout_plots(out, field="density", out_dir="./plots")

Notebook (explicit ``.pt`` path):
    out = predict_autoregressive_fno(
        model_pt_path="models_trained/fno_delta_lr1e-3_50ep/best_by_valid_rollout_vrmse_delta.pt",
        prediction_mode="delta",
        t_roll=10,
    )

Notebook (local delta training run — auto-pick ``best_by_valid_*.pt`` in folder):
    from autoregressive_pretrained_fno import LOCAL_DELTA_TRAINED_DIR, predict_autoregressive_fno

    out = predict_autoregressive_fno(
        trained_model_dir=str(LOCAL_DELTA_TRAINED_DIR),
        prediction_mode="delta",
        t_start=50,
        t_roll=10,
    )

Notebook (local full-frame run):
    from autoregressive_pretrained_fno import LOCAL_FULL_FRAME_TRAINED_DIR, predict_autoregressive_fno

    out = predict_autoregressive_fno(
        trained_model_dir=str(LOCAL_FULL_FRAME_TRAINED_DIR),
        prediction_mode="full",
        t_roll=10,
    )

CLI:
    python autoregressive_pretrained_fno.py --model-pt models_trained/.../best_by_valid_rollout_vrmse_delta.pt --prediction-mode delta
    python autoregressive_pretrained_fno.py --prediction-mode delta --trained-model-dir models_trained/fno_delta_lr1e-3_50ep
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

from neuralop.models import FNO as NeuralOpFNO
from the_well.benchmark.models import FNO as WellFNO
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization

FIELD_NAMES = ["density", "pressure", "velocity_x", "velocity_y"]
N_STEPS_INPUT = 4
DEFAULT_N_MODES = (16, 16)
PRETRAINED_FNO_REPO = "polymathic-ai/FNO-turbulent_radiative_layer_2D"
DATASET_NAME = "turbulent_radiative_layer_2D"

_REPO_ROOT = Path(__file__).resolve().parent
# Default dirs for trained runs in this repo (override with model_pt_path / trained_model_dir).
LOCAL_DELTA_TRAINED_DIR = _REPO_ROOT / "models_trained" / "fno_delta_lr1e-3_50ep"
LOCAL_FULL_FRAME_TRAINED_DIR = _REPO_ROOT / "models_trained" / "fno_full_frame_lr1e-3_50ep"

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


def _normalize_n_modes(n_modes: Sequence[int] | tuple[int, int], *, name: str = "n_modes") -> tuple[int, int]:
    """Ensure ``(n_modes_x, n_modes_y)`` for ``neuralop`` FNO construction."""
    if len(n_modes) != 2:
        raise ValueError(f"{name} must be a length-2 sequence, e.g. (16, 16); got {n_modes!r}")
    return (int(n_modes[0]), int(n_modes[1]))


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
        base_path,
        "datasets",
        DATASET_NAME,
        "data",
        "test",
        "*.hdf5",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[0]


def resolve_test_hdf5_path(base_path: str, hdf5_path: str | None) -> str:
    """Return an existing HDF5 path; if hdf5_path is empty, use the first test *.hdf5 under base_path."""
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


def load_pretrained_well_fno(
    device: torch.device,
    repo_id: str = PRETRAINED_FNO_REPO,
) -> torch.nn.Module:
    return WellFNO.from_pretrained(repo_id).to(device).eval()


def load_neuralop_fno_from_checkpoint(
    checkpoint_path: str,
    n_fields: int,
    device: torch.device,
    *,
    n_modes: Sequence[int] | tuple[int, int] = DEFAULT_N_MODES,
    hidden_channels: int = 128,
    n_layers: int = 4,
) -> torch.nn.Module:
    """Load ``neuralop.models.FNO`` weights saved via ``torch.save(state_dict, ...)`` from this repo's training scripts."""
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

    nm = _normalize_n_modes(n_modes, name="n_modes")
    model = NeuralOpFNO(
        n_modes=nm,
        in_channels=N_STEPS_INPUT * n_fields,
        out_channels=n_fields,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def resolve_checkpoint_in_dir(
    directory: str | Path,
    prediction_mode: PredictionMode,
    prefer: CheckpointPrefer = "rollout",
) -> str:
    """Pick a default ``.pt`` inside a training output folder."""
    d = Path(directory).expanduser().resolve()
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    if prediction_mode == "delta":
        rollout_first = [
            d / "best_by_valid_rollout_vrmse_delta.pt",
            d / "best_by_valid_1step_vrmse_delta.pt",
        ]
        step_first = [rollout_first[1], rollout_first[0]]
    else:
        rollout_first = [
            d / "best_by_valid_rollout_vrmse.pt",
            d / "best_by_valid_1step_vrmse.pt",
        ]
        step_first = [rollout_first[1], rollout_first[0]]

    order = rollout_first if prefer == "rollout" else step_first
    for p in order:
        if p.is_file():
            return str(p)

    raise FileNotFoundError(
        f"No matching checkpoint in {d} for prediction_mode={prediction_mode!r}, prefer={prefer!r}. "
        f"Expected one of: {[str(x.name) for x in rollout_first]}"
    )


def _coalesce_model_pt_path(
    model_pt_path: str | None,
    checkpoint_path: str | None,
) -> str | None:
    """Return a single ``.pt`` path; ``model_pt_path`` and ``checkpoint_path`` are aliases."""
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
    prediction_mode: PredictionMode,
    prefer: CheckpointPrefer,
    *,
    checkpoint_path: str | None = None,
) -> str | None:
    """
    Returns a ``.pt`` path for a local model, or ``None`` to use the HuggingFace pretrained FNO.
    ``model_pt_path`` / ``checkpoint_path`` (same meaning) win over ``trained_model_dir``.
    """
    cp = _coalesce_model_pt_path(model_pt_path, checkpoint_path)
    if cp:
        if not os.path.isfile(cp):
            raise FileNotFoundError(cp)
        return cp
    td = trained_model_dir
    if td is not None and str(td).strip():
        return resolve_checkpoint_in_dir(td, prediction_mode, prefer=prefer)
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


def autoregressive_rollout_delta(
    model: torch.nn.Module,
    x_init_norm: torch.Tensor,
    t_roll: int,
    mean_full: torch.Tensor,
    std_full: torch.Tensor,
    mean_delta: torch.Tensor,
    std_delta: torch.Tensor,
) -> torch.Tensor:
    """
    Model output: normalized **delta** (same convention as the benchmark FNO).

    x_init_norm: (1, Ti=4, H, W, F) normalized full states.
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

            pred_delta_norm = pred[:, 0]
            last_input_full_norm = x_roll[:, -1]

            last_raw = last_input_full_norm * std_full + mean_full
            delta_raw = pred_delta_norm * std_delta + mean_delta
            pred_full_raw = last_raw + delta_raw

            preds.append(pred_full_raw.squeeze(0))

            pred_full_norm = (pred_full_raw - mean_full) / std_full
            x_roll = torch.cat([x_roll[:, 1:], pred_full_norm.unsqueeze(1)], dim=1)

    return torch.stack(preds, dim=0)


def autoregressive_rollout_full_frame(
    model: torch.nn.Module,
    x_init_norm: torch.Tensor,
    t_roll: int,
    mean_full: torch.Tensor,
    std_full: torch.Tensor,
) -> torch.Tensor:
    """
    Model output: **normalized full** next frame (``train_fno_full_frame.py``).

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


# Backward-compatible name (same as autoregressive_rollout_delta).
autoregressive_rollout_pretrained = autoregressive_rollout_delta


def predict_autoregressive_fno(
    *,
    base_path: str = "./data",
    hdf5_path: str | None = None,
    traj_id: int = 0,
    t_start: int = 0,
    t_roll: int = 10,
    device: torch.device | str | None = None,
    model: torch.nn.Module | None = None,
    repo_id: str = PRETRAINED_FNO_REPO,
    prediction_mode: PredictionMode = "delta",
    model_pt_path: str | None = None,
    checkpoint_path: str | None = None,
    trained_model_dir: str | Path | None = None,
    checkpoint_prefer: CheckpointPrefer = "rollout",
    n_modes: Sequence[int] | tuple[int, int] = DEFAULT_N_MODES,
    n_layers: int = 4,
) -> dict[str, Any]:
    """
    Load test normalization, trajectory from HDF5, run autoregressive rollout.

    **Model selection** (first match wins):

    1. If ``model`` is passed, it is used (you must set ``prediction_mode`` correctly).
    2. Else if ``model_pt_path`` (or alias ``checkpoint_path``) or ``trained_model_dir`` is set,
       load ``neuralop`` FNO from disk (uses ``n_modes`` and ``n_layers`` for architecture; must match training).
    3. Else load HuggingFace ``WellFNO`` (``prediction_mode`` must be ``\"delta\"``; ``n_modes`` ignored).

    Returns a dict with:
        pred_raw, traj, t0, t_indices, model, device, normalization tensors, n_fields,
        hdf5_path, traj_id, t_start, t_roll,
        prediction_mode, model_backend (``\"pretrained_well\"`` | ``\"neuralop_checkpoint\"``),
        model_pt_path (resolved ``.pt`` path, ``None`` if HuggingFace), plot_tag (subfolder hint for plots),
        n_modes ``(mx, my)`` and ``n_layers`` used when instantiating ``neuralop`` FNO
        (ignored for HuggingFace / ``model=``).
    """
    dev = _as_device(device)
    path = resolve_test_hdf5_path(base_path, hdf5_path)
    mean_full, std_full, mean_d, std_d, n_fields = load_test_normalization(base_path, dev)
    nm = _normalize_n_modes(n_modes)

    ckpt_resolved = resolve_checkpoint_path(
        model_pt_path,
        trained_model_dir,
        prediction_mode,
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
        m = load_neuralop_fno_from_checkpoint(
            ckpt_resolved, n_fields, dev, n_modes=nm, n_layers=n_layers
        )
        mode = prediction_mode
        backend = "neuralop_checkpoint"
        ckpt_used = ckpt_resolved
    else:
        if prediction_mode != "delta":
            raise ValueError(
                'HuggingFace pretrained FNO only supports prediction_mode="delta" '
                '(normalized Δ). Use trained_model_dir= / model_pt_path= for full-frame checkpoints.'
            )
        m = load_pretrained_well_fno(dev, repo_id=repo_id)
        mode = "delta"
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
            m, x_init_norm, t_roll, mean_full, std_full, mean_d, std_d
        )

    t0 = t_start + N_STEPS_INPUT
    if backend == "pretrained_well":
        plot_tag = "pretrained_delta"
    elif backend == "neuralop_checkpoint":
        stem = Path(ckpt_used).stem if ckpt_used else "checkpoint"
        plot_tag = f"{mode}_{stem}"
    else:
        plot_tag = f"custom_{mode}"

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
        "n_modes": nm,
        "n_layers": n_layers,
    }


def predict_autoregressive_pretrained_fno(
    **kwargs: Any,
) -> dict[str, Any]:
    """Backward-compatible alias: same as :func:`predict_autoregressive_fno` with pretrained defaults."""
    return predict_autoregressive_fno(**kwargs)


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
    ``rollout`` must be the dict returned by :func:`predict_autoregressive_fno`.
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
        description="Autoregressive rollout: HuggingFace or local neuralop FNO (delta or full-frame)."
    )
    p.add_argument("--base-path", type=str, default="./data", help="Contains datasets/ subtree.")
    p.add_argument(
        "--hdf5-path",
        type=str,
        default="",
        help="Test HDF5 file. If empty, uses the first *.hdf5 under .../data/test/.",
    )
    p.add_argument("--traj-id", type=int, default=0)
    p.add_argument(
        "--t-start",
        type=int,
        default=0,
        help="First time index of the 4-frame context: uses traj[t_start : t_start + 4].",
    )
    p.add_argument(
        "--t-roll",
        type=int,
        default=10,
        dest="t_roll",
        help="Number of autoregressive steps to predict and plot.",
    )
    p.add_argument(
        "--prediction-mode",
        type=str,
        default="delta",
        choices=["delta", "full"],
        help='Training target: "delta" (Δ in z-score) or "full" (next frame in z-score of full state).',
    )
    p.add_argument(
        "--model-pt",
        "--checkpoint",
        type=str,
        default="",
        dest="model_pt",
        metavar="PATH",
        help="Path to neuralop FNO weights (.pt). Overrides --trained-model-dir. Same as --checkpoint.",
    )
    p.add_argument(
        "--trained-model-dir",
        type=str,
        default="",
        help=(
            "Directory with best_by_valid_*.pt (e.g. models_trained/fno_delta_lr1e-3_50ep). "
            "If empty, uses HuggingFace pretrained FNO (delta only)."
        ),
    )
    p.add_argument(
        "--checkpoint-prefer",
        type=str,
        default="rollout",
        choices=["rollout", "1step"],
        dest="checkpoint_prefer",
        help="When resolving --trained-model-dir, prefer rollout or 1-step metric checkpoint.",
    )
    p.add_argument("--field", type=str, default="density", choices=FIELD_NAMES)
    p.add_argument("--out-dir", type=str, default="./autoregressive_plots_pretrained_fno")
    p.add_argument("--cmap", type=str, default="turbo")
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda | mps | cpu. Empty = auto.",
    )
    p.add_argument(
        "--n-modes",
        type=int,
        nargs=2,
        metavar=("MX", "MY"),
        default=list(DEFAULT_N_MODES),
        dest="n_modes",
        help="Fourier modes per spatial dim for neuralop FNO (must match checkpoint training). Default: 16 16.",
    )
    p.add_argument(
        "--n-layers",
        type=int,
        default=4,
        dest="n_layers",
        help="Number of FNO layers for neuralop checkpoint loading (must match training). Default: 4.",
    )
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

    rollout = predict_autoregressive_fno(
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
        n_modes=tuple(args.n_modes),
        n_layers=args.n_layers,
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
