"""
Read train_metrics_delta.csv and save a training-curve figure (one PNG per epoch).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_metrics(csv_path: str) -> list[dict[str, str]]:
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def save_epoch_metrics_figure(csv_path: str, epoch: int, out_dir: str) -> None:
    rows = _read_metrics(csv_path)
    if not rows:
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ep = [_f(r, "epoch") for r in rows]
    train_total = [_f(r, "train_total") for r in rows]
    train_one = [_f(r, "train_one_step") for r in rows]
    train_roll = [_f(r, "train_rollout") for r in rows]
    v1_mean = [_f(r, "valid_1step_mean_per_field") for r in rows]
    v1_sqrt_mean = [_f(r, "valid_1step_sqrt_mean_mse_over_mean_var") for r in rows]
    v1_sqrt_sum = [_f(r, "valid_1step_sqrt_sum_mse_over_sum_var") for r in rows]
    vr_mean = [_f(r, "valid_rollout_mean_per_field") for r in rows]
    vr_sqrt_mean = [_f(r, "valid_rollout_sqrt_mean_mse_over_mean_var") for r in rows]
    vr_sqrt_sum = [_f(r, "valid_rollout_sqrt_sum_mse_over_sum_var") for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Delta FNO metrics (through epoch {int(epoch)})", fontsize=12)

    ax0 = axes[0]
    ax0.plot(ep, train_total, label="train total", color="#1f77b4")
    ax0.plot(ep, train_one, label="train 1-step", color="#ff7f0e", alpha=0.9)
    ax0.plot(ep, train_roll, label="train rollout", color="#2ca02c", alpha=0.9)
    ax0.set_ylabel("MSE (train)")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(ep, v1_mean, label="valid 1-step mean(per-field) VRMSE", color="#d62728")
    ax1.plot(ep, vr_mean, label="valid rollout mean(per-field) VRMSE", color="#9467bd")
    ax1.plot(ep, v1_sqrt_mean, "--", label="valid 1-step √(mean MSE/mean Var)", color="#d62728", alpha=0.6)
    ax1.plot(ep, vr_sqrt_mean, "--", label="valid rollout √(mean MSE/mean Var)", color="#9467bd", alpha=0.6)
    ax1.plot(ep, v1_sqrt_sum, ":", label="valid 1-step √(sum MSE/sum Var)", color="#d62728", alpha=0.5)
    ax1.plot(ep, vr_sqrt_sum, ":", label="valid rollout √(sum MSE/sum Var)", color="#9467bd", alpha=0.5)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("VRMSE / variants")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"metrics_epoch_{epoch:04d}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
