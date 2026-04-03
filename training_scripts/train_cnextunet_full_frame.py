import os
import csv
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F_fn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from timm.layers import DropPath
from tqdm import tqdm

from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from plot_functions.plot_metrics import save_epoch_metrics_figure

# =========================================================================
# CNextU-Net model definition
# =========================================================================

_conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
_conv_t = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}

_permute_strings = {
    2: ["N C H W -> N H W C", "N H W C -> N C H W"],
    3: ["N C D H W -> N D H W C", "N D H W C -> N C D H W"],
}


class _LayerNorm(nn.Module):
    def __init__(self, dim, n_spatial, eps=1e-6, fmt="channels_last"):
        super().__init__()
        shape = (dim,) if fmt == "channels_last" else (dim,) + (1,) * n_spatial
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.n_spatial = n_spatial
        self.eps = eps
        self.fmt = fmt
        self.norm_shape = (dim,)

    def forward(self, x):
        if self.fmt == "channels_last":
            return F_fn.layer_norm(x, self.norm_shape, self.weight, self.bias, self.eps)
        x = F_fn.normalize(x, p=2, dim=1, eps=self.eps) * self.weight
        return x


class _Upsample(nn.Module):
    def __init__(self, d_in, d_out, n_spatial=2):
        super().__init__()
        self.block = nn.Sequential(
            _LayerNorm(d_in, n_spatial, fmt="channels_first"),
            _conv_t[n_spatial](d_in, d_out, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class _Downsample(nn.Module):
    def __init__(self, d_in, d_out, n_spatial=2):
        super().__init__()
        self.block = nn.Sequential(
            _LayerNorm(d_in, n_spatial, fmt="channels_first"),
            _conv[n_spatial](d_in, d_out, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class _Block(nn.Module):
    def __init__(self, dim, n_spatial, kernel_size=7, drop_path=0.0,
                 layer_scale_init=1e-6):
        super().__init__()
        self.n_spatial = n_spatial
        padding = kernel_size // 2
        self.dwconv = _conv[n_spatial](dim, dim, kernel_size=kernel_size,
                                       padding=padding, groups=dim)
        self.norm = _LayerNorm(dim, n_spatial)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)
            if layer_scale_init > 0 else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = rearrange(x, _permute_strings[self.n_spatial][0])
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = rearrange(x, _permute_strings[self.n_spatial][1])
        return shortcut + self.drop_path(x)


class _Stage(nn.Module):
    def __init__(self, d_in, d_out, n_spatial, depth=1, kernel_size=7,
                 drop_path=0.0, mode="down", skip_project=False):
        super().__init__()
        self.skip_proj = (
            _conv[n_spatial](2 * d_in, d_in, 1) if skip_project else nn.Identity()
        )
        if mode == "down":
            self.resample = _Downsample(d_in, d_out, n_spatial)
        elif mode == "up":
            self.resample = _Upsample(d_in, d_out, n_spatial)
        else:
            self.resample = nn.Identity()

        self.blocks = nn.ModuleList([
            _Block(d_in, n_spatial, kernel_size=kernel_size, drop_path=drop_path)
            for _ in range(depth)
        ])

    def forward(self, x):
        x = self.skip_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.resample(x)


class CNextUNet(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int = 2,
        stages: int = 4,
        blocks_per_stage: int = 2,
        blocks_at_neck: int = 1,
        init_features: int = 42,
        kernel_size: int = 7,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        feat = init_features
        enc_dims = [feat * 2**i for i in range(stages + 1)]
        dec_dims = [feat * 2**i for i in range(stages, -1, -1)]

        self.in_proj = _conv[n_spatial_dims](dim_in, feat, kernel_size=3, padding=1)
        self.out_proj = _conv[n_spatial_dims](feat, dim_out, kernel_size=3, padding=1)

        encoder, decoder = [], []
        for i in range(stages):
            encoder.append(_Stage(
                enc_dims[i], enc_dims[i + 1], n_spatial_dims,
                depth=blocks_per_stage, kernel_size=kernel_size, mode="down",
            ))
            decoder.append(_Stage(
                dec_dims[i], dec_dims[i + 1], n_spatial_dims,
                depth=blocks_per_stage, kernel_size=kernel_size, mode="up",
                skip_project=(i != 0),
            ))
        self.encoder = nn.ModuleList(encoder)
        self.neck = _Stage(
            enc_dims[-1], enc_dims[-1], n_spatial_dims,
            depth=blocks_at_neck, kernel_size=kernel_size, mode="neck",
        )
        self.decoder = nn.ModuleList(decoder)

    def _maybe_ckpt(self, layer, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            return checkpoint(layer, *args, use_reentrant=False, **kwargs)
        return layer(*args, **kwargs)

    def forward(self, x):
        x = self.in_proj(x)
        skips = []
        for enc in self.encoder:
            skips.append(x)
            x = self._maybe_ckpt(enc, x)
        x = self.neck(x)
        for j, dec in enumerate(self.decoder):
            if j > 0:
                x = torch.cat([x, skips[-j]], dim=1)
            x = dec(x)
        return self.out_proj(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .pt checkpoint to resume training from")
    args = parser.parse_args()

    # =========================================================================
    # Config
    # =========================================================================
    base_path = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    N_STEPS_INPUT = 4
    N_STEPS_OUTPUT = 4

    BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 16

    EPOCHS = 50
    LR = 1e-3

    LAMBDA_ONE_STEP = 0.5
    LAMBDA_ROLLOUT = 1.0
    USE_STEP_WEIGHTS = True

    # =========================================================================
    # Metrics logging
    # =========================================================================
    METRICS_CSV_PATH = "train_metrics_cnextunet.csv"
    METRICS_CSV_FIELDNAMES = [
        "epoch",
        "train_total",
        "train_one_step",
        "train_rollout",
        "valid_1step_mean_per_field",
        "valid_1step_sqrt_mean_mse_over_mean_var",
        "valid_1step_sqrt_sum_mse_over_sum_var",
        "valid_1step_per_field",
        "valid_rollout_mean_per_field",
        "valid_rollout_sqrt_mean_mse_over_mean_var",
        "valid_rollout_sqrt_sum_mse_over_sum_var",
        "valid_rollout_per_field",
    ]
    metrics_csv_needs_header = (
        not os.path.exists(METRICS_CSV_PATH) or os.path.getsize(METRICS_CSV_PATH) == 0
    )

    # =========================================================================
    # Datasets
    # =========================================================================
    train_dataset = WellDataset(
        well_base_path=f"{base_path}/datasets",
        well_dataset_name="turbulent_radiative_layer_2D",
        well_split_name="train",
        n_steps_input=N_STEPS_INPUT,
        n_steps_output=N_STEPS_OUTPUT,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)

    N_FIELDS = train_dataset.metadata.n_fields

    valid_dataset_1step = WellDataset(
        well_base_path=f"{base_path}/datasets",
        well_dataset_name="turbulent_radiative_layer_2D",
        well_split_name="valid",
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )
    valid_loader_1step = DataLoader(valid_dataset_1step, shuffle=False, batch_size=EVAL_BATCH_SIZE, num_workers=4)

    valid_dataset_rollout = WellDataset(
        well_base_path=f"{base_path}/datasets",
        well_dataset_name="turbulent_radiative_layer_2D",
        well_split_name="valid",
        n_steps_input=4,
        n_steps_output=4,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )
    valid_loader_rollout = DataLoader(valid_dataset_rollout, shuffle=False, batch_size=EVAL_BATCH_SIZE, num_workers=4)

    # =========================================================================
    # Model  –  CNextU-Net
    # =========================================================================
    model = CNextUNet(
        dim_in=N_STEPS_INPUT * N_FIELDS,
        dim_out=1 * N_FIELDS,
        n_spatial_dims=2,
        stages=4,              # Up/Down Blocks
        blocks_per_stage=2,     # Blocks per Stage
        blocks_at_neck=1,       # Bottleneck Blocks
        init_features=42,       # Initial Dimension
        kernel_size=7,          # Spatial Filter Size
        gradient_checkpointing=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_epoch = 1
    best_valid_1step = float("inf")
    best_valid_rollout = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_valid_1step = ckpt.get("best_valid_1step", float("inf"))
            best_valid_rollout = ckpt.get("best_valid_rollout", float("inf"))
            print(f"Resumed from checkpoint '{args.resume}' (epoch {ckpt['epoch']})")
        else:
            model.load_state_dict(ckpt)
            print(f"Loaded model weights from '{args.resume}' (no optimizer/scheduler state, starting fresh from epoch 1)")

    # =========================================================================
    # Rollout helper
    # =========================================================================
    def rollout_predict_full(
        model: nn.Module,
        x_init_norm: torch.Tensor,
        rollout_steps: int,
        n_fields: int,
    ) -> torch.Tensor:
        x_roll = x_init_norm.clone()
        preds = []
        for _ in range(rollout_steps):
            x_in = rearrange(x_roll, "B Ti H W F -> B (Ti F) H W")
            pred_next = model(x_in)
            pred_next = rearrange(pred_next, "B (To F) H W -> B To H W F",
                                  To=1, F=n_fields)[:, 0]
            preds.append(pred_next)
            x_roll = torch.cat([x_roll[:, 1:], pred_next.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    # =========================================================================
    # Train losses
    # =========================================================================
    def compute_train_losses(batch: dict):
        x = batch["input_fields"].to(device)
        y = batch["output_fields"].to(device)

        pred_seq = rollout_predict_full(model, x, N_STEPS_OUTPUT, N_FIELDS)

        one_step_loss = ((pred_seq[:, 0] - y[:, 0]) ** 2).mean()
        per_step_mse = ((pred_seq - y) ** 2).mean(dim=(0, 2, 3, 4))

        if USE_STEP_WEIGHTS:
            w = torch.tensor([1.0, 1.1, 1.25, 1.5], device=device)
            w = w / w.sum()
            rollout_loss = (w * per_step_mse).sum()
        else:
            rollout_loss = per_step_mse.mean()

        total = LAMBDA_ONE_STEP * one_step_loss + LAMBDA_ROLLOUT * rollout_loss
        return total, one_step_loss, rollout_loss

    # =========================================================================
    # Validation helpers
    # =========================================================================
    def _vrmse_stats(pred: torch.Tensor, y: torch.Tensor, n_fields: int,
                     accumulators: dict):
        err2 = (pred - y) ** 2
        accumulators["sum_err2"] += err2.sum(dim=(0, 1, 2, 3))
        accumulators["sum_y"] += y.sum(dim=(0, 1, 2, 3))
        accumulators["sum_y2"] += (y * y).sum(dim=(0, 1, 2, 3))
        accumulators["count"] += y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3]

    def _finalize_vrmse(acc: dict):
        mse_f = acc["sum_err2"] / acc["count"]
        Ey = acc["sum_y"] / acc["count"]
        Ey2 = acc["sum_y2"] / acc["count"]
        Var_f = Ey2 - Ey * Ey

        per_field = torch.sqrt(mse_f / (Var_f + 1e-7))
        mean_pf = per_field.mean()
        mse_m, Var_m = mse_f.mean(), Var_f.mean()
        return {
            "per_field": per_field.detach().cpu(),
            "mean_per_field": mean_pf.item(),
            "sqrt_mean_mse_over_mean_var": torch.sqrt(mse_m / (Var_m + 1e-7)).item(),
            "sqrt_sum_mse_over_sum_var": torch.sqrt(mse_f.sum() / (Var_f.sum() + 1e-7)).item(),
        }

    def evaluate_valid_1step_vrmse(model: nn.Module, loader: DataLoader):
        model.eval()
        acc = {k: (torch.zeros(N_FIELDS, device=device) if k != "count" else 0)
               for k in ("sum_err2", "sum_y", "sum_y2", "count")}
        with torch.no_grad():
            for batch in tqdm(loader, desc="valid 1-step", leave=False):
                x = batch["input_fields"].to(device)
                y = batch["output_fields"].to(device)
                x_in = rearrange(x, "B Ti H W F -> B (Ti F) H W")
                pred = model(x_in)
                pred = rearrange(pred, "B (To F) H W -> B To H W F", To=1, F=N_FIELDS)
                _vrmse_stats(pred, y, N_FIELDS, acc)
        return _finalize_vrmse(acc)

    def evaluate_valid_rollout_vrmse(model: nn.Module, loader: DataLoader,
                                     rollout_steps: int = 4):
        model.eval()
        acc = {k: (torch.zeros(N_FIELDS, device=device) if k != "count" else 0)
               for k in ("sum_err2", "sum_y", "sum_y2", "count")}
        with torch.no_grad():
            for batch in tqdm(loader, desc="valid rollout", leave=False):
                x = batch["input_fields"].to(device)
                y = batch["output_fields"].to(device)
                pred_seq = rollout_predict_full(model, x, rollout_steps, N_FIELDS)
                _vrmse_stats(pred_seq, y, N_FIELDS, acc)
        return _finalize_vrmse(acc)

    # =========================================================================
    # Training loop
    # =========================================================================
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_total = 0.0
        train_one = 0.0
        train_roll = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"epoch {epoch:03d} train"):
            total_loss, one_step_loss, rollout_loss = compute_train_losses(batch)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total += total_loss.detach().item()
            train_one += one_step_loss.detach().item()
            train_roll += rollout_loss.detach().item()
            n_batches += 1

        scheduler.step()
        train_total /= n_batches
        train_one /= n_batches
        train_roll /= n_batches

        valid_1step = evaluate_valid_1step_vrmse(model, valid_loader_1step)
        valid_rollout = evaluate_valid_rollout_vrmse(model, valid_loader_rollout, rollout_steps=4)

        print(
            f"\nEpoch {epoch:03d} | "
            f"train total={train_total:.6f}, "
            f"train 1-step={train_one:.6f}, "
            f"train rollout={train_roll:.6f}"
        )
        print(
            f"VALID 1-step VRMSE mean(per-field)={valid_1step['mean_per_field']:.6f}, "
            f"sqrt(mean MSE / mean Var)={valid_1step['sqrt_mean_mse_over_mean_var']:.6f}, "
            f"sqrt(sum MSE / sum Var)={valid_1step['sqrt_sum_mse_over_sum_var']:.6f}"
        )
        print(f"VALID 1-step per-field={valid_1step['per_field'].tolist()}")
        print(
            f"VALID rollout-4 VRMSE mean(per-field)={valid_rollout['mean_per_field']:.6f}, "
            f"sqrt(mean MSE / mean Var)={valid_rollout['sqrt_mean_mse_over_mean_var']:.6f}, "
            f"sqrt(sum MSE / sum Var)={valid_rollout['sqrt_sum_mse_over_sum_var']:.6f}"
        )
        print(f"VALID rollout-4 per-field={valid_rollout['per_field'].tolist()}")

        row = {
            "epoch": epoch,
            "train_total": train_total,
            "train_one_step": train_one,
            "train_rollout": train_roll,
            "valid_1step_mean_per_field": valid_1step["mean_per_field"],
            "valid_1step_sqrt_mean_mse_over_mean_var": valid_1step["sqrt_mean_mse_over_mean_var"],
            "valid_1step_sqrt_sum_mse_over_sum_var": valid_1step["sqrt_sum_mse_over_sum_var"],
            "valid_1step_per_field": valid_1step["per_field"].tolist(),
            "valid_rollout_mean_per_field": valid_rollout["mean_per_field"],
            "valid_rollout_sqrt_mean_mse_over_mean_var": valid_rollout["sqrt_mean_mse_over_mean_var"],
            "valid_rollout_sqrt_sum_mse_over_sum_var": valid_rollout["sqrt_sum_mse_over_sum_var"],
            "valid_rollout_per_field": valid_rollout["per_field"].tolist(),
        }

        with open(METRICS_CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_CSV_FIELDNAMES)
            if metrics_csv_needs_header:
                writer.writeheader()
                metrics_csv_needs_header = False
            writer.writerow(row)

        save_epoch_metrics_figure(METRICS_CSV_PATH, epoch, "cnextunet_metrics")

        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_valid_1step": best_valid_1step,
            "best_valid_rollout": best_valid_rollout,
        }

        if valid_1step["mean_per_field"] < best_valid_1step:
            best_valid_1step = valid_1step["mean_per_field"]
            ckpt_state["best_valid_1step"] = best_valid_1step
            torch.save(ckpt_state, "best_cnextunet_by_valid_1step_vrmse.pt")
            print("Saved best_cnextunet_by_valid_1step_vrmse.pt")

        if valid_rollout["mean_per_field"] < best_valid_rollout:
            best_valid_rollout = valid_rollout["mean_per_field"]
            ckpt_state["best_valid_rollout"] = best_valid_rollout
            torch.save(ckpt_state, "best_cnextunet_by_valid_rollout_vrmse.pt")
            print("Saved best_cnextunet_by_valid_rollout_vrmse.pt")

    torch.save(ckpt_state, "final_cnextunet_full_frame.pt")


if __name__ == "__main__":
    main()
