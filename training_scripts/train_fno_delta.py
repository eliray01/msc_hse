"""
Train FNO to predict normalized *delta* (like the authors' pretrained model), then reconstruct
the next full frame in raw space and map back to normalized full space for loss / VRMSE.

Loss schedule matches train.py: weighted 1-step MSE + weighted multi-step rollout MSE on
reconstructed normalized full states.
"""
import os
import csv

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm

from neuralop.models import FNO
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization

# -------------------------
# config
# -------------------------
base_path = "./data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

N_STEPS_INPUT = 4
N_STEPS_OUTPUT = 4

BATCH_SIZE = 16
EVAL_BATCH_SIZE = 4

EPOCHS = 50
LR = 1e-3

LAMBDA_ONE_STEP = 0.5
LAMBDA_ROLLOUT = 1.0
USE_STEP_WEIGHTS = True

# -------------------------
# metrics logging (separate from full-frame train.py)
# -------------------------
METRICS_CSV_PATH = "train_metrics_delta.csv"
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
_metrics_csv_needs_header = not os.path.exists(METRICS_CSV_PATH) or os.path.getsize(METRICS_CSV_PATH) == 0

# -------------------------
# train dataset: normalized full inputs / full targets (same as train.py)
# -------------------------
train_dataset = WellDataset(
    well_base_path=f"{base_path}/datasets",
    well_dataset_name="turbulent_radiative_layer_2D",
    well_split_name="train",
    n_steps_input=N_STEPS_INPUT,
    n_steps_output=N_STEPS_OUTPUT,
    use_normalization=True,
    normalization_type=ZScoreNormalization,
)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=4,
)

F = train_dataset.metadata.n_fields

# Z-score stats for full state and delta (same as authors_eval)
_norm = train_dataset.norm
mean_full = _norm.flattened_means["variable"].to(device)  # (F,)
std_full = _norm.flattened_stds["variable"].to(device)
mean_delta = _norm.flattened_means_delta["variable"].to(device)
std_delta = _norm.flattened_stds_delta["variable"].to(device)

# -------------------------
# valid datasets
# -------------------------
valid_dataset_1step = WellDataset(
    well_base_path=f"{base_path}/datasets",
    well_dataset_name="turbulent_radiative_layer_2D",
    well_split_name="valid",
    n_steps_input=4,
    n_steps_output=1,
    use_normalization=True,
    normalization_type=ZScoreNormalization,
)

valid_loader_1step = DataLoader(
    valid_dataset_1step,
    shuffle=False,
    batch_size=EVAL_BATCH_SIZE,
    num_workers=4,
)

valid_dataset_rollout = WellDataset(
    well_base_path=f"{base_path}/datasets",
    well_dataset_name="turbulent_radiative_layer_2D",
    well_split_name="valid",
    n_steps_input=4,
    n_steps_output=4,
    use_normalization=True,
    normalization_type=ZScoreNormalization,
)

valid_loader_rollout = DataLoader(
    valid_dataset_rollout,
    shuffle=False,
    batch_size=EVAL_BATCH_SIZE,
    num_workers=4,
)

# -------------------------
# model: outputs normalized DELTA for one step (F channels)
# -------------------------
model = FNO(
    n_modes=(16, 16),
    in_channels=N_STEPS_INPUT * F,
    out_channels=1 * F,
    hidden_channels=128,
    n_layers=4,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def delta_norm_to_next_full_norm(
    x_roll_norm: torch.Tensor,
    pred_delta_norm: torch.Tensor,
) -> torch.Tensor:
    """
    x_roll_norm: (B, Ti, H, W, F) — window of normalized full states
    pred_delta_norm: (B, H, W, F) — model output (normalized delta)

    Returns next-frame normalized full state (B, H, W, F).
    """
    last_input_full_norm = x_roll_norm[:, -1]  # (B,H,W,F)
    last_raw = last_input_full_norm * std_full + mean_full
    delta_raw = pred_delta_norm * std_delta + mean_delta
    pred_full_raw = last_raw + delta_raw
    return (pred_full_raw - mean_full) / std_full


def rollout_predict_delta_reconstructed(
    model: torch.nn.Module,
    x_init_norm: torch.Tensor,
    rollout_steps: int,
    n_fields: int,
) -> torch.Tensor:
    """
    Autoregressive rollout: each step model predicts normalized delta, then we reconstruct
    normalized full frame for the window update.

    Returns:
        pred_seq_norm: (B, T, H, W, F) reconstructed normalized full states
    """
    x_roll = x_init_norm.clone()
    preds = []

    for _ in range(rollout_steps):
        x_in = rearrange(x_roll, "B Ti H W F -> B (Ti F) H W")
        pred = model(x_in)
        pred_delta_norm = rearrange(
            pred,
            "B (To F) H W -> B To H W F",
            To=1,
            F=n_fields,
        )[:, 0]

        pred_full_norm = delta_norm_to_next_full_norm(x_roll, pred_delta_norm)
        preds.append(pred_full_norm)
        x_roll = torch.cat([x_roll[:, 1:], pred_full_norm.unsqueeze(1)], dim=1)

    return torch.stack(preds, dim=1)


def compute_train_losses(batch: dict):
    x = batch["input_fields"].to(device)
    y = batch["output_fields"].to(device)

    pred_seq_norm = rollout_predict_delta_reconstructed(
        model=model,
        x_init_norm=x,
        rollout_steps=N_STEPS_OUTPUT,
        n_fields=F,
    )

    one_step_loss = ((pred_seq_norm[:, 0] - y[:, 0]) ** 2).mean()

    per_step_mse = ((pred_seq_norm - y) ** 2).mean(dim=(0, 2, 3, 4))

    if USE_STEP_WEIGHTS:
        step_weights = torch.tensor([1.0, 1.1, 1.25, 1.5], device=device)
        step_weights = step_weights / step_weights.sum()
        rollout_loss = (step_weights * per_step_mse).sum()
    else:
        rollout_loss = per_step_mse.mean()

    total_loss = LAMBDA_ONE_STEP * one_step_loss + LAMBDA_ROLLOUT * rollout_loss
    return total_loss, one_step_loss, rollout_loss


def evaluate_valid_1step_vrmse(model: torch.nn.Module, loader: DataLoader):
    model.eval()

    sum_err2 = torch.zeros(F, device=device)
    sum_y = torch.zeros(F, device=device)
    sum_y2 = torch.zeros(F, device=device)
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="valid 1-step", leave=False):
            x = batch["input_fields"].to(device)
            y = batch["output_fields"].to(device)

            x_in = rearrange(x, "B Ti H W F -> B (Ti F) H W")
            pred = model(x_in)
            pred_delta_norm = rearrange(
                pred, "B (To F) H W -> B To H W F", To=1, F=F
            )[:, 0]

            pred_full_norm = delta_norm_to_next_full_norm(x, pred_delta_norm).unsqueeze(1)

            err2 = (pred_full_norm - y) ** 2

            sum_err2 += err2.sum(dim=(0, 1, 2, 3))
            sum_y += y.sum(dim=(0, 1, 2, 3))
            sum_y2 += (y * y).sum(dim=(0, 1, 2, 3))
            count += y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3]

    mse_f = sum_err2 / count
    Ey_f = sum_y / count
    Ey2_f = sum_y2 / count
    Var_f = Ey2_f - Ey_f * Ey_f

    vrmse_per_field = torch.sqrt(mse_f / (Var_f + 1e-7))
    vrmse_mean_fields = vrmse_per_field.mean()

    mse_mean = mse_f.mean()
    Var_mean = Var_f.mean()
    vrmse_fields_averaged_then_sqrt = torch.sqrt(mse_mean / (Var_mean + 1e-7))
    vrmse_flattened = torch.sqrt(mse_f.sum() / (Var_f.sum() + 1e-7))

    return {
        "per_field": vrmse_per_field.detach().cpu(),
        "mean_per_field": vrmse_mean_fields.item(),
        "sqrt_mean_mse_over_mean_var": vrmse_fields_averaged_then_sqrt.item(),
        "sqrt_sum_mse_over_sum_var": vrmse_flattened.item(),
    }


def evaluate_valid_rollout_vrmse(model: torch.nn.Module, loader: DataLoader, rollout_steps: int = 4):
    model.eval()

    sum_err2 = torch.zeros(F, device=device)
    sum_y = torch.zeros(F, device=device)
    sum_y2 = torch.zeros(F, device=device)
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="valid rollout", leave=False):
            x = batch["input_fields"].to(device)
            y = batch["output_fields"].to(device)

            pred_seq = rollout_predict_delta_reconstructed(
                model=model,
                x_init_norm=x,
                rollout_steps=rollout_steps,
                n_fields=F,
            )

            err2 = (pred_seq - y) ** 2

            sum_err2 += err2.sum(dim=(0, 1, 2, 3))
            sum_y += y.sum(dim=(0, 1, 2, 3))
            sum_y2 += (y * y).sum(dim=(0, 1, 2, 3))
            count += y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3]

    mse_f = sum_err2 / count
    Ey_f = sum_y / count
    Ey2_f = sum_y2 / count
    Var_f = Ey2_f - Ey_f * Ey_f

    vrmse_per_field = torch.sqrt(mse_f / (Var_f + 1e-7))
    vrmse_mean_fields = vrmse_per_field.mean()

    mse_mean = mse_f.mean()
    Var_mean = Var_f.mean()
    vrmse_fields_averaged_then_sqrt = torch.sqrt(mse_mean / (Var_mean + 1e-7))
    vrmse_flattened = torch.sqrt(mse_f.sum() / (Var_f.sum() + 1e-7))

    return {
        "per_field": vrmse_per_field.detach().cpu(),
        "mean_per_field": vrmse_mean_fields.item(),
        "sqrt_mean_mse_over_mean_var": vrmse_fields_averaged_then_sqrt.item(),
        "sqrt_sum_mse_over_sum_var": vrmse_flattened.item(),
    }


best_valid_1step = float("inf")
best_valid_rollout = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()

    train_total = 0.0
    train_one = 0.0
    train_roll = 0.0
    n_batches = 0

    for batch in tqdm(train_loader, desc=f"epoch {epoch:03d} train (delta)"):
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
        f"VALID 1-step VRMSE (reconstructed full) mean(per-field)={valid_1step['mean_per_field']:.6f}, "
        f"sqrt(mean MSE / mean Var)={valid_1step['sqrt_mean_mse_over_mean_var']:.6f}, "
        f"sqrt(sum MSE / sum Var)={valid_1step['sqrt_sum_mse_over_sum_var']:.6f}"
    )
    print(f"VALID 1-step per-field={valid_1step['per_field'].tolist()}")

    print(
        f"VALID rollout-4 VRMSE (reconstructed full) mean(per-field)={valid_rollout['mean_per_field']:.6f}, "
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
        if _metrics_csv_needs_header:
            writer.writeheader()
            _metrics_csv_needs_header = False
        writer.writerow(row)

    if valid_1step["mean_per_field"] < best_valid_1step:
        best_valid_1step = valid_1step["mean_per_field"]
        torch.save(model.state_dict(), "best_by_valid_1step_vrmse_delta.pt")
        print("Saved best_by_valid_1step_vrmse_delta.pt")

    if valid_rollout["mean_per_field"] < best_valid_rollout:
        best_valid_rollout = valid_rollout["mean_per_field"]
        torch.save(model.state_dict(), "best_by_valid_rollout_vrmse_delta.pt")
        print("Saved best_by_valid_rollout_vrmse_delta.pt")
torch.save(model.state_dict(), "final_model_delta.pt")