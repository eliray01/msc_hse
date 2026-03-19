import torch
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from the_well.benchmark.models import FNO
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization

# -------------------------
# config
# -------------------------
base_path = "./data"
device = torch.device("mps")
T_ROLLOUT = 10       # number of future timesteps to predict autoregressively
BATCH_SIZE = 8

# -------------------------
# model
# -------------------------
model = FNO.from_pretrained(
    "polymathic-ai/FNO-turbulent_radiative_layer_2D"
).to(device).eval()

# -------------------------
# dataset
# IMPORTANT: ask dataset for T_ROLLOUT target frames
# -------------------------
dset = WellDataset(
    well_base_path=f"{base_path}/datasets",
    well_dataset_name="turbulent_radiative_layer_2D",
    well_split_name="test",
    n_steps_input=4,
    n_steps_output=T_ROLLOUT,
    use_normalization=True,
    normalization_type=ZScoreNormalization,
)

loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False)

F = dset.metadata.n_fields
norm = dset.norm

mean_full = norm.flattened_means["variable"].to(device)         # (F,)
std_full  = norm.flattened_stds["variable"].to(device)          # (F,)
mean_d    = norm.flattened_means_delta["variable"].to(device)   # (F,)
std_d     = norm.flattened_stds_delta["variable"].to(device)    # (F,)

# -------------------------
# accumulators over entire rollout horizon
# -------------------------
sum_err2 = torch.zeros(F, device=device)
sum_y = torch.zeros(F, device=device)
sum_y2 = torch.zeros(F, device=device)
count = 0

# optional: track error at each rollout step separately
sum_err2_per_step = torch.zeros(T_ROLLOUT, F, device=device)
sum_y_per_step = torch.zeros(T_ROLLOUT, F, device=device)
sum_y2_per_step = torch.zeros(T_ROLLOUT, F, device=device)
count_per_step = torch.zeros(T_ROLLOUT, device=device)

with torch.no_grad():
    for batch in tqdm(loader, desc="rollout eval"):
        # x: (B, Ti=4, Lx, Ly, F), normalized full states
        # y: (B, To=T_ROLLOUT, Lx, Ly, F), normalized full states
        x = batch["input_fields"].to(device)
        y = batch["output_fields"].to(device)

        B, Ti, Lx, Ly, F_ = x.shape
        assert F_ == F

        # current autoregressive window of normalized FULL states
        x_roll = x.clone()

        preds = []

        for step in range(T_ROLLOUT):
            # model input: merge time and field channels
            x_in = rearrange(x_roll, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

            # model output: normalized DELTA for one next step
            pred_delta_norm = model(x_in)
            pred_delta_norm = rearrange(
                pred_delta_norm,
                "B (To F) Lx Ly -> B To Lx Ly F",
                To=1,
                F=F,
            )

            pred_delta_norm = pred_delta_norm[:, 0]   # (B, Lx, Ly, F)

            # last frame of current window = normalized FULL state
            last_full_norm = x_roll[:, -1]            # (B, Lx, Ly, F)

            # convert to raw space
            last_raw = last_full_norm * std_full + mean_full
            delta_raw = pred_delta_norm * std_d + mean_d

            # next predicted FULL frame in raw space
            pred_full_raw = last_raw + delta_raw

            # back to normalized FULL state
            pred_full_norm = (pred_full_raw - mean_full) / std_full  # (B, Lx, Ly, F)

            preds.append(pred_full_norm)

            # autoregressive update:
            # drop oldest frame, append predicted frame
            x_roll = torch.cat([x_roll[:, 1:], pred_full_norm.unsqueeze(1)], dim=1)

        # stack rollout predictions
        # pred_seq: (B, T_ROLLOUT, Lx, Ly, F)
        pred_seq = torch.stack(preds, dim=1)

        # squared error over whole rollout
        err2 = (pred_seq - y) ** 2   # (B, T, Lx, Ly, F)

        # accumulate global stats across all rollout steps together
        sum_err2 += err2.sum(dim=(0, 1, 2, 3))
        sum_y += y.sum(dim=(0, 1, 2, 3))
        sum_y2 += (y * y).sum(dim=(0, 1, 2, 3))
        count += y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3]

        # accumulate per-step stats too
        for step in range(T_ROLLOUT):
            err2_step = err2[:, step]      # (B, Lx, Ly, F)
            y_step = y[:, step]            # (B, Lx, Ly, F)

            sum_err2_per_step[step] += err2_step.sum(dim=(0, 1, 2))
            sum_y_per_step[step] += y_step.sum(dim=(0, 1, 2))
            sum_y2_per_step[step] += (y_step * y_step).sum(dim=(0, 1, 2))
            count_per_step[step] += y_step.shape[0] * y_step.shape[1] * y_step.shape[2]

# -------------------------
# global rollout VRMSE across all predicted steps
# -------------------------
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

print(f"=== Rollout horizon: {T_ROLLOUT} ===")
print("per-field VRMSE:", vrmse_per_field.tolist())
print("mean(per-field VRMSE):", vrmse_mean_fields.item())
print("sqrt(mean MSE / mean Var):", vrmse_fields_averaged_then_sqrt.item())
print("sqrt(sum MSE / sum Var):", vrmse_flattened.item())

# -------------------------
# per-step rollout VRMSE
# useful to see error growth with horizon
# -------------------------
print("\n=== Per-step rollout VRMSE ===")
for step in range(T_ROLLOUT):
    mse_f_step = sum_err2_per_step[step] / count_per_step[step]
    Ey_f_step = sum_y_per_step[step] / count_per_step[step]
    Ey2_f_step = sum_y2_per_step[step] / count_per_step[step]
    Var_f_step = Ey2_f_step - Ey_f_step * Ey_f_step

    vrmse_step_per_field = torch.sqrt(mse_f_step / (Var_f_step + 1e-7))
    vrmse_step_mean = vrmse_step_per_field.mean()

    print(
        f"step {step+1:02d}: "
        f"mean VRMSE = {vrmse_step_mean.item():.6f}, "
        f"per-field = {vrmse_step_per_field.tolist()}"
    )