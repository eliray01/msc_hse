import torch
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from neuralop.models import FNO as NeuralFNO
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_my_model(checkpoint_path, n_fields, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    else:
        state_dict = None
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "state_dict", "model", "net"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break
            if state_dict is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state_dict = ckpt

        if state_dict is None:
            raise ValueError(
                "Unsupported checkpoint format. Expected nn.Module or state_dict-like dict."
            )

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        model = NeuralFNO(
            n_modes=(16, 16),
            in_channels=4 * n_fields,
            out_channels=1 * n_fields,
            hidden_channels=128,
            n_layers=4,
        )
        model.load_state_dict(state_dict, strict=True)

    return model.to(device).eval()


base_path = "./data"
checkpoint_path = "./my_model/best_by_valid_rollout_vrmse.pt"
device = pick_device()
print(f"Using device: {device}")

dset = WellDataset(
    well_base_path=f"{base_path}/datasets",
    well_dataset_name="turbulent_radiative_layer_2D",
    well_split_name="test",
    n_steps_input=4,
    n_steps_output=1,
    use_normalization=True,
    normalization_type=ZScoreNormalization,
)
loader = DataLoader(dset, batch_size=8, shuffle=False)
F = dset.metadata.n_fields
model = load_my_model(checkpoint_path, n_fields=F, device=device)

sum_err2 = torch.zeros(F, device=device)
sum_y = torch.zeros(F, device=device)
sum_y2 = torch.zeros(F, device=device)
count = 0

with torch.no_grad():
    for batch in tqdm(loader, desc="test 1-step"):
        x = batch["input_fields"].to(device)
        y = batch["output_fields"].to(device)
        # model expects normalized inputs already because dset uses normalization
        x_in = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
        pred = model(x_in)
        pred_full_norm = rearrange(pred, "B (To F) Lx Ly -> B To Lx Ly F", To=1, F=F)

        err2 = (pred_full_norm - y) ** 2  # (B,1,Lx,Ly,F)

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

# Variant: compute MSE and Var after averaging over fields first
mse_mean = mse_f.mean()
Var_mean = Var_f.mean()
vrmse_fields_averaged_then_sqrt = torch.sqrt(mse_mean / (Var_mean + 1e-7))

# Variant: flatten fields too (treat each field equally by summing and dividing by F)
vrmse_flattened = torch.sqrt(mse_f.sum() / (Var_f.sum() + 1e-7))

print("per-field VRMSE:", vrmse_per_field.tolist())
print("mean(per-field VRMSE):", vrmse_mean_fields.item())
print("sqrt(mean MSE / mean Var):", vrmse_fields_averaged_then_sqrt.item())
print("sqrt(sum MSE / sum Var):", vrmse_flattened.item())