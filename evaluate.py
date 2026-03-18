import torch
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from the_well.benchmark.models import FNO
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.benchmark.metrics import VRMSE

base_path='./data'
device=torch.device('mps')
model = FNO.from_pretrained('polymathic-ai/FNO-turbulent_radiative_layer_2D').to(device).eval()

dset = WellDataset(
    well_base_path=f'{base_path}/datasets',
    well_dataset_name='turbulent_radiative_layer_2D',
    well_split_name='test',
    n_steps_input=4,
    n_steps_output=1,
    use_normalization=True,
    normalization_type=ZScoreNormalization,
)
loader = DataLoader(dset, batch_size=8, shuffle=False)
F = dset.metadata.n_fields
norm = dset.norm
mean_full = norm.flattened_means['variable'].to(device)       # (F,)
std_full  = norm.flattened_stds['variable'].to(device)
mean_d    = norm.flattened_means_delta['variable'].to(device)
std_d     = norm.flattened_stds_delta['variable'].to(device)

sum_err2 = torch.zeros(F, device=device)
sum_y = torch.zeros(F, device=device)
sum_y2 = torch.zeros(F, device=device)
count = 0

with torch.no_grad():
    for batch in tqdm(loader, desc='stream accum'):
        x = batch['input_fields'].to(device)
        y = batch['output_fields'].to(device)
        # model expects normalized inputs already because dset uses normalization
        x_in = rearrange(x, 'B Ti Lx Ly F -> B (Ti F) Lx Ly')
        pred = model(x_in)
        pred = rearrange(pred, 'B (To F) Lx Ly -> B To Lx Ly F', To=1, F=F)

        last_input_full_norm = x[:, -1]      # (B,Lx,Ly,F) normalized full
        pred_delta_norm = pred[:, 0]         # (B,Lx,Ly,F) delta normalized

        # delta -> full normalized inversion
        last_raw = last_input_full_norm * std_full + mean_full
        delta_raw = pred_delta_norm * std_d + mean_d
        full_raw = last_raw + delta_raw
        pred_full_norm = (full_raw - mean_full) / std_full
        pred_full_norm = pred_full_norm.unsqueeze(1)  # (B,To=1,Lx,Ly,F)

        err2 = (pred_full_norm - y) ** 2  # (B,1,Lx,Ly,F)

        sum_err2 += err2.sum(dim=(0,1,2,3))
        sum_y += y.sum(dim=(0,1,2,3))
        sum_y2 += (y*y).sum(dim=(0,1,2,3))
        count += y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3]

mse_f = sum_err2 / count
Ey_f = sum_y / count
Ey2_f = sum_y2 / count
Var_f = Ey2_f - Ey_f*Ey_f

vrmse_per_field = torch.sqrt(mse_f / (Var_f + 1e-7))
vrmse_mean_fields = vrmse_per_field.mean()

# Variant: compute MSE and Var after averaging over fields first
mse_mean = mse_f.mean()
Var_mean = Var_f.mean()
vrmse_fields_averaged_then_sqrt = torch.sqrt(mse_mean / (Var_mean + 1e-7))

# Variant: flatten fields too (treat each field equally by summing and dividing by F)
vrmse_flattened = torch.sqrt((mse_f.sum()) / (Var_f.sum() + 1e-7))

print('per-field VRMSE:', vrmse_per_field.tolist())
print('mean(per-field VRMSE):', vrmse_mean_fields.item())
print('sqrt(mean MSE / mean Var):', vrmse_fields_averaged_then_sqrt.item())
print('sqrt(sum MSE / sum Var):', vrmse_flattened.item())