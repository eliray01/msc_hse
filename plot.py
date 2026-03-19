import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization

base_path = "./data"

dset = WellDataset(
    well_base_path=f'{base_path}/datasets',
    well_dataset_name='turbulent_radiative_layer_2D',
    well_split_name='test',
    n_steps_input=1,
    n_steps_output=1,
    use_normalization=False,
)

def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def _extract_xy(sample):
    """
    Tries to extract input/output tensors from a WellDataset sample.
    Supports both dict and tuple-style outputs.
    """
    if isinstance(sample, dict):
        possible_x_keys = ["x", "input_fields", "inputs"]
        possible_y_keys = ["y", "output_fields", "targets"]

        x = None
        y = None

        for k in possible_x_keys:
            if k in sample:
                x = sample[k]
                break

        for k in possible_y_keys:
            if k in sample:
                y = sample[k]
                break

        if x is None or y is None:
            raise KeyError(f"Could not find x/y in sample keys: {list(sample.keys())}")

        return x, y

    elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[0], sample[1]

    else:
        raise ValueError("Unsupported dataset sample format.")

def save_full_dataset_gif(
    dset,
    save_path="full_dataset.gif",
    field_names=None,
    cmap="RdBu_r",
    fps=4,
    max_samples=None,
):
    n_samples = len(dset) if max_samples is None else min(len(dset), max_samples)

    frames = []

    x0, y0 = _extract_xy(dset[0])
    x0 = _to_numpy(x0)
    y0 = _to_numpy(y0)

    if x0.ndim != 4 or y0.ndim != 4:
        raise ValueError(f"Expected x,y with shape (T,Lx,Ly,F), got {x0.shape}, {y0.shape}")

    if x0.shape[0] != 1 or y0.shape[0] != 1:
        raise ValueError(
            f"This function expects n_steps_input=1 and n_steps_output=1, "
            f"but got x.shape={x0.shape}, y.shape={y0.shape}"
        )

    frames.append(x0[0])
    frames.append(y0[0])

    for i in range(1, n_samples):
        x, y = _extract_xy(dset[i])
        x = _to_numpy(x)
        y = _to_numpy(y)

        if x.shape[0] != 1 or y.shape[0] != 1:
            raise ValueError(
                f"Sample {i} does not have single-step input/output: "
                f"x.shape={x.shape}, y.shape={y.shape}"
            )

        frames.append(y[0])

    seq = np.stack(frames, axis=0)   # (T, Lx, Ly, F)
    T, Lx, Ly, F = seq.shape

    if field_names is None:
        if hasattr(dset, "metadata") and hasattr(dset.metadata, "field_names"):
            field_names = [
                name
                for group in dset.metadata.field_names.values()
                for name in group
            ]
        else:
            field_names = []

    if len(field_names) < F:
        field_names = field_names + [f"field_{i}" for i in range(len(field_names), F)]
    elif len(field_names) > F:
        field_names = field_names[:F]

    fig, axs = plt.subplots(F, 1, figsize=(4.5, 2.8 * F))
    if F == 1:
        axs = [axs]

    images = []

    for f in range(F):
        vmin = seq[..., f].min()
        vmax = seq[..., f].max()

        im = axs[f].imshow(
            seq[0, :, :, f],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
        )
        axs[f].set_ylabel(field_names[f])
        axs[f].set_xticks([])
        axs[f].set_yticks([])
        if f == 0:
            axs[f].set_title("Full dataset rollout")
        images.append(im)

    def update(t):
        fig.suptitle(f"Timestep {t+1}/{T}")
        artists = []
        for f in range(F):
            images[f].set_data(seq[t, :, :, f])
            artists.append(images[f])
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=1000 // fps,
        blit=False,
    )

    plt.tight_layout()
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved GIF to: {save_path}")


field_names = [
    name for group in dset.metadata.field_names.values() for name in group
]

save_full_dataset_gif(
    dset,
    save_path="turbulent_full_dataset.gif",
    field_names=field_names,
    fps=5,
)