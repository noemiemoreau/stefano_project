import numpy as np
import torch
from torchvision import transforms
from collections import OrderedDict
from torch import tensor

rng = np.random.default_rng(seed=0)

def list_subdir_filter(source_folder, check_subfolders: bool = False, search_pattern: str = ''):
    """
    Simple wrapper for os.walk() or os.listdir() according to whether one wants to scan subfolders.
    It includes a basic re.search() for filtering purposes.
    :param source_folder: the folder to scan for content
    :param check_subfolders: bool, whether subfolders of source_folder should be checked for files (recursively!)
    :param search_pattern: str,
    :return:
    """
    if check_subfolders:
        all_items = []
        for path, subdirs, files in os.walk(source_folder):
            for filename in files:
                if re.search(search_pattern, filename):
                    all_items.append(os.path.join(path, filename))
    else:
        all_items = [
            os.path.join(source_folder, filename)
            for filename in os.listdir(source_folder)
            if re.search(search_pattern, filename)
        ]
    return sorted(all_items)

def normalize_quantile(x):
    '''
    Clip and normalize an array to its 1% - 99.5%
    '''
    array = x  # .numpy()
    out = np.zeros_like(array)
    for c in range(array.shape[0]):
        channel = array[c]
        mask = channel > 0  # non-background pixels

        if mask.sum() == 0:
            continue  # skip if whole channel is background

        vals = channel[mask]

        # ---- Percentile-based clipping ----
        p_low = np.quantile(vals, 0.01)
        p_high = np.quantile(vals, 0.99)

        clipped = np.clip(channel, min=p_low, max=p_high)

        # ---- Z-score normalization on clipped values ----
        vals_clipped = clipped[mask]
        mean = np.mean(vals_clipped)
        std = np.std(vals_clipped)

        norm = ((clipped - mean) / (std + 1e-6))

        # ---- rescale to [0, 255] ----
        vals_norm = norm[mask]
        norm_min = np.min(vals_norm)
        norm_max = np.max(vals_norm)
        rescaled = (norm - norm_min) / (norm_max - norm_min + 1e-6)  # → [0,1]

        # ---- reserve 0 for background ----
        rescaled[mask] = rescaled[mask] * 254 + 1  # → [1..255]
        rescaled[~mask] = 0  # → background

        out[c] = rescaled

    return out

def get_patches(tensor, tile_size, stride, return_unfold_shape = False):
    dims = tensor.dim()
    tensor_unfold = tensor.unfold(dims-2, size = tile_size, step = stride).unfold(dims-1, size = tile_size, step = stride)
    tensor_patches = tensor_unfold.reshape(*list(tensor.shape)[:-2], -1, tile_size, tile_size)
    if return_unfold_shape:
        return tensor_patches, tensor_unfold.shape
    else:
        return tensor_patches

def calculate_areas(tensor):
    dims = tensor.dim()
    return tensor.sum(dim = dims-2).sum(dim = dims-2)

def get_valid_patches(img_tensor, tile_size, stride, rand_offset = True):
    if rand_offset:
        x_off, y_off = rng.integers(stride), rng.integers(stride)
    else:
        x_off, y_off = 0, 0
    # print(x_off, y_off)
    img_tensor = img_tensor[..., y_off:, x_off:]
    mask_tensor = np.zeros((img_tensor.shape[1], img_tensor.shape[2]))
    mask_tensor[img_tensor[0] > 0] = 1
    img_patches = get_patches(img_tensor, tile_size, stride)
    mask_patches = get_patches(tensor(np.array([mask_tensor])), tile_size, stride)
    mask_patches_areas = calculate_areas(mask_patches)
    area_th = 0.05 * tile_size * tile_size
    valid_mask_indices = mask_patches_areas > area_th
    mask_patches = mask_patches[valid_mask_indices].view(*list(mask_patches.shape)[:-3], -1, tile_size, tile_size)
    valid_img_indices = torch.cat(14 * [valid_mask_indices], dim=0)
    img_patches = img_patches[valid_img_indices].view(*list(img_patches.shape)[:-3], -1, tile_size, tile_size)
    img_patches = img_patches.permute(1, 0, 2, 3)
    return img_patches, valid_mask_indices

def calculate_weights(targets):
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    weight = targets.size()[0] / class_sample_count.double()
    samples_weight = torch.tensor([weight[t] for t in targets])
    return samples_weight

def load_model_without_ddp(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    checkpoint['model'] = new_state_dict
    model.load_state_dict(checkpoint['model'])
    return model
