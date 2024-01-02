import os
import json
import random
import argparse
import numpy as np
import typing as T
import matplotlib.pyplot as plt
from omegaconf import DictConfig, ListConfig
import torch


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        elif isinstance(obj, DictConfig):
            return dict(obj)
        elif isinstance(obj, ListConfig):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def mkdirs(dirs: T.List) -> None:
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def save_png(save_dir, image):
    plt.imsave(save_dir, image)


def save_npy(save_dir, file):
    with open(save_dir, "wb") as f:
        np.save(f, file)


def Denormalization(image, mean=0.5, std=0.5):
    return (image * std) + mean
