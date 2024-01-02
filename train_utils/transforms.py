from torchvision import transforms
from torchvision.transforms import functional as F


def vanilla_transform(resize_h_w):
    return transforms.Compose([Resize(resize_h_w), ToTensor(), Normalization(mean=0.5, std=0.5)])


def get_transforms(cfg):
    if cfg.data.augmentations.aug == "vanilla":
        return [
            vanilla_transform(cfg.data.augmentations.resize_h_w),
            vanilla_transform(cfg.data.augmentations.resize_h_w),
            vanilla_transform(cfg.data.augmentations.resize_h_w),
        ]
    else:
        raise ValueError("Unknown augmentation type: {}".format(cfg.data.augmentations.aug))


class Resize(transforms.Resize):
    def __call__(self, data):
        resized_input = F.resize(data["input"], self.size, interpolation=F.InterpolationMode.BILINEAR)
        data["input"] = resized_input
        if "label" in data:
            resized_label = F.resize(data["label"], self.size, interpolation=F.InterpolationMode.NEAREST)
            data["label"] = resized_label
        return data


class ToTensor(transforms.ToTensor):
    def __call__(self, data):
        data["input"] = F.to_tensor(data["input"])
        if "label" in data:
            data["label"] = F.to_tensor(data["label"])[0, :, :] * 255
        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data["input"] = (data["input"] - self.mean) / self.std
        return data
