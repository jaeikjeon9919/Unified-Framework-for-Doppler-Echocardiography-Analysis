import os, sys
import json
import argparse
import importlib
import numpy as np
from collections import OrderedDict
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning import callbacks

sys.path.append(os.getcwd())

from train_utils.trainer import DLTrainer
from train_utils.data_module import DataModule
from train_utils.build_model import build_model
from train_utils.transforms import get_transforms
from train_utils.batch2device import transfer_batch_to_device
from utils.config_util import Config
from utils.util import set_random_seed, str2bool, NumpyEncoder
from loggings.save_results import save_results
from loggings.eval_metrics import segmentation_eval_metrics

torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a network for Echo auto segmentation")
    parser.add_argument(
        "--config",
        help="Path to the training config file.",
        default="./configs/unetpp.yaml",
    )
    parser.add_argument("--train_mode", default=True, type=str2bool)
    # args = parser.parse_args()
    args = parser.parse_args(args=[])
    return args


def load_important_class_dict(cfg, task):
    important_class_dict_dir = os.path.join(
        cfg.logging.base_ckpt_dir, task, cfg.logging.version, "important_class_dict.json"
    )
    if os.path.exists(important_class_dict_dir):
        important_class_dict = json.load(open(important_class_dict_dir))
    else:
        important_class_dict_module = importlib.import_module(cfg.data.important_class_dict)
        important_class_dict = important_class_dict_module.important_class_dict
        json.dump(important_class_dict, open(important_class_dict_dir, "w"), cls=NumpyEncoder, indent=4, sort_keys=True)
    return important_class_dict


def configure_callbacks(cfg, ckpt_dir):
    # Set monitoring functions
    if cfg.trainer.monitor_metric == "val_loss":
        callback_mode = "min"
    elif cfg.trainer.monitor_metric == "val_dice_coeff":
        callback_mode = "max"
    else:
        raise ValueError(
            "monitor metric has to be one of val_loss, and val_dice_coeff, but got {}".format(
                cfg.trainer.monitor_metric
            )
        )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=cfg.trainer.monitor_metric, mode=callback_mode, dirpath=ckpt_dir, filename="best_model",
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=cfg.trainer.tolerance, verbose=False, mode="min",
    )

    return [early_stop, checkpoint_callback]


def custom_log(test_performances, key, value):
    if key in test_performances:
        test_performances[key].append(value)
    else:
        test_performances[key] = []
        test_performances[key].append(value)


def save_steplog(test_batch, output, pl_log, test_performances, steplog_params):
    label_seg = test_batch["label_seg"] if "label_seg" in test_batch else test_batch["label"]

    dice_coeff, iou = segmentation_eval_metrics(
        output["prediction"],
        label_seg,
        steplog_params["num_classes"],
        steplog_params["cfg"].data.view,
        test_performances["DSc"],
        test_performances["IoU"],
        steplog_params["important_class_dict"],
    )

    pl_log("test_loss", output["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
    pl_log("test_dice_coeff", dice_coeff, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    pl_log("test_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    pl_log(
        "test_loss_{}".format(test_batch["data_type"][0]),
        output["loss"],
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )
    pl_log(
        "test_dice_coeff_{}".format(test_batch["data_type"][0]),
        dice_coeff,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )
    pl_log(
        "test_iou_{}".format(test_batch["data_type"][0]), iou, on_step=False, on_epoch=True, prog_bar=True, logger=True,
    )
    custom_log(test_performances, "test_loss", output["loss"].cpu().item())
    custom_log(test_performances, "test_mean_dice_coeff", dice_coeff)
    custom_log(test_performances, "test_mean_iou", iou)
    custom_log(test_performances, "test_loss_{}".format(test_batch["data_type"][0]), output["loss"].cpu().item())
    custom_log(test_performances, "test_dice_coeff_{}".format(test_batch["data_type"][0]), dice_coeff)
    custom_log(test_performances, "test_iou_{}".format(test_batch["data_type"][0]), iou)


def save_log(test_performances):
    for key in test_performances:
        if key not in ["DSc", "IoU"]:
            test_performances[key] = np.mean(test_performances[key])
            test_performances[key] = np.mean(test_performances[key])
            test_performances[key] = np.mean(test_performances[key])

    predicted_classes = list(test_performances["DSc"].keys())
    for indv_class in predicted_classes:
        test_performances["DSc"][indv_class] = np.mean(test_performances["DSc"][indv_class])
    for indv_class in predicted_classes:
        test_performances["IoU"][indv_class] = np.mean(test_performances["IoU"][indv_class])

    custom_log(test_performances, "test_dice_coeff_by_class", test_performances["DSc"])
    custom_log(test_performances, "test_iou_by_class", test_performances["IoU"])

    del test_performances["DSc"], test_performances["IoU"]


def train_steplog(train_batch, output, pl_log):
    for k, v in output.items():
        pl_log("train" + k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)


def val_steplog(val_batch, output, pl_log):
    pl_log("val_loss", output["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
    for k, v in output.items():
        pl_log("val" + k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)


def test_steplog(test_batch, output, pl_log):
    for k, v in output.items():
        pl_log("test" + k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)


def run(cfg, args, seed):
    set_random_seed(seed)
    print(cfg.__dict__)

    """logging directory"""
    data_json_dir = cfg.data.data_json_dir
    task, which_data, which_data_idx = (
        data_json_dir.split("/")[-3],
        data_json_dir.split("/")[-2],
        data_json_dir.split("/")[-1],
    )

    ckpt_dir = os.path.join(
        cfg.logging.base_ckpt_dir,
        task,
        cfg.logging.version,
        which_data,
        which_data_idx,
        cfg.data.view,
        cfg.trainer.model_type,
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    """Build dataset"""
    train_json_dir = os.path.join(data_json_dir, cfg.data.view, "train_dict.json")
    val_json_dir = os.path.join(data_json_dir, cfg.data.view, "val_dict.json")
    test_json_dir = os.path.join(data_json_dir, cfg.data.view, "test_dict.json")
    splited_data_dict = (
        json.load(open(train_json_dir)),
        json.load(open(val_json_dir)),
        json.load(open(test_json_dir)),
    )

    dataset = importlib.import_module(cfg.data.type).Dataset

    important_class_dict = load_important_class_dict(cfg, task)
    selected_classes = important_class_dict[cfg.data.view]
    num_classes = len(selected_classes) + 1

    transforms = get_transforms(cfg)

    dataset_dict = {
        mode: dataset(
            {
                "data_dir": cfg.data.data_dir,
                "data_dict": data_dict,
                "selected_classes": selected_classes,
                "transform": transform,
            }
        )
        for mode, data_dict, transform in zip(["train", "val", "test"], splited_data_dict, transforms)
    }

    """Data loader config"""
    data_loader_cfg = {
        "collate_fn": {"train": None, "val": None, "test": None},
        "batch_sampler": {"train": None, "val": None, "test": None},
        "num_workers": {
            "train": cfg.trainer.num_workers,
            "val": cfg.trainer.num_workers,
            "test": cfg.trainer.num_workers,
        },
        "worker_init_fn": {"train": None, "val": None, "test": None},
        "generator": {"train": None, "val": None, "test": None},
        "batch_size": {"train": cfg.trainer.batch_size, "val": cfg.trainer.batch_size, "test": 1},
    }

    """Create a data module to set up dataset and data_loader"""
    data_module = DataModule(dataset_dict, data_loader_cfg, transfer_batch_to_device)

    """Build network"""
    model = build_model(cfg, cfg.trainer.model_type, num_classes).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.trainer.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    cfg_optimizer = {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    cfg_callbacks = configure_callbacks(cfg, ckpt_dir)

    """Create a trainer module to set up trainer"""
    test_performances = {
        "DSc": {indv_class: [] for indv_class in important_class_dict[cfg.data.view]},
        "IoU": {indv_class: [] for indv_class in important_class_dict[cfg.data.view]},
    }

    steplog_params = {"num_classes": num_classes, "important_class_dict": important_class_dict, "cfg": cfg}

    step_result_params = {"cfg": cfg, "ckpt_dir": ckpt_dir, "version_idx": cfg.trainer.model_idx}

    echo_trainer = DLTrainer(
        cfg,
        data_module,
        model,
        cfg_optimizer,
        cfg_callbacks,
        save_steplog=save_steplog,
        test_performances=test_performances,
        steplog_params=steplog_params,
        save_log=save_log,
        step_result_params=step_result_params,
        save_step_result=save_results,
        step_logs={"train": train_steplog, "val": val_steplog, "test": test_steplog},
    )

    tb_logger = loggers.TensorBoardLogger(ckpt_dir, log_graph=False)

    gpu_accelerator = "ddp" if cfg.trainer.gpu_num > 1 else None

    """Build Trainer"""
    trainer = pl.Trainer(
        gpus=cfg.trainer.gpu_num,
        logger=tb_logger,
        max_epochs=cfg.trainer.num_epoch,
        check_val_every_n_epoch=1,
        accelerator=gpu_accelerator,
    )

    """Start training"""
    if args.train_mode:
        if cfg.trainer.pretrained != "None":
            checkpoint = torch.load(cfg.trainer.pretrained)
            echo_trainer.load_state_dict(checkpoint["state_dict"])

        trainer.fit(echo_trainer, datamodule=data_module)

    """Load the best trained model"""
    if cfg.trainer.model_idx == 0:
        best_model_dir = os.path.join(ckpt_dir, "best_model.ckpt")
    else:
        best_model_dir = os.path.join(ckpt_dir, "best_model-v{}.ckpt".format(cfg.trainer.model_idx))

    checkpoint = torch.load(best_model_dir)
    echo_trainer.load_state_dict(checkpoint["state_dict"], strict=False)

    """Test the trained model"""
    trainer.test(echo_trainer, datamodule=data_module)

    log_dict = OrderedDict(cfg.__dict__)
    log_dict["test_performances"] = echo_trainer.test_performances
    json.dump(
        log_dict,
        open(os.path.join(ckpt_dir, "results-v{}.json".format(cfg.trainer.model_idx)), "w"),
        cls=NumpyEncoder,
        indent=4,
        sort_keys=True,
    )


"""
conda activate /home/wer2956/anaconda3/envs/echo_vc
CUDA_VISIBLE_DEVICES=4 python jiyeon/unified_approach_doppler/train.py
"""
if __name__ == "__main__":
    args = parse_args()
    cfg = Config(args.config)
    run(cfg, args, seed=0)
