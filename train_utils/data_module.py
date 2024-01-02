import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, data_loader_cfg, batch2device):
        super().__init__()
        self.dataset = dataset
        self.data_loader_cfg = data_loader_cfg
        self.batch2device = batch2device

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return self.batch2device(batch, device, dataloader_idx)

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            collate_fn=self.data_loader_cfg["collate_fn"]["train"],
            batch_sampler=self.data_loader_cfg["batch_sampler"]["train"],
            num_workers=self.data_loader_cfg["num_workers"]["train"],
            worker_init_fn=self.data_loader_cfg["worker_init_fn"]["train"],
            generator=self.data_loader_cfg["generator"]["train"],
            batch_size=self.data_loader_cfg["batch_size"]["train"],
            drop_last=True,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["val"],
            collate_fn=self.data_loader_cfg["collate_fn"]["val"],
            batch_sampler=self.data_loader_cfg["batch_sampler"]["val"],
            num_workers=self.data_loader_cfg["num_workers"]["val"],
            worker_init_fn=self.data_loader_cfg["worker_init_fn"]["val"],
            generator=self.data_loader_cfg["generator"]["val"],
            batch_size=self.data_loader_cfg["batch_size"]["val"],
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            collate_fn=self.data_loader_cfg["collate_fn"]["test"],
            batch_sampler=self.data_loader_cfg["batch_sampler"]["test"],
            num_workers=self.data_loader_cfg["num_workers"]["test"],
            worker_init_fn=self.data_loader_cfg["worker_init_fn"]["test"],
            generator=self.data_loader_cfg["generator"]["test"],
            batch_size=self.data_loader_cfg["batch_size"]["test"],
            drop_last=True,
            pin_memory=True,
        )
