import pytorch_lightning as pl


class DLTrainer(pl.LightningModule):
    def __init__(
        self,
        cfg,
        data_module,
        model,
        cfg_optimizer,
        cfg_callbacks,
        step_logs=None,
        save_log=None,
        save_steplog=None,
        steplog_params=None,
        save_step_result=None,
        step_result_params=None,
        test_performances=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.data_module = data_module
        self.model = model
        self.cfg_optimizer = cfg_optimizer
        self.cfg_callbacks = cfg_callbacks
        self.step_logs = step_logs
        self.save_step_result = save_step_result
        self.save_steplog = save_steplog
        self.save_log = save_log

        self.test_performances = test_performances if test_performances is not None else {}
        self.steplog_params = steplog_params if steplog_params is not None else {}
        self.step_result_params = step_result_params if step_result_params is not None else {}

    def training_step(self, train_batch, batch_idx):
        output = self.model(train_batch)
        if self.step_logs is not None and self.step_logs.get("train", None) is not None:
            self.step_logs["train"](train_batch, output, self.log)
        return output["loss"]

    def validation_step(self, val_batch, batch_idx):
        output = self.model.inference_forward(val_batch)
        if self.step_logs is not None and self.step_logs.get("val", None) is not None:
            self.step_logs["val"](val_batch, output, self.log)

    def test_step(self, test_batch, batch_idx):
        output = self.model.inference_forward(test_batch)
        if self.step_logs is not None and self.step_logs.get("test", None) is not None:
            self.step_logs["test"](test_batch, output, self.log)
        if self.save_step_result is not None:
            self.save_step_result(test_batch, output, self.step_result_params)
        if self.save_steplog is not None:
            self.save_steplog(
                test_batch, output, self.log, self.test_performances, self.steplog_params,
            )

    def test_epoch_end(self, test_step_outputs) -> None:
        if (self.test_performances != {}) and (self.save_log is not None):
            self.save_log(self.test_performances)

    def configure_optimizers(self):
        if self.cfg_optimizer.get("lr_scheduler", None) is not None:
            return [self.cfg_optimizer["optimizer"]], [self.cfg_optimizer["lr_scheduler"]]
        else:
            return [self.cfg_optimizer["optimizer"]]

    def configure_callbacks(self):
        return self.cfg_callbacks

    # PERFORMANCE TUNING
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
