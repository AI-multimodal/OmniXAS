import logging
from lightning import Callback


class TensorboardLogTestTrainLoss(Callback):
    """Logs two scalars in same plot: train_loss and val_loss"""

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        pl_module.logger.experiment.add_scalars(
            "losses", {"train_loss": loss}, trainer.global_step
        )

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     loss = outputs["loss"]
    #     pl_module.logger.experiment.add_scalars(
    #         "losses", {"train_loss": loss}, trainer.global_step
    #     )

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        pl_module.logger.experiment.add_scalars(
            "losses", {"val_loss": val_loss}, trainer.global_step
        )


class SuppressLightningLogs:
    """Context manager to suppress all pytorch lightning logs"""

    class IgnorePLFilter(logging.Filter):
        def filter(self, record):
            return "available:" not in record.getMessage()

    def __init__(self, logger_name="pytorch_lightning.utilities.rank_zero"):
        self.logger_name = logger_name
        self.previous_level = None

    def __enter__(self):
        self.logger = logging.getLogger(self.logger_name)
        self.previous_level = self.logger.level
        self.logger.setLevel(0)
        self.logger.addFilter(SuppressLightningLogs.IgnorePLFilter())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.previous_level)
        self.logger.removeFilter(SuppressLightningLogs.IgnorePLFilter())
