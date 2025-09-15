from pytorch_lightning import Callback


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