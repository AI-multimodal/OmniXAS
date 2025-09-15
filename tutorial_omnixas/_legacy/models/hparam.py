import logging
import sys

import hydra
import optuna
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback

from omnixas.utils.callbacks import TensorboardLogTestTrainLoss


class Optimizer:
    def __init__(self, hydra_configs):
        self.cfg = hydra_configs

    def optimize(self, trial):
        depth = trial.suggest_int(
            "depth",
            self.cfg.optuna.params.min_depth,
            self.cfg.optuna.params.max_depth,
        )

        hidden_widths = [
            trial.suggest_int(
                f"width_{i}",
                self.cfg.optuna.params.min_width,
                self.cfg.optuna.params.max_width,
                self.cfg.optuna.params.step_width,
            )
            for i in range(depth)
        ]

        batch_size = trial.suggest_int(
            "batch_size",
            self.cfg.optuna.params.min_batch_size,
            self.cfg.optuna.params.max_batch_size,
            self.cfg.optuna.params.step_batch_size,
        )
        batch_size = 2**batch_size
        self.cfg["data_module"]["batch_size"] = batch_size

        data_module = instantiate(self.cfg.data_module)
        data_sample = data_module.train_dataloader().dataset[0]
        input_width = data_sample[0].shape[0]
        output_width = data_sample[1].shape[0]

        widths = [input_width] + hidden_widths + [output_width]
        self.cfg["model"]["model"]["widths"] = widths

        pl_model = instantiate(self.cfg.model)

        trainer = instantiate(self.cfg.trainer)
        trainer.callbacks.extend(
            [
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                TensorboardLogTestTrainLoss(),
            ]
        )
        trainer.fit(pl_model, data_module)

        return trainer.callback_metrics["val_loss"]


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def run_optmization(cfg: DictConfig):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        storage=cfg.optuna.storage,
        load_if_exists=True,
    )
    optimizer_class = Optimizer(hydra_configs=cfg)
    study.optimize(
        optimizer_class.optimize,
        n_trials=cfg.optuna.n_trials,
        n_jobs=cfg.optuna.n_jobs,
    )
    logger.info("Best params: ", study.best_params)
    logger.info("Best value: ", study.best_value)
    logger.info("Best Trial: ", study.best_trial)


if __name__ == "__main__":
    run_optmization()
