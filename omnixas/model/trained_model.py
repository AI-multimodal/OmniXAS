# %%
from loguru import logger
from abc import ABC, abstractmethod
from typing import Any, Optional, Self

import numpy as np
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, computed_field
from torch import nn

from omnixas.data import (
    DataTag,
    IdentityScaler,
    MLSplits,
    ScaledMlSplit,
    ThousandScaler,
)
from omnixas.model.metrics import ModelMetrics, ModelTag
from omnixas.model.training import PlModule
from omnixas.model.xasblock import XASBlock
from omnixas.utils.io import DEFAULTFILEHANDLER, FileHandler
import omnixas
import os

class TrainedModel(BaseModel, ABC):
    tag: ModelTag
    model: Optional[Any] = None
    train_x_scaler: type = ThousandScaler
    train_y_scaler: type = ThousandScaler

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @computed_field
    def metrics(self) -> ModelMetrics:
        return self.compute_metrics(self.default_split)

    @property
    def default_split(self) -> ScaledMlSplit:
        return TrainedModelLoader.load_scaled_splits(
            self.tag,
            self.train_x_scaler,
            self.train_y_scaler,
        )

    def compute_metrics(self, splits: ScaledMlSplit) -> ModelMetrics:
        predictions = self.predict(splits.test.X)
        targets = splits.test.y
        predictions = splits.y_scaler.inverse_transform(predictions)
        targets = splits.y_scaler.inverse_transform(targets)
        return ModelMetrics(targets=targets, predictions=predictions)

    class Config:
        arbitrary_types_allowed = True


class MeanModel(TrainedModel):
    @classmethod
    def from_data_tag(cls, data_tag: DataTag, **kwargs) -> Self:
        model_tag = ModelTag(name="meanModel", **data_tag.dict())
        return cls(tag=model_tag, **kwargs)

    @computed_field
    def train_mean(self) -> np.ndarray:
        return self.default_split.train.y.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.repeat(self.train_mean[None, :], len(X), axis=0)


class TrainedXASBlock(TrainedModel):
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32)
        return self.model(X.to(self.model.device)).detach().cpu().numpy()

    @classmethod
    def load(
        cls,
        tag: ModelTag,
        x_scaler: type = ThousandScaler,
        y_scaler: type = ThousandScaler,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
        **kwargs,
    ):
        model = TrainedModelLoader.load_model(tag, file_handler, **kwargs)
        model.eval()
        model.freeze()
        instance = cls(
            tag=tag,
            model=model,
            train_x_scaler=x_scaler,
            train_y_scaler=y_scaler,
        )
        return instance


class TrainedModelLoader:
    @staticmethod
    def _disable_dropout(model):
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        return model

    @staticmethod
    def _reset_batchnorm(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.reset_running_stats()
        return model

    @staticmethod
    def _freeze_layers(model, freeze_first_k_layers):
        layer_count = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if layer_count < freeze_first_k_layers:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                layer_count += 1
        return model

    @staticmethod
    def load_model_for_finetuning(
        tag: ModelTag,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
        freeze_first_k_layers: int = 0,
        disable_dropout: bool = True,
        reset_batchnorm: bool = False,
    ) -> PlModule:
        model = TrainedModelLoader.load_model(tag, file_handler)
        if disable_dropout:
            model = TrainedModelLoader._disable_dropout(model)
        if reset_batchnorm:
            model = TrainedModelLoader._reset_batchnorm(model)
        model = TrainedModelLoader._freeze_layers(model, freeze_first_k_layers)
        return model

    @staticmethod
    def load_model(
        tag: ModelTag,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
        **kwargs,
    ) -> PlModule:
        ckpt_path = TrainedModelLoader.get_ckpt_path(tag, file_handler)
        logger.info(f"Loading model from {ckpt_path}")
        return PlModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=XASBlock(**TrainedModelLoader.get_layer_widths(tag, **kwargs)),
        )

    @staticmethod
    def get_layer_widths(
        tag: ModelTag,
        hparams: dict = OmegaConf.load(omnixas.__path__[0].replace('omnixas',"config/training/hparams.yaml")).hparams,
        **kwargs,
    ):
        if tag.name != "tunedUniversalXAS":
            hidden_widths = hparams[tag.name][tag.type][tag.element].widths
        else:
            hidden_widths = hparams["universalXAS"]["FEFF"]["All"].widths  # TODO: hacky

        return dict(
            input_dim=kwargs.get("input_dim", 64),
            hidden_dims=hidden_widths,
            output_dim=kwargs.get("output_dim", 141),
        )

    @staticmethod
    def get_ckpt_path(
        tag: ModelTag,
        file_handler=DEFAULTFILEHANDLER(),
    ):
        paths = file_handler.serialized_objects_filepaths(
            "TrainedXASBlock", **tag.dict()
        )
        if len(paths) != 1:
            raise ValueError(f"Expected 1 path, got {len(paths)}")
        return paths[0]

    @staticmethod
    def load_ml_splits(
        tag: ModelTag,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
    ):
        return file_handler.deserialize_json(
            MLSplits,
            DataTag(element=tag.element, type=tag.type, feature=tag.feature),
        )

    @staticmethod
    def load_scaled_splits(
        tag: ModelTag,
        x_scaler: type = IdentityScaler,
        y_scaler: type = ThousandScaler,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
    ):
        splits = TrainedModelLoader.load_ml_splits(tag, file_handler)
        return ScaledMlSplit(
            x_scaler=x_scaler(),
            y_scaler=y_scaler(),
            **splits.dict(),
        )


# %%
