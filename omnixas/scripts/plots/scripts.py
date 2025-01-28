# %%
from typing import Dict
import numpy as np
from omnixas.model.trained_model import ModelMetrics, ModelTag
from functools import partial
from typing import Dict, Tuple

import numpy as np

from omnixas.data import (
    AllDataTags,
    FEFFDataTags,
    FEFFDataTags,
    VASPDataTags,
    ScaledMlSplit,
    ThousandScaler,
)
from omnixas.model.trained_model import (
    ComparisonMetrics,
    MeanModel,
    ModelTag,
    TrainedModelLoader,
    TrainedXASBlock,
)


class XASModelsOfCategory:
    def __new__(cls, model_name: str, **kwargs):
        return {
            ModelTag(name=model_name, **tag.dict()): TrainedXASBlock.load(
                ModelTag(name=model_name, **tag.dict()),
                **kwargs,
            )
            for tag in AllDataTags()
        }


class SplitsOfCategory:
    def __new__(
        cls, data_tags, x_scaler=ThousandScaler, y_scaler=ThousandScaler, **kwargs
    ):
        return {
            tag: ScaledMlSplit.from_splits(
                TrainedModelLoader.load_ml_splits(tag),
                x_scaler=x_scaler,
                y_scaler=y_scaler,
            )
            for tag in data_tags
        }


FEFFMlSplits = partial(SplitsOfCategory, data_tags=FEFFDataTags())
VASPMlSplits = partial(SplitsOfCategory, data_tags=VASPDataTags())


class AllMlSplits:
    def __new__(cls, **kwargs):
        return {**FEFFMlSplits(**kwargs), **VASPMlSplits(**kwargs)}


class AllMlSplits:
    def __new__(cls, x_scaler=ThousandScaler, y_scaler=ThousandScaler, **kwargs):
        return {
            tag: ScaledMlSplit.from_splits(
                TrainedModelLoader.load_ml_splits(tag),
                x_scaler=x_scaler,
                y_scaler=y_scaler,
            )
            for tag in AllDataTags()
        }


class MetricsForModelCategory:
    def __new__(cls, model_name: str, **kwargs):
        return {
            model.tag: model.compute_metrics(splits)
            for splits, model in zip(
                AllMlSplits(**kwargs).values(),
                XASModelsOfCategory(model_name, **kwargs).values(),
            )
        }


AllExpertModels = partial(XASModelsOfCategory, model_name="expertXAS")
AllTunedModels = partial(XASModelsOfCategory, model_name="tunedUniversalXAS")


class AllXASModels:
    def __new__(cls, **kwargs):
        expert_models = AllExpertModels(**kwargs)
        tuned_models = AllTunedModels(**kwargs)
        return {**expert_models, **tuned_models}


AllExpertMetrics = partial(MetricsForModelCategory, model_name="expertXAS")
AllTunedMetrics = partial(MetricsForModelCategory, model_name="tunedUniversalXAS")


class CompareAllExpertAndTuned:
    def __new__(cls, **kwargs):  # kwargs can incldue file_handler, x_scaler, y_scaler
        out = {}
        for m1, m2 in zip(
            AllExpertMetrics(**kwargs).items(),
            AllTunedMetrics(**kwargs).items(),
        ):
            out[m1[0]] = ComparisonMetrics(metric1=m1[1], metric2=m2[1])
        return out


class EtasOfCategory:
    def __new__(cls, model_name: str, **kwargs) -> dict:
        all_models = XASModelsOfCategory(model_name, **kwargs)
        return {
            model_tag: (
                MeanModel(tag=model_tag, **kwargs).metrics.median_of_mse_per_spectra
                / model.metrics.median_of_mse_per_spectra
            )
            for model_tag, model in all_models.items()
        }


ExpertEtas = partial(EtasOfCategory, model_name="expertXAS")
TunedEtas = partial(EtasOfCategory, model_name="tunedUniversalXAS")


class UniversalModelEtas:
    def __new__(cls, **kwargs):
        universal_model = TrainedXASBlock.load(
            tag=ModelTag(
                name="universalXAS",
                element="All",
                type="FEFF",
            ),
            **kwargs,
        )
        return {
            ModelTag(name="universalXAS", **tag.dict()): (
                MeanModel(
                    tag=ModelTag(name="meanmodel", **tag.dict()), **kwargs
                ).metrics.median_of_mse_per_spectra
                / universal_model.compute_metrics(
                    ScaledMlSplit.from_splits(
                        TrainedModelLoader.load_ml_splits(tag),
                        x_scaler=kwargs.get("x_scaler", ThousandScaler),
                        y_scaler=kwargs.get("y_scaler", ThousandScaler),
                    )
                ).median_of_mse_per_spectra
            )
            for tag in FEFFDataTags()
        }


class AllEtas:
    def __new__(cls, **kwargs):
        return {
            **ExpertEtas(**kwargs),
            **TunedEtas(**kwargs),
            **UniversalModelEtas(**kwargs),
        }


class ExpertTunedWinRates:
    def __new__(cls, **kwargs) -> Dict[ModelTag, Tuple[float, float]]:
        comparison_metrics = CompareAllExpertAndTuned(**kwargs)
        win_rates = {}
        for tag, metrics in comparison_metrics.items():
            differences = metrics.residual_diff
            energy_rate = float(np.mean(np.mean(differences, axis=0) > 0) * 100)
            spectra_rate = float(np.mean(np.mean(differences, axis=1) > 0) * 100)
            win_rates[tag] = dict(energy=energy_rate, spectra=spectra_rate)
        return win_rates


class AllUniversalMetrics:
    def __new__(cls, **kwargs) -> Dict[ModelTag, ModelMetrics]:
        universal_model = TrainedXASBlock.load(
            tag=ModelTag(
                name="universalXAS",
                element="All",
                type="FEFF",
            ),
            **kwargs,
        )
        metrics = {}
        for tag in FEFFDataTags():
            model_tag = ModelTag(name="universalXAS", element=tag.element, type="FEFF")
            splits = ScaledMlSplit.from_splits(
                TrainedModelLoader.load_ml_splits(tag),
                x_scaler=kwargs.get("x_scaler", ThousandScaler),
                y_scaler=kwargs.get("y_scaler", ThousandScaler),
            )
            metrics[model_tag] = universal_model.compute_metrics(splits)
        return metrics


# %%

if __name__ == "__main__":
    from omnixas.utils import DEFAULTFILEHANDLER

    etas = AllEtas(
        x_scaler=ThousandScaler,
        y_scaler=ThousandScaler,
        file_handler=DEFAULTFILEHANDLER(),
    )
    print(etas)
# %%
