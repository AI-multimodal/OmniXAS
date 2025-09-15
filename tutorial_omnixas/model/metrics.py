import numpy as np
from pydantic import BaseModel, Field, computed_field
from sklearn.metrics import mean_squared_error

from omnixas.data import DataTag


class ModelMetrics(BaseModel):
    predictions: np.ndarray
    targets: np.ndarray

    @computed_field
    def mse(self) -> float:
        return mean_squared_error(self.targets, self.predictions)

    @property
    def median_of_mse_per_spectra(self) -> float:
        return np.median(self.mse_per_spectra)

    def _sorted_predictions(self, sort_array=None):
        sort_array = self.mse_per_spectra
        pair = np.column_stack((self.targets, self.predictions))
        pair = pair.reshape(-1, 2, self.targets.shape[1])
        pair = pair[sort_array.argsort()]
        return pair, sort_array.argsort()

    def top_predictions(self, splits=10, drop_last_split=True):
        pair = self._sorted_predictions()[0]
        new_len = len(pair) - divmod(len(pair), splits)[1]
        pair = pair[:new_len]
        top_spectra = [s[-1] for s in np.split(pair, splits)]  # bottom of each split
        if drop_last_split:
            top_spectra = top_spectra[:-1]
        return np.array(top_spectra)

    @property
    def residuals(self):
        return self.targets - self.predictions

    @property
    def mse_per_spectra(self):
        return np.mean(self.residuals**2, axis=1)

    @property
    def deciles(self):
        return self.top_predictions(splits=10, drop_last_split=True)

    class Config:
        arbitrary_types_allowed = True


class ModelTag(DataTag):
    name: str = Field(..., description="Name of the model")


class ComparisonMetrics(BaseModel):
    metric1: ModelMetrics = Field(..., description="First metric to compare")
    metric2: ModelMetrics = Field(..., description="Second metric to compare")

    @property
    def residual_diff(self):
        return np.abs(self.metric1.residuals) - np.abs(self.metric2.residuals)
