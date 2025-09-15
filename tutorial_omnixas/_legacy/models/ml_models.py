from functools import cached_property

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge

# multioutput regressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from _legacy.models.trained_model import TrainedModel
from _legacy.data.ml_data import DataQuery


class MedianModel(TrainedModel):
    name = "MedianModel"

    @property
    def model(self):
        return lambda *args, **kwargs: torch.Tensor(self.predictions)

    @cached_property
    def predictions(self):
        return np.median(self.data.train.y, axis=0) * np.ones_like(self.data.test.y)


class MeanModel(TrainedModel):
    name = "MeanModel"

    @property
    def model(self):
        return lambda *args, **kwargs: torch.Tensor(self.predictions)

    @cached_property
    def predictions(self):
        return np.array([np.mean(self.data.train.y, axis=0)] * len(self.data.test.y))


class LinReg(TrainedModel):
    def __init__(self, query: DataQuery, name="LinReg"):
        super().__init__(query)
        self._name = name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        return LinearRegression().fit(self.data.train.X, self.data.train.y)

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)


class RidgeReg(TrainedModel):
    def __init__(self, query: DataQuery, name="RidgeReg"):
        super().__init__(query)
        self._name = name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        return Ridge().fit(self.data.train.X, self.data.train.y)

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)


class SVReg(TrainedModel):
    def __init__(self, query: DataQuery, name="SVReg"):
        super().__init__(query)
        self._name = name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        return MultiOutputRegressor(estimator=SVR()).fit(
            self.data.train.X, self.data.train.y
        )

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)


class RFReg(TrainedModel):
    def __init__(self, query: DataQuery, name="RFReg"):
        super().__init__(query)
        self._name = name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        return RandomForestRegressor().fit(self.data.train.X, self.data.train.y)

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)


class XGBReg(TrainedModel):
    def __init__(self, query: DataQuery, name="XGBReg"):
        super().__init__(query)
        self._name = name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        return XGBRegressor().fit(self.data.train.X, self.data.train.y)

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)


class ElastNet(TrainedModel):
    def __init__(self, query: DataQuery, name="ElasticNet"):
        super().__init__(query)
        self._name = name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        return ElasticNet().fit(self.data.train.X, self.data.train.y)

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)
