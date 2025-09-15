# %%
import warnings
from typing import Any, Self

from pydantic import Field, model_validator
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from omnixas.data import MLData, MLSplits


class UncenteredRobustScaler(RobustScaler):
    """A RobustScaler variant that doesn't center the data.

    This scaler is specifically designed for spectra prediction models that
    produce only positive values (e.g., XASBlock). It inherits from sklearn's
    RobustScaler but forces `with_centering=False` to maintain positivity.

    Examples:
        >>> scaler = UncenteredRobustScaler()
        >>> X = np.array([[1, 2, 3], [4, 5, 6]])
        >>> X_scaled = scaler.fit_transform(X)
        >>> # All values remain positive
        >>> assert (X_scaled >= 0).all()
    """

    def __init__(self):
        super().__init__(with_centering=False)


class IdentityScaler(FunctionTransformer):
    """A pass-through scaler that returns data unchanged.

    This scaler is useful for maintaining code consistency when scaling is
    not desired but a scaler interface is required.

    Examples:
        >>> scaler = IdentityScaler()
        >>> X = np.array([[1, 2], [3, 4]])
        >>> np.allclose(X, scaler.fit_transform(X))
        True
    """

    def __init__(self):
        super().__init__(
            func=lambda X: X,
            inverse_func=lambda X: X,
        )


class MultiplicativeScaler(FunctionTransformer):
    """A scaler that multiplies and divides by a constant factor.

    Args:
        factor (float): The multiplication factor for scaling

    Examples:
        >>> scaler = MultiplicativeScaler(factor=10)
        >>> X = np.array([[1, 2], [3, 4]])
        >>> X_scaled = scaler.fit_transform(X)
        >>> # All values are multiplied by 10
        >>> np.allclose(X_scaled, X * 10)
        True
    """

    def __init__(self, factor: float):
        self.factor = factor
        super().__init__(
            func=lambda X: X * self.factor,
            inverse_func=lambda X: X / self.factor,
        )


class ThousandScaler(FunctionTransformer):
    """A scaler that multiplies data by 1000, as used in arxiv_v1 manuscript.

    This scaler is equivalent to MultiplicativeScaler(factor=1000) but is
    provided as a separate class for clarity and direct use in configurations.

    Examples:
        >>> scaler = ThousandScaler()
        >>> X = np.array([[0.001, 0.002], [0.003, 0.004]])
        >>> X_scaled = scaler.fit_transform(X)
        >>> np.allclose(X_scaled, X * 1000)
        True
    """

    def __init__(self):
        super().__init__(
            func=lambda X: X * 1000,
            inverse_func=lambda X: X / 1000,
        )


class ScaledMlSplit(MLSplits):
    """A class for handling scaled machine learning data splits.

    This class extends MLSplits to provide scaling functionality for both
    features (X) and targets (y). It supports different scalers for X and y,
    and ensures consistent scaling across train/val/test splits.

    Attributes:
        x_scaler (Any): Scaler for features. Defaults to IdentityScaler.
        y_scaler (Any): Scaler for targets. Defaults to IdentityScaler.

    Examples:
        >>> # Create scaled splits with RobustScaler
        >>> splits = MLSplits(train=train_data, val=val_data, test=test_data)
        >>> scaled_splits = ScaledMlSplit.from_splits(
        ...     splits,
        ...     x_scaler=UncenteredRobustScaler,
        ...     y_scaler=ThousandScaler
        ... )
        >>>
        >>> # Access scaled data
        >>> X_scaled = scaled_splits.train.X
        >>> y_scaled = scaled_splits.train.y
        >>>
        >>> # Inverse transform predictions
        >>> predictions = MLData(X=X_pred, y=y_pred)
        >>> original_scale = scaled_splits.inverse_transform(predictions)
    """

    x_scaler: Any = Field(default_factory=ThousandScaler)
    y_scaler: Any = Field(default_factory=ThousandScaler)

    @model_validator(mode="after")
    def fit_transform(self):
        """Validates and transforms the data after model initialization.

        Raises:
            ValueError: If train data is not provided.
        """
        if not self.train:
            raise ValueError("train data is required")
        self.fit(self.train)._self_transform()
        return self

    def fit(self, split: MLData) -> Self:
        """Fits the scalers to the provided data split.

        Args:
            split (MLData): Data to fit the scalers on, typically training data

        Returns:
            Self: The instance with fitted scalers
        """
        self.x_scaler.fit(split.X)
        self.y_scaler.fit(split.y)
        return self

    def _self_transform(self) -> Self:
        self.train = self.transform(self.train)
        self.test = self.transform(self.test)
        self.val = self.transform(self.val)
        return self

    def transform(self, data: MLData) -> MLData:
        """Transforms a single data split using the fitted scalers.

        Args:
            data (MLData): Data to transform

        Returns:
            MLData: Transformed data

        Warns:
            UserWarning: If scaled y values contain negatives when using
                models that only produce positive values
        """
        split = MLData(
            X=self.x_scaler.transform(data.X),
            y=self.y_scaler.transform(data.y),
        )
        if (split.y < 0).any():
            msg = "y values are negative after scaling. "
            msg += "Avoid using models that only gives positive values "
            msg += "e.g: XASBlock with softplus activation"
            warnings.warn(msg)
        return split

    def inverse_transform(self, data: MLData) -> MLData:
        """Inverse transforms scaled data back to original scale.

        Args:
            data (MLData): Scaled data to inverse transform

        Returns:
            MLData: Data in original scale
        """
        return MLData(
            X=self.x_scaler.inverse_transform(data.X),
            y=self.y_scaler.inverse_transform(data.y),
        )

    @classmethod
    def from_splits(
        cls,
        splits: MLSplits,
        x_scaler: Any = ThousandScaler,
        y_scaler: Any = ThousandScaler,
    ):
        """Creates a ScaledMlSplit instance from existing splits.

        Args:
            splits (MLSplits): Original unscaled splits
            x_scaler (Any, optional): Scaler class for features.
                Defaults to ThousandScaler.
            y_scaler (Any, optional): Scaler class for targets.
                Defaults to ThousandScaler.

        Returns:
            ScaledMlSplit: New instance with scaled data

        Examples:
            >>> # Scale with different scalers for X and y
            >>> scaled = ScaledMlSplit.from_splits(
            ...     splits,
            ...     x_scaler=UncenteredRobustScaler,
            ...     y_scaler=ThousandScaler
            ... )
        """
        return cls(
            train=splits.train,
            test=splits.test,
            val=splits.val,
            x_scaler=x_scaler(),
            y_scaler=y_scaler(),
        )
