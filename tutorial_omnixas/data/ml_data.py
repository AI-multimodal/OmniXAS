from typing import Optional

import numpy as np
from pydantic import BaseModel, field_validator, model_serializer

from omnixas.core import Element, SpectrumType
from omnixas.utils.readable_enums import ReadableEnums


class MLData(BaseModel):
    """A container for machine learning data containing features (X) and labels (y).

    This class provides a structured way to handle machine learning datasets with
    features and labels, offering functionality for data manipulation and validation.

    Attributes:
        X (Optional[np.ndarray]): Feature matrix of shape (n_samples, n_features)
        y (Optional[np.ndarray]): Label array of shape (n_samples,)

    Examples:
        >>> # Create an MLData instance with numpy arrays
        >>> X = np.array([[1, 2], [3, 4]])
        >>> y = np.array([0, 1])
        >>> data = MLData(X=X, y=y)

        >>> # Create from lists (automatic conversion to numpy arrays)
        >>> data = MLData(X=[[1, 2], [3, 4]], y=[0, 1])

        >>> # Get a shuffled view of the data
        >>> shuffled_data = data.shuffled_view(seed=42)

        >>> # Slice the data
        >>> subset = data[0:1]
    """

    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    def shuffled_view(self, seed: Optional[int] = 42) -> "MLData":
        """Creates a shuffled view of the dataset with consistent indexing.

        Args:
            seed (Optional[int]): Random seed for reproducibility. Defaults to 42.

        Returns:
            MLData: A new MLData instance with shuffled data.

        Raises:
            ValueError: If either X or y is None.

        Examples:
            >>> data = MLData(X=np.array([[1, 2], [3, 4]]), y=np.array([0, 1]))
            >>> shuffled = data.shuffled_view(seed=42)
        """
        if self.X is None or self.y is None:
            raise ValueError("X and y must be set before shuffling.")
        np.random.seed(seed)
        indices = np.random.permutation(len(self.X))
        return MLData(X=self.X[indices], y=self.y[indices])

    def __getitem__(self, idx):
        return MLData(X=self.X[idx], y=self.y[idx])

    @field_validator("X", "y", mode="before")
    @classmethod
    def _to_numpy(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

    @model_serializer
    def _serialize(self):
        return {
            "X": self.X.tolist(),
            "y": self.y.tolist(),
        }

    def __len__(self):
        len_X = len(self.X) if self.X is not None else 0
        len_y = len(self.y) if self.y is not None else 0
        if len_X != len_y:
            raise ValueError(
                f"X and y must have the same length. Got {len_X} and {len_y}."
            )
        return len_X

    def __eq__(self, other):
        return np.allclose(self.X, other.X) and np.allclose(self.y, other.y)

    class Config:
        arbitrary_types_allowed = True


@ReadableEnums()
class DataTag(BaseModel):
    """A tag for identifying and categorizing spectral data.

    This class provides a way to tag spectral data with element, type, and feature
    information, making it easier to organize and retrieve specific datasets.

    Attributes:
        element (Element): The chemical element associated with the data
        type (SpectrumType): The type of spectrum
        feature (Optional[str]): The feature extraction method. Defaults to "m3gnet"

    Examples:
        >>> from omnixas.utils.constants import Element, SpectrumType
        >>> tag = DataTag(
        ...     element=Element.Fe,
        ...     type=SpectrumType.XANES,
        ...     feature="m3gnet"
        ... )
        >>> hash(tag)  # Can be used as dictionary key
    """

    element: Element
    type: SpectrumType = None
    feature: Optional[str] = "m3gnet"

    def __hash__(self) -> int:  # store as dict key
        return hash((self.element, self.type))


class MLSplits(BaseModel):
    """Container for train/validation/test splits of machine learning data.

    This class manages the standard splits of a machine learning dataset,
    providing functionality to manipulate all splits consistently.

    Attributes:
        train (Optional[MLData]): Training data
        val (Optional[MLData]): Validation data
        test (Optional[MLData]): Test data

    Examples:
        >>> # Create splits with numpy arrays
        >>> splits = MLSplits(
        ...     train=MLData(X=np.array([[1, 2]]), y=np.array([0])),
        ...     val=MLData(X=np.array([[3, 4]]), y=np.array([1])),
        ...     test=MLData(X=np.array([[5, 6]]), y=np.array([2]))
        ... )

        >>> # Get shuffled view of all splits
        >>> shuffled_splits = splits.shuffled_view(seed=42)

        >>> # Get total number of samples across all splits
        >>> total_samples = len(splits)
    """

    train: Optional[MLData] = None
    val: Optional[MLData] = None
    test: Optional[MLData] = None

    def shuffled_view(self, seed: Optional[int] = 42) -> "MLSplits":
        """Creates a shuffled view of all splits while maintaining split boundaries.

        Args:
            seed (Optional[int]): Random seed for reproducibility. Defaults to 42.

        Returns:
            MLSplits: A new MLSplits instance with shuffled data in each split.

        Examples:
            >>> splits = MLSplits(
            ...     train=MLData(X=np.array([[1, 2]]), y=np.array([0])),
            ...     val=MLData(X=np.array([[3, 4]]), y=np.array([1]))
            ... )
            >>> shuffled = splits.shuffled_view(seed=42)
        """
        return MLSplits(
            train=self.train.shuffled_view(seed),
            val=self.val.shuffled_view(seed),
            test=self.test.shuffled_view(seed),
        )

    def __getitem__(self, idx):
        # useful for slicing
        return MLSplits(train=self.train[idx], val=self.val[idx], test=self.test[idx])

    def __len__(self):
        return sum(
            len(getattr(self, split_name)) for split_name in ["train", "val", "test"]
        )

    def __eq__(self, other):
        return (
            self.train == other.train
            and self.val == other.val
            and self.test == other.test
        )
