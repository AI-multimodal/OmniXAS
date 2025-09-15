# %%
from typing import List

import numpy as np

from omnixas.data.ml_data import DataTag, MLSplits
from omnixas.utils.io import FileHandler


class MergedSplits(MLSplits):
    """A class for merging multiple MLSplits objects with their associated tags.

    This class extends MLSplits to handle the combination of multiple datasets,
    with options for balanced or unbalanced merging. It's particularly useful
    when working with multiple spectrum types or elements that need to be
    combined into a single dataset.

    Examples:
        >>> from omnixas.utils.constants import Element, SpectrumType
        >>> # Create sample tags
        >>> tags = [
        ...     DataTag(element=Element.Fe, type=SpectrumType.XANES),
        ...     DataTag(element=Element.Ni, type=SpectrumType.XANES)
        ... ]
        >>>
        >>> # Load and merge splits with balanced datasets
        >>> file_handler = FileHandler("path/to/data")
        >>> merged = MergedSplits.load(
        ...     tags=tags,
        ...     file_handler=file_handler,
        ...     balanced=True
        ... )
        >>>
        >>> # Append additional data
        >>> new_split = MLSplits(...)
        >>> new_tag = DataTag(element=Element.Cu, type=SpectrumType.XANES)
        >>> merged.append(new_tag, new_split)
    """

    @classmethod
    def load(
        cls,
        tags: List[DataTag],
        file_handler: "FileHandler",
        balanced: bool = False,
        **kwargs,
    ) -> "MergedSplits":
        """Loads and merges multiple MLSplits objects from files.

        This method loads multiple datasets using their tags and optionally
        balances them by reducing each split to the size of the smallest
        corresponding split across all datasets.

        Args:
            tags (List[DataTag]): List of tags identifying the datasets to load
            file_handler (FileHandler): Handler for loading the data files
            balanced (bool, optional): If True, ensures all splits have the same
                size by reducing larger splits. Defaults to False.
            **kwargs: Additional arguments passed to file_handler

        Returns:
            MergedSplits: A new instance containing the merged datasets

        Examples:
            >>> # Load with balanced datasets (equal sizes)
            >>> merged = MergedSplits.load(
            ...     tags=[
            ...         DataTag(element=Element.Fe, type=SpectrumType.XANES),
            ...         DataTag(element=Element.Ni, type=SpectrumType.XANES)
            ...     ],
            ...     file_handler=FileHandler(...),
            ...     balanced=True
            ... )
            >>>
            >>> # Load without balancing (keeps original sizes)
            >>> merged_unbalanced = MergedSplits.load(
            ...     tags=tags,
            ...     file_handler=file_handler,
            ...     balanced=False
            ... )
        """
        splits = [
            file_handler.deserialize_json(MLSplits, supplemental_info=tag)
            for tag in tags
        ]

        if balanced:
            min_sizes = dict(
                train=min(split.train.X.shape[0] for split in splits),
                val=min(split.val.X.shape[0] for split in splits),
                test=min(split.test.X.shape[0] for split in splits),
            )
            splits = [
                MLSplits(
                    train=split.train.shuffled_view()[: min_sizes["train"]],
                    val=split.val.shuffled_view()[: min_sizes["val"]],
                    test=split.test.shuffled_view()[: min_sizes["test"]],
                )
                for split in splits
            ]

        merged = cls()
        for split, tag in zip(splits, tags):
            merged.append(tag, split)
        return merged

    def append(self, split: MLSplits):
        """Appends a new MLSplits object to the existing merged splits.

        This method concatenates the feature matrices (X) and labels (y) from
        the new split with the existing data for each split type (train/val/test).

        Args:
            split (MLSplits): The splits to append

        Examples:
            >>> merged = MergedSplits()
            >>> # Append new data
            >>> new_split = MLSplits(
            ...     train=MLData(X=np.array([[1, 2]]), y=np.array([0])),
            ...     val=MLData(X=np.array([[3, 4]]), y=np.array([1]))
            ... )
            >>> merged.append(new_split)
        """
        for attr in MLSplits.__fields__.keys():
            new_data = getattr(split, attr)
            if new_data is None:
                continue

            existing_data = getattr(self, attr)
            if existing_data is None:
                setattr(self, attr, new_data)
            else:
                existing_data.X = np.concatenate([existing_data.X, new_data.X])
                existing_data.y = np.concatenate([existing_data.y, new_data.y])
