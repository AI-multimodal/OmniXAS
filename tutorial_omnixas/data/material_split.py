import warnings
from typing import List

import numpy as np


class MaterialSplitter:

    @staticmethod
    def split(
        idSite: List[tuple],
        target_fractions: List[float],
        seed: int = 42,
    ):
        """Split the dataset into multiple partitions based on target fractions.

        Args:
            idSite: List of tuples where each tuple contains a material ID as the first
                element and associated data as subsequent elements.
            target_fractions: List of float values representing the desired fraction of
                data in each partition. Must sum to 1.0 (e.g., [0.8, 0.1, 0.1] for
                train/val/test split).
            seed: Integer seed for random number generation to ensure reproducibility.
                Defaults to 42.

        Returns:
            List[np.ndarray]: List of numpy arrays, where each array contains the
                tuples for one partition. The length of the list matches the length
                of target_fractions.

        Raises:
            ValueError: If there is any overlap between splits (same material ID
                appears in multiple partitions).

        Warns:
            UserWarning: If any IDs are not assigned to any partition.


        Example:
            >>> splitter = MaterialSplitter()
            # >>> data = [
            # ... ("mp" + str(np.random.randint(0, 20)), int(np.random.randint(0, 10)))
            # ... for _ in range(20) ]
            >>> splits = splitter.split(data, [0.8, 0.1, 0.1])
            >>> print([len(split) for split in splits])
            [16, 2, 2]
        """

        target_sums = [int(x * len(idSite)) for x in target_fractions]

        splits = MaterialSplitter._greedy_multiway_partition(idSite, target_sums, seed)

        # sanity check
        # overlap between splits
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                overlap = np.isin(splits[i][:, 0], splits[j][:, 0]).any()
                if overlap:
                    msg = f"Overlap between split {i} and {j}"
                    raise ValueError(msg)

        missing_count = len(idSite) - sum([len(x) for x in splits])
        if missing_count > 0:
            msg = f"{missing_count} ids are not part of any (train/val/test) split"
            warnings.warn(msg)

        return splits

    @staticmethod
    def _greedy_multiway_partition(
        id_pairs: List[tuple],
        target_sums: List[int],
        seed=42,
    ):
        np.random.seed(seed)
        np.random.shuffle(id_pairs)
        unique_ids_and_count = MaterialSplitter._get_unique_ids_and_counts(id_pairs)
        groups = MaterialSplitter._partition_ids(unique_ids_and_count, target_sums)
        return MaterialSplitter._assign_ids_to_groups(id_pairs, groups)

    @staticmethod
    def _get_unique_ids_and_counts(id_pair):
        ids = np.array([x[0] for x in id_pair])
        unique_ids, id_count = np.unique(ids, return_counts=True)
        return list(zip(unique_ids, id_count))

    @staticmethod
    def _partition_ids(unique_ids_and_count, target_sums):
        unique_ids_and_count = sorted(
            unique_ids_and_count, key=lambda x: x[1], reverse=True
        )
        c = len(target_sums)
        groups = [[] for _ in range(c)]
        current_sums = [0] * c
        for unique_id, count in unique_ids_and_count:
            diffs = [abs(current_sums[i] + count - target_sums[i]) for i in range(c)]
            min_group = diffs.index(max(diffs))
            groups[min_group].append(unique_id)
            current_sums[min_group] += count
        return groups

    @staticmethod
    def _assign_ids_to_groups(id_pairs, groups):
        grouped_ids = {group: [] for group in range(len(groups))}
        for id_pair in id_pairs:
            id_1 = id_pair[0]
            for group_index, group in enumerate(groups):
                if id_1 in group:
                    grouped_ids[group_index].append(id_pair)
                    break
        return [np.array(x) for x in grouped_ids.values()]
