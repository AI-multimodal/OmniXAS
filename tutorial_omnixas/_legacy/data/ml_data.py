from dataclasses import dataclass
from typing import List, Union, Literal
import os
import pickle
import warnings
from functools import cached_property

import numpy as np
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.defaults import cfg
from omnixas.data.material_split import MaterialSplitter


@dataclass
class DataQuery:
    element: str
    simulation: Literal["FEFF", "VASP"]


@dataclass
class DataSplit:
    X: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        if not len(self.X) == len(self.y):
            raise ValueError("X and y must have same length")
        if self.X.dtype != self.y.dtype:
            raise ValueError("X and y must have same dtype")

    def __len__(self):
        return len(self.X)


@dataclass
class MlSplit:
    train: DataSplit
    val: DataSplit
    test: DataSplit

    def __iter__(self):
        return iter([self.train, self.val, self.test])

    def __len__(self):
        return sum([len(x) for x in self])


class FeatureProcessor:
    def __init__(self, query: DataQuery, data_splits: MlSplit = None):
        self.query = query
        # none option set to access saved pca and scaler from cache
        self.data_splits = data_splits

    @cached_property
    def splits(self):
        return MlSplit(
            train=self._reduce_feature_dims(self.data_splits.train),
            val=self._reduce_feature_dims(self.data_splits.val),
            test=self._reduce_feature_dims(self.data_splits.test),
        )

    def _reduce_feature_dims(self, data_splits: DataSplit):
        return DataSplit(
            self.pca.transform(self.scaler.transform(data_splits.X)),
            data_splits.y,
        )

    @cached_property
    def scaler(self):
        # # load from cache if cache exists
        scaler_cache_path = cfg.paths.cache.scaler.format(**self.query.__dict__)
        if os.path.exists(scaler_cache_path):  # load from cache
            with open(scaler_cache_path, "rb") as f:
                return pickle.load(f)
        # # else fit scaler and save to cache
        if self.data_splits is None:
            raise ValueError("data_splits is None. Cannot fit scaler.")
        scaler = StandardScaler().fit(self.data_splits.train.X)
        os.makedirs(os.path.dirname(scaler_cache_path), exist_ok=True)
        with open(scaler_cache_path, "wb") as f:
            pickle.dump(scaler, f)
        return scaler

    @cached_property
    def pca(self):
        # load from cache if cache exists
        pca_cache_path = cfg.paths.cache.pca.format(
            **self.query.__dict__
        )  # TODO: use sys path
        if os.path.exists(pca_cache_path):
            with open(pca_cache_path, "rb") as f:
                return pickle.load(f)
        # else fit pca and save to cache
        if self.data_splits is None:
            raise ValueError("data_splits is None. Cannot fit pca.")
        pca = PCA(n_components=cfg.dscribe.pca.n_components)
        pca.fit(self.data_splits.train.X)
        os.makedirs(os.path.dirname(pca_cache_path), exist_ok=True)
        with open(pca_cache_path, "wb") as f:
            pickle.dump(pca, f)
        return pca

    def _test_if_pca_matches_config(self):
        # expected number of components or the explained variance
        expected_pca_param = cfg.dscribe.pca.n_components_
        if not self.pca.n_components == expected_pca_param:
            msg = "PCA components mismatch: "
            msg += f"{self.pca.n_components_} != {expected_pca_param} for {self.query}"
            raise ValueError(msg)


def post_split_anomlay_filter(
    ml_splits: MlSplit,
    std_cutoff: float,
) -> MlSplit:
    def bounds(spectras, std_factor):
        mean = np.mean(spectras, axis=0)
        std = np.std(spectras, axis=0)
        upper_bound = mean + std_factor * std
        lower_bound = mean - std_factor * std
        return upper_bound, lower_bound

    def filter_spectra(spectras, upper_bound, lower_bound):
        return np.all((spectras <= upper_bound) & (spectras >= lower_bound), axis=1)

    all_spectras = np.concatenate(
        [ml_splits.train.y, ml_splits.val.y, ml_splits.test.y]
    )
    upper_bound, lower_bound = bounds(all_spectras, std_cutoff)

    for data in [ml_splits.train, ml_splits.val, ml_splits.test]:
        mask = filter_spectra(data.y, upper_bound, lower_bound)
        data.y = data.y[mask]
        data.X = data.X[mask]

    return ml_splits


def load_xas_ml_data(
    query: DataQuery,
    split_fractions: Union[List[float], None] = None,
    scaling_factor: float = 1000.0,
    post_split_anomaly_filter: bool = True,
    pre_splits_anomaly_filter: bool = False,
    anomaly_std_cutoff: float = None,  # default loaded from config if None
) -> MlSplit:
    assert not (
        post_split_anomaly_filter and pre_splits_anomaly_filter
    ), "Only one of post and pre should be valid"

    if query.element == "ALL":  # TODO: hacky
        return load_all_data(query.simulation, split_fractions=split_fractions)

    file_path = OmegaConf.load("./config/paths.yaml").paths.ml_data
    file_path = file_path.format(**query.__dict__)
    npz_data = np.load(file_path, allow_pickle=True)
    data = {
        "features": npz_data["features"],
        "spectras": npz_data["spectras"],
        "ids": npz_data["ids"],
        "sites": npz_data["sites"],
    }

    anamoly_std_cutoff = (
        cfg.ml_data.anamoly_filter_std_cutoff.get(query.simulation)
        if anomaly_std_cutoff is None
        else anomaly_std_cutoff
    )

    if pre_splits_anomaly_filter:
        mask = MlSplit._identify_anomalies(data["spectras"], anamoly_std_cutoff)
        data["features"] = data["features"][mask]
        data["spectras"] = data["spectras"][mask]
        data["ids"] = data["ids"][mask]
        data["sites"] = data["sites"][mask]

    # greedy multiway partitioning
    idSite = list(zip(data["ids"], data["sites"]))
    train_idSites, val_idSites, test_idSites = MaterialSplitter.split(
        idSite=idSite,
        target_fractions=split_fractions or cfg.data_module.split_fractions,
    )

    def to_split(ids):
        material_ids = ids[:, 0]
        sites = ids[:, 1]
        id_match = np.isin(data["ids"], material_ids)
        site_match = np.isin(data["sites"], sites)
        filter = np.where(id_match & site_match)[0]
        X = data["features"][filter].astype(np.float32)
        y = data["spectras"][filter].astype(np.float32)
        return DataSplit(X, y)

    splits = MlSplit(
        train=to_split(train_idSites),
        val=to_split(val_idSites),
        test=to_split(test_idSites),
    )

    def scale_mlsplits(ml_splits: MlSplit, factor: float) -> MlSplit:
        return MlSplit(*(DataSplit(X=s.X * factor, y=s.y * factor) for s in ml_splits))

    splits = scale_mlsplits(splits, scaling_factor)

    if post_split_anomaly_filter:
        splits = post_split_anomlay_filter(
            splits,
            std_cutoff=anamoly_std_cutoff,
        )

    return splits


def load_all_data(
    sim_type="FEFF",
    split_fractions=None,
    seed=42,
    return_compound_name=False,
    compounds=None,
    sample_method: Literal[
        "same_size", "all", "stratified"
    ] = "all",  # TODO: revert to default
):
    if compounds is None:
        compounds = ["Cu", "Ti"] if sim_type == "VASP" else cfg.compounds

    data_dict = {c: load_xas_ml_data(DataQuery(c, sim_type)) for c in compounds}

    if sample_method == "same_size":
        sizes = np.array(
            [
                np.array([len(data.train.X), len(data.val.X), len(data.test.X)])
                for data in data_dict.values()
            ]
        )
        split_sizes = sizes.min(axis=0)
        warnings.warn(
            f"Using same size splits: {split_sizes} for {compounds} with {sim_type}"
        )
        for c in compounds:
            data = data_dict[c]
            data_dict[c] = MlSplit(
                train=DataSplit(
                    data.train.X[: split_sizes[0]], data.train.y[: split_sizes[0]]
                ),
                val=DataSplit(
                    data.val.X[: split_sizes[1]], data.val.y[: split_sizes[1]]
                ),
                test=DataSplit(
                    data.test.X[: split_sizes[2]], data.test.y[: split_sizes[2]]
                ),
            )

    train_compounds = [[c] * len(data_dict[c].train.X) for c in compounds]
    val_compounds = [[c] * len(data_dict[c].val.X) for c in compounds]
    test_compounds = [[c] * len(data_dict[c].test.X) for c in compounds]

    data_all = MlSplit(
        train=DataSplit(
            np.concatenate([data.train.X for data in data_dict.values()]),
            np.concatenate([data.train.y for data in data_dict.values()]),
        ),
        val=DataSplit(
            np.concatenate([data.val.X for data in data_dict.values()]),
            np.concatenate([data.val.y for data in data_dict.values()]),
        ),
        test=DataSplit(
            np.concatenate([data.test.X for data in data_dict.values()]),
            np.concatenate([data.test.y for data in data_dict.values()]),
        ),
    )
    train_compounds = np.concatenate(train_compounds)
    val_compounds = np.concatenate(val_compounds)
    test_compounds = np.concatenate(test_compounds)

    # randomize Merged data
    np.random.seed(seed)
    train_shuffle = np.random.permutation(len(data_all.train.X))
    val_shuffle = np.random.permutation(len(data_all.val.X))
    test_shuffle = np.random.permutation(len(data_all.test.X))

    data_all = MlSplit(
        train=DataSplit(
            data_all.train.X[train_shuffle], data_all.train.y[train_shuffle]
        ),
        val=DataSplit(data_all.val.X[val_shuffle], data_all.val.y[val_shuffle]),
        test=DataSplit(data_all.test.X[test_shuffle], data_all.test.y[test_shuffle]),
    )
    train_compounds = train_compounds[train_shuffle]
    val_compounds = val_compounds[val_shuffle]
    test_compounds = test_compounds[test_shuffle]

    compound_names = (train_compounds, val_compounds, test_compounds)

    return (data_all, compound_names) if return_compound_name else data_all


if __name__ == "__main__":
    splits = load_xas_ml_data(
        DataQuery("Cu", "VASP"),
        post_split_anomaly_filter=False,
        pre_splits_anomaly_filter=True,
    )
    print(splits.train.X.shape[0], splits.val.X.shape[0], splits.test.X.shape[0])
    total = splits.train.X.shape[0] + splits.val.X.shape[0] + splits.test.X.shape[0]
    print(
        splits.train.X.shape[0] / total,
        splits.val.X.shape[0] / total,
        splits.test.X.shape[0] / total,
    )
