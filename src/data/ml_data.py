import itertools
from typing import List, Union, Literal
import os
import pickle
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, Literal, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from config.defaults import cfg
from src.data.material_split import MaterialSplitter
from utils.src.lightning.pl_data_module import PlDataModule


@dataclass
class DataSplit:
    X: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        if not len(self.X) == len(self.y):
            raise ValueError("X and y must have same length")
        if self.X.dtype != self.y.dtype:
            raise ValueError("X and y must have same dtype")

    @property
    def tensors(self):
        return torch.tensor(self.X), torch.tensor(self.y)


@dataclass
class MLSplits:
    train: DataSplit
    val: DataSplit
    test: DataSplit

    @property
    def tensors(self):
        out_tensor = (self.train.tensors, self.val.tensors, self.test.tensors)
        flatten = itertools.chain.from_iterable
        return tuple(flatten(out_tensor))


@dataclass
class DataQuery:
    compound: str
    simulation_type: Literal["FEFF", "VASP"]


class FeatureProcessor:
    def __init__(self, query: DataQuery, data_splits: MLSplits = None):
        self.query = query
        # none option set to access saved pca and scaler from cache
        self.data_splits = data_splits

    @cached_property
    def splits(self):
        return MLSplits(
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


def filter_anamolous_spectras(
    ml_splits: MLSplits,
    std_cutoff: float,
    id_site: List[tuple],
) -> MLSplits:

    spectras = np.concatenate([ml_splits.train.y, ml_splits.val.y, ml_splits.test.y])
    mean = np.mean(spectras, axis=0)
    std = np.std(spectras, axis=0)
    upper_bound = mean + std_cutoff * std
    lower_bound = mean - std_cutoff * std
    select_filters_count = 0

    # use removed spectra in dict format
    removed_spectra = np.array(
        [
            {
                "id": id_site[i][0],
                "site": id_site[i][1],
                "spectrum": s,
            }
            for i, s in enumerate(spectras)
            if not np.all((s <= upper_bound) & (s >= lower_bound))
        ]
    )
    for data in [ml_splits.train, ml_splits.val, ml_splits.test]:
        bound_condition = (data.y <= upper_bound) & (data.y >= lower_bound)
        filter = np.all(bound_condition, axis=1)
        select_filters_count += sum(filter)
        data.y = data.y[filter]
        data.X = data.X[filter]

    remove_filter_count = len(spectras) - select_filters_count
    remove_pct = remove_filter_count / len(spectras) * 100

    msg = f"Removed {remove_filter_count}/{len(spectras)} ({remove_pct:.2f}%) anamolies"
    msg += f"with std_cutoff = {std_cutoff}"
    warnings.warn(msg)

    return (
        MLSplits(
            train=ml_splits.train,
            val=ml_splits.val,
            test=ml_splits.test,
        ),
        removed_spectra,  # for data publication
    )


def load_xas_ml_data(
    query: DataQuery,
    split_fractions: Union[List[float], None] = None,
    pca_with_std_scaling: Union[bool, None] = None,
    scale_feature_and_target: bool = True,
    for_m3gnet: bool = False,
    filter_spectra_anomalies: bool = True,
    anomaly_std_cutoff: float = None,  # default loaded from config if None
    use_cache: bool = True,  # TODO: flip to False
    save_anomalous_spectras: bool = False,  # for data publication
    save_loaded_data: bool = False,  # for data publication
) -> MLSplits:
    """Loads data and does material splitting."""

    if use_cache:
        cache_file = cfg.paths.cache.splits.format(**query.__dict__)
        if not os.path.exists(os.path.dirname(cache_file)):
            os.makedirs(os.path.dirname(cache_file))
        if os.path.exists(cache_file):
            warnings.warn(f"Using cached data from {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            return MLSplits(
                train=DataSplit(data["train_X"], data["train_y"]),
                val=DataSplit(data["val_X"], data["val_y"]),
                test=DataSplit(data["test_X"], data["test_y"]),
            )

    if query.compound == "ALL":  # TODO: hacky
        return load_all_data(query.simulation_type, split_fractions=split_fractions)

    file_path = OmegaConf.load("./config/paths.yaml").paths.ml_data
    file_path = file_path.format(**query.__dict__)
    data_all = np.load(file_path, allow_pickle=True)

    # greedy multiway partitioning
    idSite = list(zip(data_all["ids"], data_all["sites"]))
    train_idSites, val_idSites, test_idSites = MaterialSplitter.split(
        idSite=idSite,
        target_fractions=split_fractions or cfg.data_module.split_fractions,
    )

    def to_split(ids):
        material_ids = ids[:, 0]
        sites = ids[:, 1]
        id_match = np.isin(data_all["ids"], material_ids)
        site_match = np.isin(data_all["sites"], sites)
        filter = np.where(id_match & site_match)[0]
        X = data_all["features"][filter].astype(np.float32)
        y = data_all["spectras"][filter].astype(np.float32)
        return DataSplit(X, y)

    splits = MLSplits(
        train=to_split(train_idSites),
        val=to_split(val_idSites),
        test=to_split(test_idSites),
    )

    if pca_with_std_scaling is None:
        pca_with_std_scaling = query.simulation_type in cfg.dscribe.features
    out = FeatureProcessor(query, splits).splits if pca_with_std_scaling else splits

    if scale_feature_and_target:  # TODO: use standard scaler or stg
        out.train.X *= 1000
        out.val.X *= 1000
        out.test.X *= 1000
        out.train.y *= 1000
        out.val.y *= 1000
        out.test.y *= 1000

    if filter_spectra_anomalies:
        anamoly_std_cutoff = (
            cfg.ml_data.anamoly_filter_std_cutoff.get(query.simulation_type)
            if anomaly_std_cutoff is None
            else anomaly_std_cutoff
        )
        out, removed_spectra = filter_anamolous_spectras(
            out,
            std_cutoff=anamoly_std_cutoff,
            id_site=idSite,
        )

        if save_anomalous_spectras:
            removed_spectra_file = cfg.paths.cache.removed_spectra.format(
                **query.__dict__
            )
            os.makedirs(os.path.dirname(removed_spectra_file), exist_ok=True)
            for i in range(len(removed_spectra)):
                removed_spectra[i]["spectrum"] = [
                    float(x) / 1000 for x in removed_spectra[i]["spectrum"]
                ]
            if os.path.exists(removed_spectra_file):
                raise ValueError(f"File exists: {removed_spectra_file}")
            with open(removed_spectra_file, "w") as f:
                for s in removed_spectra:
                    f.write(f"{s['id']} {s['site']}\n")

    if save_loaded_data:

        directory = cfg.paths.cache.ml_dir
        os.makedirs(directory, exist_ok=True)
        # save train,val, text ids and X and y in text file
        for split in ["train", "val", "test"]:
            split_data = getattr(out, split)
            file_X = os.path.join(
                directory, f"{query.compound}_{query.simulation_type}_{split}_X.txt"
            )
            file_y = os.path.join(
                directory, f"{query.compound}_{query.simulation_type}_{split}_y.txt"
            )
            np.savetxt(file_X, split_data.X / 1000)  # TODO: remove hardcoding
            np.savetxt(file_y, split_data.y / 1000)  # TODO: remove hardcoding
            # save id sites
            id_site_dir = os.path.join(directory, "id_site")
            os.makedirs(id_site_dir, exist_ok=True)
            for split, idSite in zip(
                ["train", "val", "test"],
                [train_idSites, val_idSites, test_idSites],
            ):
                split_data = getattr(out, split)
                file_name = os.path.join(
                    id_site_dir,
                    f"{query.compound}_{query.simulation_type}_{split}.txt",
                )
                with open(file_name, "w") as f:
                    for i in range(len(idSite)):
                        f.write(f"{idSite[i][0]} {idSite[i][1]}\n")

        # save with id site and spectra

        # save material id, site, and spectra without scaling
        # di
        # os.makedirs(os.path.dirname(filtered_data_file), exist_ok=True)
        # if os.path.exists(filtered_data_file):
        #     raise ValueError(f"File exists: {filtered_data_file}")
        # with open(filtered_data_file, "w") as f:
        #     # use index as
        #     header = "id site"
        #     header += " ".join([f"grid_{i}" for i in range(out.train.X.shape[1])])

        # f.write(header + "\n")

    if use_cache:
        cache_file = cfg.paths.cache.splits.format(**query.__dict__)
        if not os.path.exists(os.path.dirname(cache_file)):
            os.makedirs(os.path.dirname(cache_file))
        np.savez(
            cache_file,
            train_X=out.train.X,
            train_y=out.train.y,
            val_X=out.val.X,
            val_y=out.val.y,
            test_X=out.test.X,
            test_y=out.test.y,
        )

    return out


class XASPlData(PlDataModule):
    def __init__(
        self,
        query: DataQuery,
        dtype: torch.dtype = torch.float,
        split_fractions: Union[List[float], None] = None,
        **data_loader_kwargs,
    ):
        def dataset(split: DataSplit):
            X, y = split.tensors
            return TensorDataset(X.type(dtype), y.type(dtype))

        split_fractions = split_fractions or cfg.data_module.split_fractions

        ml_split = load_xas_ml_data(query, split_fractions=split_fractions)
        super().__init__(
            train_dataset=dataset(ml_split.train),
            val_dataset=dataset(ml_split.val),
            test_dataset=dataset(ml_split.test),
            **data_loader_kwargs,
        )


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
            data_dict[c] = MLSplits(
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

    data_all = MLSplits(
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

    data_all = MLSplits(
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
    pass

    # # =============================================================================
    # #  anmoly filter test
    # # =============================================================================
    # from utils.src.plots.heatmap_of_lines import heatmap_of_lines
    # import matplotlib.pyplot as plt
    # data = load_xas_ml_data(DataQuery("Cu", "VASP"), filter_anamolies=False)
    # heatmap_of_lines(data.train.y)
    # plt.show()
    # data = load_xas_ml_data(DataQuery("Cu", "VASP"), filter_anamolies=True)
    # heatmap_of_lines(data.train.y)
    # plt.show
    # # ==============================================================================

    # # %%

    # # =============================================================================
    # # tests if pca and scaler are cached
    # # =============================================================================
    # from p_tqdm import p_map

    # load_xas_ml_data(DataQuery("Cu", "SOAP"))

    # # # should cache pca and scaler
    # # p_map(
    # #     lambda q: load_xas_ml_data(q),
    # #     [DataQuery(c, "SOAP") for c in cfg.compounds[-1]],
    # #     num_cpus=1,
    # # )

    # # for compound in cfg.compounds:
    # #     query = DataQuery(compound=compound, simulation_type="SOAP")
    # #     # caching pca
    # #     ml_split = load_xas_ml_data(query)
    # #     pca = FeatureProcessor(query).pca  # should load from cache
    # #     print(f"PCA components: {pca.n_components_} for {query}")

    # # =============================================================================
    # print("dummy")

    # # pl_data = XASPlData(query=DataQuery(compound="Cu", simulation_type="FEFF"))
    # # def print_fractions(xas_data):
    # #     dataset = [xas_data.train_dataset, xas_data.val_dataset, xas_data.test_dataset]
    # #     sizes = np.array([len(x) for x in dataset])
    # #     sizes = sizes / sizes.sum()
    # #     print(sizes)
    # # print_fractions(pl_data)  # [0.8 0.1 0.1]
