# %%

import os
from functools import cached_property

import numpy as np
from ase.io import read
from p_tqdm import p_map

from config.defaults import cfg
from _legacy.data.ml_data import DataQuery


class DscribeFeaturizer:
    def __init__(
        self,
        featurizer_class: type,  # ACSF, SOAP, LMBTR, etc.
        compound: str,
        top_n: int = 10,
        target_data_query: DataQuery = None,
    ):
        self.featurizer_class = featurizer_class
        self.compound = compound
        self.target_data_query = target_data_query or DataQuery(self.compound, "FEFF")
        self.top_n = top_n
        self._features = self._create_features()

    @property
    def features(self):
        return self._features

    def _create_features(self):
        featurizer_kwargs = cfg.dscribe[self.featurizer_class.__name__]
        species = self._top_common_elements + [self._dummy]
        featurizer = self.featurizer_class(species=species, **featurizer_kwargs)
        features = p_map(
            # element of interest is the first element only
            lambda x: featurizer.create(x)[0],
            self.materials_with_dummy,
            num_cpus=12,
            desc=f"{self.featurizer_class.__name__} for {self.compound}",
        )
        return np.array(features)

    @cached_property
    def _ml_data(self):
        file_path_pattern = cfg.paths.ml_data
        format_args = self.target_data_query.__dict__
        file_path = file_path_pattern.format(**format_args)
        ml_data = np.load(file_path, allow_pickle=True)
        return ml_data

    @property
    def _dummy(self):
        return self._top_common_elements[-1]

    @property
    def ids(self):
        return self._ml_data["ids"]

    @property
    def sites(self):
        return self._ml_data["sites"]

    @property
    def energies(self):
        return self._ml_data["energies"]

    @property
    def spectras(self):
        return self._ml_data["spectras"]

    @cached_property
    def _top_common_elements(self):
        elements_in_all_materials = [
            element
            for atom in self.materials
            for element in atom.get_chemical_symbols()
        ]
        unique_elements, count = np.unique(
            elements_in_all_materials,
            return_counts=True,
        )
        most_common_elements = unique_elements[np.argsort(count)][::-1]
        top_common_elements = most_common_elements[: self.top_n]
        return list(top_common_elements)

    def get_atoms(self, id):
        path = cfg.paths.poscar.format(compound=self.compound, id=id)
        atoms = read(path)
        return atoms

    @cached_property
    def materials_with_dummy(self):
        materials_with_dummy = []
        for atoms in self.materials:
            atomic_symbol = atoms.get_chemical_symbols()
            for i, symbol in enumerate(atomic_symbol):
                if symbol not in self._top_common_elements:
                    atomic_symbol[i] = self._dummy
            atoms.set_chemical_symbols(atomic_symbol)
            materials_with_dummy.append(atoms)
        return materials_with_dummy

    @cached_property
    def materials(self):
        materials = [self.get_atoms(id) for id in self.ids]
        return materials

    def save(self, path=None):
        # make file name
        featurizer_name = self.featurizer_class.__name__
        path = path or cfg.paths.ml_data.format(
            compound=self.compound,
            simulation_type=featurizer_name,
        )
        if os.path.exists(path):
            raise ValueError(f"{path} already exists")
        np.savez_compressed(
            path,
            ids=self.ids,
            features=self.features,
            sites=self.sites,
            energies=self.energies,
            spectras=self.spectras,
        )


# %%

if __name__ == "__main__":
    from dscribe.descriptors import ACSF, SOAP

    top_n = 40
    simulation_type = f"ACSF{top_n}"
    for name, cls in zip(["ACSF", "SOAP"], [ACSF, SOAP]):
        for c in cfg.compounds:
            DscribeFeaturizer(featurizer_class=cls, compound=c, top_n=top_n).save(
                f"{c}_{name}{top_n}.npz"
            )

# %%
