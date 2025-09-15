from typing import List
import os
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, Set


@dataclass
class RAWData(ABC):
    compound: Optional[str] = field(default=str(), init=True)
    simulation_type: Optional[str] = field(default=None, init=True)
    intermediate_dir: Optional[str] = field(default=None, init=True)
    base_dir: Optional[str] = field(default=str(), init=False)
    missing_data: Set[tuple] = field(default_factory=set, init=False)

    def __post_init__(self):
        if self.base_dir is str():
            self.base_dir = self._default_base_dir()
        self.parameters

    def _default_base_dir(self) -> str:
        default_base_dir = os.path.join(  # default
            "dataset",
            f"{self.simulation_type}-raw-data",
            f"{self.compound}",
        )
        if not os.path.exists(default_base_dir):
            raise ValueError(f"{default_base_dir} does not exist")
        return default_base_dir

    @cached_property
    def _material_ids(self) -> List[str]:
        """Can include ids with missing data"""
        ids = os.listdir(self.base_dir)
        id_filter = re.compile(r"(mp-\d+|mvc-\d+)")
        ids = list(filter(lambda x: id_filter.match(x), ids))
        if len(ids) == 0:
            raise ValueError(f"No ids found for {self.compound}")
        return ids

    @property
    def ids(self):
        # includes valid id and site combinations
        # return np.array(["_site_".join(x) for x in list(self.parameters.keys())])
        return list(self.parameters.keys())

    @cached_property
    def _sites(self):
        """Can include sites with missing data"""
        sites = {}
        for id in self._material_ids:
            dir_path = os.path.join(
                self.base_dir or "",
                id,
                self.intermediate_dir or "",
            )
            # dir_path = os.path.join(self.base_dir, id, self.intermediate_dir)
            folders = os.listdir(dir_path)
            pattern = r"(\d+)_" + (self.compound or "")
            site_list = list(filter(lambda x: re.match(pattern, x), folders))
            if len(site_list) == 0:
                warnings.warn(f"No sites found for {id} at {folders}")
            sites[id] = site_list
        return sites

    @cached_property
    def total_sites(self):
        return len([s for sites in self._sites.values() for s in sites])

    @staticmethod
    def configs(cfg_path="cfg/transformations.yaml"):
        from omegaconf import OmegaConf

        return OmegaConf.load(cfg_path)

    def __len__(self):
        return self.total_sites

    @abstractmethod
    def mu(self, id, site):
        pass

    @abstractmethod
    def parameters(self):
        pass
