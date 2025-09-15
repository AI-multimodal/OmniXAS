import matplotlib.pyplot as plt
from typing import Union, Tuple
from _legacy.data.raw_data import RAWData
import numpy as np
from dtw import dtw
import os


import warnings
from abc import ABC, abstractmethod


class ProcessedData(ABC):
    def __init__(
        self,
        compound: str,
        simulation_type: str,
        params=None,
        id: Union[Tuple[str, str], None] = None,
        do_transform: bool = True,
    ):
        self.compound = compound
        self.simulation_type = simulation_type
        self._id = id
        self._energy = self._spectra = np.array([])

        if params is not None:
            self.params = params
            self._energy = self.energy_full = self.params["mu"][:, 0]
            self._spectra = self.spectra_full = self.params["mu"][:, 1]
            if do_transform:
                self.transform()

    @property
    def id(self) -> Tuple[str, str]:
        if self._id is None:
            raise ValueError("id not set")
        return self._id

    @id.setter
    def id(self, id) -> None:
        assert (
            isinstance(id, tuple)
            and len(id) == 2
            and all(isinstance(i, str) for i in id)
        ), "id must be a tuple of two strings"
        self._id = id

    def load(self, id, file_path=None):
        if self._energy is not None or self._spectra is not None:
            warnings.warn("Data already loaded. Overwriting.")
        self.id = id
        if file_path is None:
            file_path = os.path.join(
                "dataset",
                f"{self.simulation_type}-processed-data",
                self.compound,
                "_site_".join(self.id) + ".dat",
            )
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        data = np.loadtxt(file_path)
        self._energy = self.energy_full = data[:, 0]
        self._spectra = self.spectra_full = data[:, 1]
        return self

    @property
    def energy(self) -> np.ndarray:
        if self._energy is None:
            raise ValueError("Energy values empty. Load data first.")
        return self._energy

    @energy.setter
    def energy(self, energy) -> None:
        if not np.all(np.diff(energy) > 0):
            raise ValueError("Energy values must be monotonically increasing.")
        self._energy = energy
        if self._spectra is not None and len(self._spectra) != len(self._energy):
            self._spectra = np.array([])
            warnings.warn("Spectra values reset to None.")

    @property
    def spectra(self) -> np.ndarray:
        if self._spectra is None:
            raise ValueError("Spectra values empty. Load data first.")
        return self._spectra

    @spectra.setter
    def spectra(self, spectra) -> None:
        self._spectra = spectra
        if len(self._spectra) != len(self._energy):
            self._energy = np.array([])
            warnings.warn("Energy values reset to None.")

    def reset(self):
        self._energy, self._spectra = self.energy_full, self.spectra_full
        return self

    def truncate_emperically(self):
        cfg = RAWData.configs()
        e_start = cfg["e_start"][self.compound]
        e_range_diff = cfg["e_range_diff"]
        e_end = e_start + e_range_diff
        return self.filter(energy_range=(e_start, e_end))

    def filter(self, energy_range=(None, None), spectral_range=(None, None)):
        e_start, e_end = energy_range
        s_start, s_end = spectral_range

        energy, spectra = self._energy, self._spectra
        e_filter = True if e_start is None else energy > e_start
        e_filter &= True if e_end is None else energy < e_end
        s_filter = True if s_start is None else spectra > s_start
        s_filter &= True if s_end is None else spectra < s_end

        all_filters = e_filter & s_filter
        indices = np.where(all_filters)[0]
        if len(indices) == 0:
            warnings.warn("No data points are left after filtering.")
            return self
        min_idx, max_idx = indices[0], indices[-1] + 1
        self._energy = self._energy[min_idx:max_idx]
        self._spectra = self._spectra[min_idx:max_idx]
        return self

    def __repr__(self):
        string = "Data post transformations:\n"
        string += f"energy: {self.energy}\n"
        string += f"spectra: {self.spectra}\n"
        return string

    @staticmethod
    def dtw_shift(source: "ProcessedData", target: "ProcessedData"):
        # 4x faster than compare_between_spectra
        # not consistently better compare_between_spectra
        # sometime worse
        d, cost_matrix, acc_cost_matrix, path = dtw(
            source.spectra,
            target.spectra,
            dist=lambda x, y: np.abs(x - y),  # euclidean norm
        )
        path0 = np.array(path[0]) if isinstance(path[0], range) else path[0].astype(int)
        path1 = np.array(path[1]) if isinstance(path[1], range) else path[1].astype(int)
        shifts = source.energy[path0] - target.energy[path1]
        dominant_shift = np.round(np.median(shifts)).astype(int)
        return dominant_shift

    def __len__(self):
        if len(self._energy) != len(self._spectra):
            raise ValueError("Energy and spectra is not of same length.")
        return len(self._energy)

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def truncate(self):
        pass

    @abstractmethod
    def scale(self):
        pass

    def align_energy(self, energy_offset=0):
        self._energy = self._energy + energy_offset
        return self

    def save(self, save_dir="."):
        save_dir = os.path.join(
            save_dir,
            f"{self.simulation_type}-processed-data",
            self.compound,
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            warnings.warn(f"Created directory {save_dir} to save data.")
        file_name = "_site_".join(self.id) + ".dat"
        file_path = os.path.join(save_dir, file_name)
        data_table = np.array([self.energy, self.spectra]).T
        np.savetxt(file_path, data_table, delimiter="\t")

    def resample(self, e_start=None, e_end=None, dE=None):
        e_start = (
            RAWData.configs()["e_start"][self.compound] if e_start is None else e_start
        )
        e_end = e_start + RAWData.configs()["e_range_diff"] if e_end is None else e_end
        dE = RAWData.configs()["quarter_eV_resolution"] if dE is None else dE
        num_points = int((e_end - e_start) / dE) + 1
        new_energy_grid = np.linspace(e_start, e_end, num_points)
        new_spectra = np.interp(new_energy_grid, self.energy, self.spectra)
        self.spectra = new_spectra
        self.energy = new_energy_grid
        return self

    def plot(self, ax=plt.gca(), **kwargs):
        ax.plot(self.energy, self.spectra, **kwargs)
        return ax
