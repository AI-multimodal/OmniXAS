from loguru import logger
from scipy.constants import physical_constants
import os
import re
import warnings
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import numpy as np

from _legacy.data.raw_data import RAWData


@dataclass
class RAWDataVASP(RAWData):
    compound: str
    simulation_type: Literal["VASP"] = field(default="VASP", init=True)
    intermediate_dir: Literal["VASP"] = field(default="VASP", init=True)

    def __len__(self):
        return len(self.parameters)

    @cached_property
    def parameters(self):
        """Parameters for each site in each id where all data is available"""
        parameters = {}
        for id in self._material_ids:
            E_GS_val = self.E_GS(id)
            if E_GS_val is None:
                sites_with_id = [(id, site) for site in self._sites[id]]
                self.missing_data.update(sites_with_id)
                continue
            for site in self._sites.get(id, []):
                mu_val = self.mu(id, site)
                e_cbm_val = self.e_cbm(id, site)
                E_ch_val = self.E_ch(id, site)
                volume_val = self.volume(id, site)
                if (
                    mu_val is None
                    or e_cbm_val is None
                    or E_ch_val is None
                    or volume_val is None
                ):
                    self.missing_data.add((id, site))
                    continue
                parameters[id, site] = {
                    "mu": mu_val,
                    "e_cbm": e_cbm_val,
                    "E_ch": E_ch_val,
                    "E_GS": E_GS_val,
                    "e_core": self.e_core,
                    "volume": volume_val,
                }
        if len(self.missing_data) > 0:
            warnings.warn(f"{len(self.missing_data)} missing data for {self.compound}")
        return parameters

    @property
    def poscar_paths(self):
        def fn(i, s):
            return os.path.join(
                self.base_dir or "", i, self.simulation_type, s, "POSCAR"
            )

        paths = {
            (i, s): fn(i, s) for i in self._material_ids for s in self._sites.get(i, [])
        }
        return paths

    def volume(self, id, site):
        file_path = os.path.join(
            self.base_dir or "",
            id,
            self.simulation_type,
            site,
            "POSCAR",
        )
        with open(file_path, "r") as f:
            lines = f.readlines()
            if len(lines) < 5:
                warnings.warn(f"Not enough data in {file_path} to calculate volume")
                return None
            vx, vy, vz = lines[2:5]
            vx, vy, vz = tuple(
                map(
                    lambda x: np.array(x.strip().split(), dtype=float),
                    [vx, vy, vz],
                )
            )
            if len(vx) != 3 or len(vy) != 3 or len(vz) != 3:
                raise ValueError(f"Invalid lattice vectors in {file_path}")
            volume = np.abs(np.dot(vx, np.cross(vy, vz)))
            if volume == 0:
                raise ValueError(f"Volume is zero for {file_path}")

            bohrs_radius = physical_constants["Bohr radius"][0]
            bohrs_radius /= physical_constants["Angstrom star"][0]
            volume /= bohrs_radius**3  # in bohr^3 units
            return volume

    @cached_property
    def e_core(self):
        return RAWData.configs()["e_core"][self.compound]

    def _read_mu_file(self, file_path):
        with open(file_path, "r") as f:
            mu = f.readlines()
            mu = list(map(lambda x: x.strip().split(), mu))
            mu = np.array(mu, dtype=float)
            if len(mu) == 0:
                raise ValueError(f"File is empty: {file_path}")
            return mu

    def _generate_xmu_avg(self, mu_avg_file_path, mu_sum_file_path):
        # because simulation data was saved as sum instead of avg
        mu_sum = self._read_mu_file(mu_sum_file_path)
        mu_avg = np.array([mu_sum[:, 0], mu_sum[:, 1] / 3]).T
        with open(mu_avg_file_path, "w") as f:
            for row in mu_avg:
                f.write(f"{row[0]} {row[1]}\n")
            logger.info(f"Generated {mu_avg_file_path}")

    def mu(self, id, site):
        dir_path = os.path.join(
            self.base_dir or "",
            id,
            self.simulation_type,
            site,
        )
        avg_path = os.path.join(dir_path, "xmu_avg.dat")
        sum_path = os.path.join(dir_path, "mu2.txt")
        if not os.path.exists(avg_path):
            if os.path.exists(sum_path):
                self._generate_xmu_avg(avg_path, sum_path)
                self.mu(id, site)
            else:
                return None
        return self._read_mu_file(avg_path)

    def e_cbm(self, id, site):
        file_path = os.path.join(
            self.base_dir or "",
            id,
            self.simulation_type,
            site,
            "efermi.txt",
        )
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r") as f:
            e_cbm = f.readline()
            if e_cbm == "":
                return None
            return float(e_cbm)

    def E0(self, file_path):
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r") as f:
            last_line = deque(f, maxlen=1)
            if len(last_line) == 0:
                return None
            last_line = last_line.pop()
            reg_pattern = r"E0=\s*(-?\d*\.\d*E[+-]\d*)\s"
            match = re.search(reg_pattern, last_line)
            if not match:
                return None
            return float(match.group(1))

    def E_ch(self, id, site):
        return self.E0(
            os.path.join(
                self.base_dir or "",
                id,
                self.simulation_type,
                site,
                "OSZICAR",
            )
        )

    def E_GS(self, id):
        return self.E0(
            os.path.join(
                self.base_dir or "",
                id,
                self.simulation_type,
                "SCF",
                "OSZICAR",
            )
        )


if __name__ == "__main__":
    data = RAWDataVASP(compound="Ti")
    data.parameters
    print(f"data: {len(data)}, missing: {len(data.missing_data)}")
    from matplotlib import pyplot as plt

    id = ("mp-390", "000_Ti")  # reference to another paper data

    plt.plot(data.parameters[id]["mu"][:, 1])
    plt.show()
