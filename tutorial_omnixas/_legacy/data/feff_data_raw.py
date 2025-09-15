import os
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import numpy as np

from _legacy.data.raw_data import RAWData


@dataclass
class RAWDataFEFF(RAWData):
    compound: str
    simulation_type: Literal["FEFF"] = field(default="FEFF", init=True)
    intermediate_dir: Literal["FEFF-XANES"] = field(default="FEFF-XANES", init=True)

    def _check_convergence(self, material_id: str, site: str):
        dir = os.path.join(
            self.base_dir or "",
            material_id,
            f"{self.simulation_type}-XANES",
            site,
        )
        if os.path.exists(os.path.join(dir, "feff.out")):
            with open(os.path.join(dir, "feff.out"), "r") as f:
                lines = f.readlines()
                converge_kwrd = "Convergence reached in"
                return any([line.startswith(converge_kwrd) for line in lines])
        else:
            return False

    def mu(self, material_id, site):
        if not self._check_convergence(material_id, site):
            warnings.warn(f"Unconverged {material_id} at {site}")
            return None
        dir_path = os.path.join(
            self.base_dir or "",
            material_id,
            f"{self.simulation_type}-XANES",
            site,
        )
        file_path = os.path.join(dir_path, "xmu.dat")
        if not os.path.exists(file_path):
            warnings.warn(f"Missing mu for {material_id} at {site}")
            return None
        with open(file_path, "r") as f:
            lines = f.readlines()
            line = list(filter(lambda x: x.startswith("#  xsedge+"), lines))[0]
            normalization = float(line.split()[-1])
            data = np.loadtxt(file_path)
            energy = data[:, 0]
            mu = data[:, 3] * normalization
            if mu is None:
                warnings.warn(f"Missing mu for {material_id} at {site}")
                return None
            data = np.array([energy, mu]).T
            data.setflags(write=False)
            return data

    @cached_property
    def parameters(self):
        parameters = {}
        for mat_id in self._material_ids:
            for site in self._sites.get(mat_id, []):
                mu_val = self.mu(mat_id, site)
                if mu_val is None:
                    self.missing_data.add((mat_id, site))
                    continue
                mu_val.setflags(write=False)  # redundant but clearer
                parameters[mat_id, site] = {"mu": mu_val}
        if len(self.missing_data) > 0:
            warnings.warn(f"{len(self.missing_data)} missing data for {self.compound}")
        return parameters


if __name__ == "__main__":
    data = RAWDataFEFF(compound="Ti")
    data.parameters
    print(f"data: {len(data)}, missing: {len(data.missing_data)}")

    id = next(iter(data._material_ids))
    site = next(iter(data._sites[id]))
    id = ("mp-390", "000_Ti")  # reference to another paper data

    from matplotlib import pyplot as plt

    plt.plot(
        data.parameters[id]["mu"][:, 0],
        data.parameters[id]["mu"][:, 1],
    )
    plt.show()
