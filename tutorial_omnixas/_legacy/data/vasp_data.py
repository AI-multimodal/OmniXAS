from typing import Optional
from _legacy.data.raw_data import RAWData
from scipy.constants import physical_constants
import numpy as np
from scipy.stats import cauchy
from _legacy.data.data import ProcessedData


class VASPData(ProcessedData):
    def __init__(self, compound, params=None, id=None, do_transform=True):
        if params is not None:
            self.e_core = params["e_core"]
            self.e_cbm = params["e_cbm"]
            self.E_ch = params["E_ch"]
            self.E_GS = params["E_GS"]
            self.volume = params["volume"]
        super().__init__(
            compound,
            simulation_type="VASP",
            params=params,
            id=id,
            do_transform=do_transform,
        )

    def transform(self, include_emperical_truncation=True):
        self.scale().broaden().align()
        if include_emperical_truncation:
            self.truncate_emperically()
        return self

    def truncate(
        self,
        start_offset=None,  # set to -gamma for broadening if None
    ):
        if start_offset is None:
            start_offset = -1 * RAWData.configs()["gamma"][self.compound]
        min_energy = (self.e_cbm - self.e_core) - start_offset
        self.filter(energy_range=[min_energy, None], spectral_range=[0, None])
        return self

    def scale(self):
        r = physical_constants["Rydberg constant times hc in eV"][0]
        a = physical_constants["inverse fine-structure constant"][0]
        omega = self._spectra * self._energy
        omega /= r * 2
        self.big_omega = self.volume
        self._spectra = (omega * self.big_omega) / a
        return self

    @staticmethod
    def lorentz_broaden(x, xin, yin, gamma):
        dx = xin[-1] - xin[0]
        differences = x[:, np.newaxis] - xin
        lorentzian = cauchy.pdf(differences, 0, gamma / 2)
        return np.dot(lorentzian, yin) / len(xin) * dx

    def broaden(self, gamma: Optional[float] = None):
        if gamma is None:
            gamma = RAWData.configs()["gamma"][self.compound]
        broadened_amplitude = VASPData.lorentz_broaden(
            self._energy,
            self._energy,
            self._spectra,
            gamma=gamma,
        )
        self._spectra = broadened_amplitude
        return self

    def align(self, emperical_offset=None):
        cfg = RAWData.configs()

        theoretical_offset = (self.e_core - self.e_cbm) + (self.E_ch - self.E_GS)
        super().align_energy(theoretical_offset)

        if emperical_offset is None:
            emperical_offset = cfg["emperical_offset"]["VASP"][self.compound]
        super().align_energy(emperical_offset)
        return self


# if __name__ == "__main__":
#     compound = "Ti"
#     simulation_type = "VASP"
#     data = RAWDataVASP(compound, simulation_type)

#     id = ("mp-390", "000_Ti")  # reference to another paper data
#     data = VASPData(data.parameters[id])

#     from matplotlib import pyplot as plt
#     import scienceplots

#     plt.style.use(["default", "science"])
#     fig = plt.figure(figsize=(8, 6))
#     data.truncate()
#     plt.plot(data.energy, data.spectra, label="truncated")
#     data.scale()
#     plt.plot(data.energy, data.spectra, label="trucated and scaled")
#     data.broaden()
#     plt.plot(data.energy, data.spectra, label="truncated, scaled, and broadened")
#     data.align_energy()
#     plt.plot(data.energy, data.spectra, label="truncated, scaled, broadened, aligned")
#     plt.xlabel("Energy (eV)")
#     plt.legend()
#     plt.title(f"VASP spectra for {id}")
#     # plt.savefig("vasp_transformations.pdf", bbox_inches="tight", dpi=300)
#     plt.show()
