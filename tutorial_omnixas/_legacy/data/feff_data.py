import numpy as np
from _legacy.data.data import ProcessedData
from DigitalBeamline.digitalbeamline.extern.lightshow.compare_utils import (
    compare_between_spectra,
)
from scipy import constants


class FEFFData(ProcessedData):
    def __init__(self, compound, params=None, id=None, do_transform=True):
        # note: init includes call to transform()
        super().__init__(
            compound,
            simulation_type="FEFF",
            params=params,
            id=id,
            do_transform=do_transform,
        )

    def transform(self, include_emperical_truncation=True, resample=False):
        self.truncate().scale()
        if include_emperical_truncation:
            self.truncate_emperically()
        if resample:
            self.resample()  # taking emperical range
        return self

    def truncate(self):
        return self.filter(spectral_range=[0, None])

    def scale(self):
        bohrs_radius = constants.physical_constants["Bohr radius"][0]
        bohrs_radius /= constants.angstrom
        self._spectra = self.spectra / (bohrs_radius**2)
        return self

    def align(self, target: ProcessedData):
        source = np.array([self.energy, self.spectra]).T
        shift, _ = compare_between_spectra(
            source, np.array([target.energy, target.spectra]).T
        )
        self.align_energy(-shift)
        return shift


if __name__ == "__main__":
    from _legacy.data.feff_data_raw import RAWDataFEFF

    feff_raw = RAWDataFEFF(compound="Ti")
    feff_transformed = FEFFData(feff_raw.parameters[("mp-390", "000_Ti")])

    from matplotlib import pyplot as plt

    plt.plot(feff_transformed.energy, feff_transformed.spectra)
    plt.show()
