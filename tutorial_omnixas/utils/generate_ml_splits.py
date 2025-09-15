# %%
import re
from typing import List, Tuple

import numpy as np

from omnixas.core.periodic_table import Element, SpectrumType
from omnixas.data import (
    DataTag,
    ElementSpectrum,
    Material,
    MaterialID,
    MaterialSplitter,
    MLData,
    MLSplits,
)
from omnixas.utils import DEFAULTFILEHANDLER, FileHandler


class MLSplitGenerator:
    def __init__(self, file_handler: FileHandler = None):
        self.file_handler = file_handler or DEFAULTFILEHANDLER()

    def generate_ml_splits(
        self,
        data_tag: DataTag,
        target_fractions: List[float] = [0.8, 0.1, 0.1],
        seed: int = 42,
    ):
        element = data_tag.element
        type = data_tag.type
        file_paths = self.file_handler.serialized_objects_filepaths(
            MLData, element=element, type=type
        )
        idSites = [self._ml_filename_to_idSite(file_path) for file_path in file_paths]
        train_idSites, val_idSites, test_idSites = MaterialSplitter.split(
            idSite=idSites,
            target_fractions=target_fractions,
            seed=seed,
        )
        train_data = self._to_ml_data(element, type, train_idSites)
        val_data = self._to_ml_data(element, type, val_idSites)
        test_data = self._to_ml_data(element, type, test_idSites)
        ml_splits = MLSplits(train=train_data, val=val_data, test=test_data)
        return ml_splits

    def _ml_filename_to_idSite(self, filename: str) -> Tuple[str, str]:
        pattern = ".*/(.*)_site_(.+)_(.+)_(.+).json"
        match = re.search(pattern, filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        return match.group(1), match.group(2)

    def _load_ml_data(
        self,
        idSite: Tuple[str, str],
        element: Element,
        spectra_type: SpectrumType,
    ) -> MLData:
        dummy = ElementSpectrum(
            element=element,
            type=spectra_type,
            index=int(idSite[1]),
            material=Material(id=MaterialID(idSite[0])),
        )
        dummy = {**dummy.dict(), "index_string": dummy.index_string}
        return self.file_handler.deserialize_json(MLData, dummy)

    def _to_ml_data(
        self,
        element: Element,
        spectra_type: SpectrumType,
        idSites: List[Tuple[str, str]],
    ) -> MLData:
        ml_data_list = [
            self._load_ml_data(idSite, element, spectra_type) for idSite in idSites
        ]

        X_train = np.stack([d.X for d in ml_data_list])
        y_train = np.stack([d.y for d in ml_data_list])
        return MLData(X=X_train, y=y_train)


def main():
    import tqdm

    from omnixas.data import AllDataTags

    for tag in tqdm.tqdm(AllDataTags()):
        split = MLSplitGenerator().generate_ml_splits(
            tag,
            target_fractions=[0.8, 0.1, 0.1],
            seed=42,
        )
        DEFAULTFILEHANDLER().serialize_json(
            split, supplemental_info={"element": tag.element, "type": tag.type}
        )


# %%

if __name__ == "__main__":
    # main()

    tag = DataTag(element=Element.Ti, type=SpectrumType.VASP)
    split = MLSplitGenerator().generate_ml_splits(tag)
    DEFAULTFILEHANDLER().serialize_json(split, tag)
