from omnixas.data.ml_data import MLSplits, DataTag
from omnixas.utils import DEFAULTFILEHANDLER
from typing import Dict, Union


def load_ml_splits(data_tag: Union[Dict, DataTag]) -> MLSplits:
    return DEFAULTFILEHANDLER().deserialize_json(MLSplits, supplemental_info=data_tag)
