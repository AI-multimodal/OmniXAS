# %%
import json
import os
import re
import warnings
from glob import glob
from typing import Any, Dict, Iterator, List, Optional, Set, Type, TypeVar, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel

# %%


T = TypeVar("T", bound=BaseModel)


class FileHandler:
    def __init__(
        self,
        config: Union[str, Dict[str, Any]],
        replace_existing: bool = False,
    ):
        self.config = self._load_config(config)
        self.replace_existing = replace_existing

    def _load_config(self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(config, str):
            if not os.path.exists(config):
                raise FileNotFoundError(f"Configuration file not found: {config}")
            with open(config, "r") as config_file:
                return yaml.safe_load(config_file)
        return config

    def serialize_json(
        self,
        obj: BaseModel,
        supplemental_info: Optional[Union[Dict[str, Any], T]] = None,
        custom_filepath: Optional[str] = None,
    ) -> None:
        filepath = custom_filepath or self._get_filepath(obj, supplemental_info)
        self._ensure_directory(os.path.dirname(filepath))
        self._check_file_exists(filepath)

        with open(filepath, "w") as f:
            json.dump(obj.model_dump(), f, indent=2)

    def deserialize_json(
        self,
        obj_class: Type[T],
        supplemental_info: Optional[Union[Dict[str, Any], T]] = None,
        custom_filepath: Optional[str] = None,
    ) -> T:
        filepath = custom_filepath or self._get_filepath(obj_class, supplemental_info)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)
            return obj_class(**data)

    def _get_filepath(
        self,
        obj: Union[BaseModel, Type[BaseModel]],
        supplemental_info: Optional[Union[Dict[str, Any], BaseModel]] = None,
    ) -> str:
        config_name = obj.__name__ if isinstance(obj, type) else obj.__class__.__name__
        config = self._get_config(config_name)
        if supplemental_info:
            if isinstance(supplemental_info, BaseModel):
                supplemental_info = supplemental_info.dict()
            if isinstance(obj, BaseModel):
                obj = obj.dict()
            elif isinstance(obj, type):
                obj = {}

            obj = {**obj, **supplemental_info}
        dir_path = self._resolve_template(obj, config["directory"])
        filename = self._resolve_template(obj, config["filename"])
        return os.path.join(dir_path, filename)

    def _get_config(self, config_name: str) -> Dict[str, Any]:
        config = self.config.get(config_name)
        if not config:
            raise ValueError(f"Configuration for {config_name} not found")
        return config

    def _ensure_directory(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)

    def _check_file_exists(self, filepath: str) -> None:
        if os.path.exists(filepath) and not self.replace_existing:
            raise FileExistsError(f"File already exists: {filepath}")

    @staticmethod
    def _resolve_template(obj: Any, template: str) -> str:
        for prop in re.findall(r"\{(.*?)\}", template, re.DOTALL):
            value = FileHandler._get_nested_attr(obj, prop)
            value = "" if value is None else str(value)
            template = template.replace("{" + prop + "}", value)
        return template

    @staticmethod
    def _get_nested_attr(obj: Any, attr: str) -> Any:
        parts = attr.split(".")
        for part in parts:
            if isinstance(obj, dict):
                if part not in obj:
                    raise AttributeError(f"Key {part} not found in {obj}")
                obj = obj[part]
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise AttributeError(f"Attribute {part} not found in {obj}")
        return obj

    def serialized_objects_filepaths(
        self,
        config_name: Union[Type[T], str],
        **template_params: Any,
    ) -> List[str]:
        config_name = (
            config_name.__name__ if isinstance(config_name, type) else config_name
        )
        config = self._get_config(config_name)

        dir_template = config["directory"]
        file_template = config["filename"]

        dir_path = self._resolve_template(template_params, dir_template)
        file_pattern = self._template_to_regex(file_template)

        paths = glob(os.path.join(dir_path, "*"))
        if not paths:
            msg = f"No files found for {config_name} with params {template_params}"
            msg += f"\nDir: {dir_path}\nPattern: {file_pattern}"
            warnings.warn(msg)
            return []

        return [
            filepath
            for filepath in paths
            if re.match(file_pattern, os.path.basename(filepath))
        ]

    def fetch_serialized_objects(
        self, obj_class: Type[T], **template_params: Any
    ) -> Iterator[T]:
        for filepath in self.serialized_objects_filepaths(obj_class, **template_params):
            yield self.deserialize_json(obj_class, custom_filepath=filepath)

    def _template_to_regex(self, template: str) -> str:
        # Convert template to regex pattern
        pattern = re.escape(template)
        pattern = pattern.replace(r"\{", "{").replace(r"\}", "}")
        pattern = re.sub(r"{.*?}", r".*?", pattern)
        return f"^{pattern}$"


class DEFAULTFILEHANDLER:
    def __new__(cls):
        return FileHandler(
            config=OmegaConf.load("config/serialization.yaml").serialization,
            replace_existing=False,
        )


# %%

if __name__ == "__main__":
    from omnixas.data import Element, ElementSpectrum, SpectrumType, Material
    from omnixas.utils import DEFAULTFILEHANDLER

    objects = DEFAULTFILEHANDLER().fetch_serialized_objects(
        ElementSpectrum, element=Element.Cu, type=SpectrumType.FEFF
    )
    objects = list(objects)
    print("Length:", len(objects))

# %%
file_handler = DEFAULTFILEHANDLER()
# get
