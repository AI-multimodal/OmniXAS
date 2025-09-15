import os
from enum import Enum
from pydantic import RootModel


class ReadableEnums:

    def __init__(
        self,
        make_visible: bool = False,  # add new line before and after
    ):
        self.make_visible = make_visible

    def __call__(self, cls):
        def format_value(value):

            if isinstance(value, Enum):
                return value.name
            # elif isinstance(value, list):
            #     return [format_value(item) for item in value]
            # elif isinstance(value, dict):
            #     return {format_value(k): format_value(v) for k, v in value.items()}
            # elif hasattr(value, "model_dump"):
            #     return format_value(value.model_dump())
            # else:
            #     return value
            return value

        def new_str(self_inner):
            txt = ""
            if self.make_visible:
                txt += os.linesep
            txt += self_inner.__class__.__name__ + "("
            items = []
            is_root_model = issubclass(self_inner.__class__, RootModel)
            if not is_root_model:
                for key, value in self_inner.model_dump().items():
                    formatted_value = format_value(value)
                    items.append(f"{key}={formatted_value}")
            else:
                items.append(str(self_inner.root))

            txt += ", ".join(items) + ")"
            if self.make_visible:
                txt += os.linesep
            return txt

        cls.__str__ = new_str
        cls.__repr__ = new_str
        return cls
