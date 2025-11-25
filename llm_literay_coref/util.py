import dataclasses
import json
from omegaconf import OmegaConf, DictConfig


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, DictConfig):
            return dict(o)
        if isinstance(o, Exception):
            return str(o)
        return super().default(o)
