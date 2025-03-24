

from dataclasses import dataclass, is_dataclass
from simple_parsing.helpers import Serializable
from simple_parsing import parse
from simple_parsing.utils import DataclassT

from typing import TypeVar

C = TypeVar('C')

@dataclass
class MDMConfig(Serializable):
    pass

class ConfigPrinter:
    def __init__(self, config : MDMConfig) -> None:
        print(config)

@dataclass
class CLIBuilder():
    class_to_call: type[C] = ConfigPrinter
    config_class: type = MDMConfig
    default_config : DataclassT = None
    
    def build_config(self, args: str = None) -> DataclassT:
        assert is_dataclass(self.config_class)
        cfg: DataclassT = parse(
            config_class=self.config_class, add_config_path_arg="config-file", default=self.default_config, args=args
        )
        return cfg

    def run(self)-> C:
        cfg: DataclassT = self.build_config()
        return self.class_to_call(cfg)
