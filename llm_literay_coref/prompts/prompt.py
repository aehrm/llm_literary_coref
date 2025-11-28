import logging
from abc import abstractmethod, ABC
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')

class Prompt(Generic[T], ABC):

    @abstractmethod
    def format_prompt(self) -> str:
        pass

    @abstractmethod
    def decode(self, json_lines: list) -> T:
        pass