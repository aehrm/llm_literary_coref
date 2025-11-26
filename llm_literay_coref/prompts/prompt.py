import json
import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, TypeVar, Generic, Mapping

from llm_literay_coref.mention import Mention, Entity

logger = logging.getLogger(__name__)

T = TypeVar('T')
O = TypeVar('O')

@dataclass
class MentionAnnotatorOutputLine(Generic[T]):
    mention: Mention
    annotation: T


class MentionPrompt(Generic[T], ABC):

    @abstractmethod
    def parse_response_annotation(self, response: str) -> T:
        pass

    @abstractmethod
    def prompt_tempate(self) -> str:
        pass

    def format_prompt(self, formatted_input: str, span_annotations: list) -> str:
        incomplete_response = '\n'.join(json.dumps(a) for a in span_annotations)
        prompt = self.prompt_tempate().format(doc_formatted=formatted_input, incomplete_response=incomplete_response)
        return prompt

    @abstractmethod
    def decode(self, llm_output: List[MentionAnnotatorOutputLine[T]]) -> List[Mention]:
        pass


@dataclass
class MergeAnnotatorOutputLine:
    section_id: int
    entity_id: str
    entity_id_global: str


class MergePrompt(ABC):

    @abstractmethod
    def prompt_tempate(self) -> str:
        pass

    def format_prompt(self, entity_annotations: list) -> str:
        incomplete_response = '\n'.join(json.dumps(a) for a in entity_annotations)
        prompt = self.prompt_tempate().format(incomplete_response=incomplete_response)
        return prompt

    def decode(self, llm_output: List[MergeAnnotatorOutputLine]) -> Mapping[int | str, Entity]:
        pass

