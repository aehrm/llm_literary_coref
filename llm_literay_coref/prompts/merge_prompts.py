from typing import List

from pydantic import BaseModel

from llm_literay_coref.prompts.prompt import MentionPrompt, MentionAnnotatorOutputLine, T, O


class BasicAnnotationReference(BaseModel):
    entity_id: str
    entity_global_id: str


class BasicMergePrompt(MentionPrompt[BasicAnnotationReference]):

    def parse_response_annotation(self, response: str) -> BasicAnnotationReference:
        pass

    def prompt_tempate(self) -> str:
        pass

    def decode(self, llm_output: List[MentionAnnotatorOutputLine[T]]) -> O:
        pass