import collections
import dataclasses
import itertools
import json
from typing import List

from omegaconf import OmegaConf, DictConfig

from llm_literary_coref.mention import Entity, Mention


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, DictConfig):
            return dict(o)
        if isinstance(o, Exception):
            return str(o)
        return super().default(o)


def make_generic_entity_factory():
    generic_counter = itertools.count(start=1)
    def generic_entity_factory(label, borderline):
        num = next(generic_counter)
        return Entity(id=f"generic_{num:04d}",
                      fullname=label if label is not None and label != '' else f"generic_{num:04d}",
                      gender='u',
                      possible_identity_with=None,
                      members=None,
                      all_members_given=None,
                      specialcase_entity=['generic'],
                      borderline_entity=['generic'] if borderline else [])

    return generic_entity_factory

def split_generics_into_singletons(mentions: List[Mention], generic_entity_factory):
    references_per_entity = collections.defaultdict(list)
    for mention in mentions:
        for ref in mention.references:
            references_per_entity[ref.entity.id].append(ref)
    for entity_id, refs in references_per_entity.items():
        entity = refs[0].entity
        is_generic = 'generic' in entity.specialcase_entity

        if not is_generic:
            continue

        if len(refs) == 1:
            continue

        print(f'warn: generic entity {entity_id} has {len(refs)} references; will split into singletons')
        for ref in refs:
            ref.entity = generic_entity_factory(entity.fullname, 'generic' in entity.borderline_entity)
