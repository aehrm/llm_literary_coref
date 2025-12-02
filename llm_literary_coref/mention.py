from dataclasses import dataclass
from typing import Optional, List, Iterator
import re

import pandas


def _split_at_delimiter(text, delimiter):
    """
    Splits a string by a delimiter, but respects nested parentheses.
    e.g. "key=(a|b)|key2=c" split by "|" becomes ["key=(a|b)", "key2=c"]
    """
    result = []
    current = []
    depth = 0

    for char in text:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1

        if char == delimiter and depth == 0:
            result.append("".join(current))
            current = []
        else:
            current.append(char)

    result.append("".join(current))
    return [s for s in result if s]


def _parse_key_value(text):
    """
    Parses a string like "key1=val1|key2=val2" into a dictionary.
    Uses _split_at_delimiter to handle nested structures.
    """
    # Remove surrounding parens if present
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]

    items = _split_at_delimiter(text, '|')
    data = {}
    for item in items:
        if '=' in item:
            key, value = item.split('=', 1)
            data[key] = value
    return data


@dataclass
class Entity:
    id: str
    fullname: str
    gender: str
    specialcase_entity: List[str]
    borderline_entity: List[str]
    possible_identity_with: Optional[str] = None
    members: Optional[List[str]] = None
    all_members_given: Optional[bool] = None

    def __str__(self):
        fullname = re.sub(r'"', '', self.fullname)
        fields = [f'entity_id={self.id}', f'fullname={fullname}', f'gender={self.gender}']

        if self.possible_identity_with:
            fields.append(f'possible_identity_with={self.possible_identity_with}')
        if self.members:
            fields.append(f'members={",".join(sorted(self.members))}')
            fields.append(f'all_members_given={self.all_members_given}')
        if self.borderline_entity:
            fields.append(f'borderline_entity={",".join(self.borderline_entity)}')
        if self.specialcase_entity:
            fields.append(f'specialcase_entity={",".join(self.specialcase_entity)}')
        return "(" + '|'.join(fields) + ")"

    @staticmethod
    def parse(text: str) -> 'Entity':
        data = _parse_key_value(text)

        # helper to parse comma lists
        def parse_list(key):
            val = data.get(key)
            return val.split(',') if val else None

        return Entity(
            id=data['entity_id'],
            fullname=data['fullname'],
            gender=data['gender'],
            specialcase_entity=parse_list('specialcase_entity') or [],
            borderline_entity=parse_list('borderline_entity') or [],
            possible_identity_with=data.get('possible_identity_with', None),
            members=parse_list('members') if 'members' in data else None,
            all_members_given=data.get('all_members_given').lower() == 'true' if 'all_members_given' in data else None
        )


@dataclass
class Reference:
    entity: Entity
    borderline_reference: List[str]
    specialcase_reference: List[str]

    def __str__(self):
        fields = [f'entity={str(self.entity)}']
        if self.borderline_reference:
            fields.append(f'borderline_reference={",".join(self.borderline_reference)}')
        if self.specialcase_reference:
            fields.append(f'specialcase_reference={",".join(self.specialcase_reference)}')
        return '(' + '|'.join(fields) + ')'

    @staticmethod
    def parse(text: str) -> 'Reference':
        data = _parse_key_value(text)

        def parse_list(key):
            val = data.get(key)
            return val.split(',') if val else []

        return Reference(
            entity=Entity.parse(data['entity']),
            borderline_reference=parse_list('borderline_reference'),
            specialcase_reference=parse_list('specialcase_reference')
        )


@dataclass
class Mention:
    id: int|str
    token_idx: List[int]
    references: List[Reference]

    def __str__(self):
        return f"(mention_id={self.id}|references={','.join(str(ref) for ref in self.references)})"

    def with_references(self, references):
        return Mention(id=self.id, token_idx=self.token_idx, references=references)

    @staticmethod
    def parse(text: str, token_idx) -> 'Mention':
        data = _parse_key_value(text)
        return Mention(
            id=data['mention_id'],
            token_idx=token_idx,
            references=[Reference.parse(r) for r in _split_at_delimiter(data['references'], ',')]
        )




def parse_mentions(series: pandas.Series) -> Iterator[Mention]:
    mention_id = series.fillna('').apply(lambda x: re.findall(r'mention_id=([^|]*)\|', x)).apply(
        lambda x: x[0] if x else None)
    for _, mention_rows in series.groupby(mention_id):
        mention = Mention.parse(mention_rows.iloc[0], list(mention_rows.index))
        if len(mention.references) > 0:
            yield mention
