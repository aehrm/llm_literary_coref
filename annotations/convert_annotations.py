import argparse
import collections
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Optional, Iterable
from xml.etree import ElementTree
import re

from xml.etree.ElementTree import Element

import more_itertools
import intervaltree
import pandas

from llm_literary_coref.mention import Entity, Mention, Reference


def read_entity_table(entity_table: pandas.DataFrame) -> Dict[int, Dict[str, Entity]]:
    def split_fields(s):
        if pandas.isna(s):
            return []
        return re.findall(r'\w+', s)

    assert all(~entity_table['ID_general'].str.contains('|', regex=False)), "Character '|' is disallowed in ID column"

    # verify integrity of ID
    assert entity_table['ID_general'].nunique() == len(entity_table), "ID column is not unique"

    # show fullnames with muptlipe IDs
    is_generic = entity_table['specialcase'].apply(lambda x: 'generic' in x if not pandas.isna(x) else False)
    for (name, is_generic_), g in entity_table.groupby(['fullname', is_generic]):
        if len(g) == 1:
            continue

        if is_generic_:
            continue

        print(f'warn: entities {', '.join(g["ID_general"])} have same name {name!r}')


    all_entities = {}
    entity_label_map = collections.defaultdict(dict)
    for i, row in entity_table.iterrows():
        specialcase = split_fields(row['specialcase'])
        borderline = split_fields(row['borderline' if 'borderline' in row.index else 'borderline_entity'])
        possible_identity = row['possible_identity_with' if 'possible_identity_with' in row.index else 'possible_identity']
        members = set(split_fields(row['members']))

        is_group = 'group' in specialcase
        if len(members) > 0 and not is_group:
            print(f'warn: entity {row["fullname"]}, {row["ID_general"]} has members but no group; dropping members')
            members = set()

        all_members_given = True
        if len(members) == 0:
            all_members_given = False
        if 'x' in members:
            all_members_given = False
            members.remove('x')

        # verify member IDs
        for member_name in members:
            matches = entity_table[entity_table["ID_general"] == member_name]
            if len(matches) == 0:
                print(f'warn: entity {row["fullname"]}, {row["ID_general"]} has unknown member {member_name!r} specified')

        # verify "possible identity with" field
        if not pandas.isna(possible_identity):
            matches = entity_table[entity_table["ID_general"] == possible_identity]
            if len(matches) == 0:
                print(f'warn: entity {row["fullname"]}, {row["ID_general"]} has unknown possible_identity_with {possible_identity!r}')
                possible_identity = None

        fullname = row['fullname']
        if '|' in fullname:
            print(f'warn: entity with name {fullname!r} contains pipe character; replacing with whitespace in fullname')
            fullname = fullname.replace('|', ' ')


        e = Entity(
            id=row['ID_general'],
            fullname=row['fullname'],
            gender=row['gender'],
            possible_identity_with=None if pandas.isna(possible_identity) else possible_identity,
            members=members if is_group else None,
            all_members_given=all_members_given if is_group else None,
            specialcase_entity=specialcase,
            borderline_entity=borderline,
        )
        all_entities[e.id] = e

        for chap_row in [x for x in row.index if '_chapter_' in x]:
            chap_num = int(chap_row.split('_')[-1])
            if not pandas.isna(row[chap_row]):
                entity_label_map[chap_num][row[chap_row]] = e

    return entity_label_map

def get_character_alignment(section_tokens: pandas.Series, xmi_text: str) -> List[int]:
    def char_iterator():
        for i, tok in section_tokens.items():
            for char in tok:
                yield char, i

    it = more_itertools.peekable(char_iterator())
    character_map = []

    for i in range(len(xmi_text)):
        try:
            next_char, _ = it.peek()
            if next_char == xmi_text[i]:
                _, i = next(it)
                character_map.append(i)
            else:
                character_map.append(None)
        except StopIteration:
            character_map.append(None)

    # validate
    for i, group in itertools.groupby(enumerate(character_map), key=lambda x: x[1]):
        if i is None: continue
        group = list(group)
        min_char = min(x[0] for x in group)
        max_char = max(x[0] for x in group)
        source_text_token = xmi_text[min_char:max_char+1]
        token = section_tokens.loc[i]
        assert token == source_text_token, f"character map mismatch: {token!r} != {source_text_token!r}"

    return character_map


def gather_entities(xmi: Element) -> Dict[str, Entity]:
    mention_annotations = list(xmi.findall('.//{http:///webanno/custom.ecore}Mention'))

    entities = {}
    i = 0

    # handle entities with ID; entities without ID are converted to singletons in the method `extract_references`
    keyfn = lambda x: x.get('ID', '')
    for entity_id, annotations in itertools.groupby(sorted(mention_annotations, key=keyfn), keyfn):
        if entity_id == '':
            continue

        attribs = collections.defaultdict(list)
        for annotation_obj in annotations:
            for attrib in annotation_obj:
                attribs[attrib.tag].append(attrib.text.strip())

        members = attribs['members'][0] if len(attribs['members']) > 0 else []
        all_members_given = True
        if len(members) == 0:
            all_members_given = False
        if 'x' in members:
            all_members_given = False
            members.remove('x')

        if 'generic' in attribs['specialcase_mention']:
            attribs['specialcase_entity'].append('generic')
            attribs['specialcase_mention'].remove('generic')

        if len(attribs['gender']) == 0 and 'generic' not in attribs['specialcase_entity']:
            print(f'warn: no gender specified for entity with fullname {entity_id}')

        if '|' in entity_id:
            print(f'warn: entity with ID {entity_id!r} contains pipe character; replacing with whitespace in fullname')

        i = i + 1
        entities[entity_id] = Entity(id=f'figur_{i:04d}',
             gender=attribs['gender'][0] if len(attribs['gender']) > 0 else 'u',
             possible_identity_with=attribs['possible_identity_with'][0] if len(attribs['possible_identity_with']) > 0 else None,
             members=members if 'group' in attribs['specialcase_entity'] else None,
             all_members_given=all_members_given if 'group' in attribs['specialcase_entity'] else None,
             fullname=entity_id.replace('|', ' '),
             specialcase_entity=attribs['specialcase_entity'],
             borderline_entity=attribs['borderline_entity'],
        )

    return entities



def extract_references(xmi: Element, entities: Dict[str, Entity], generic_entity_factory) -> Iterator[Tuple[Tuple[int, int], Reference]]:
    mention_annotations = list(xmi.findall('.//{http:///webanno/custom.ecore}Mention'))
    xmi_text = xmi.find('.//{http:///uima/cas.ecore}Sofa').get('sofaString')

    for annotation_obj in mention_annotations:
        mention_start = int(annotation_obj.get('begin'))
        mention_end = int(annotation_obj.get('end'))
        mention_label = annotation_obj.get('ID')

        if mention_end > len(xmi_text):
            continue

        if mention_start >= mention_end:
            continue

        attribs = [(attrib.tag, attrib.text.strip()) for attrib in annotation_obj]
        borderline_reference = [v for k, v in attribs if k == 'borderline_mention']
        specialcase_reference = [v for k, v in attribs if k == 'specialcase_mention']

        is_generic = 'generic' in specialcase_reference or 'generic' in borderline_reference
        if is_generic:
            # Generic entities may be implicitly marked by the 'specialcase_mention=generic' attribute.
            # In this case, we construct a new singleton generic entity for that reference.

            entity = generic_entity_factory(mention_label, borderline='generic' in borderline_reference)
            ref = Reference(
                entity=entity,
                borderline_reference=[x for x in borderline_reference if x != 'generic'],
                specialcase_reference=[x for x in specialcase_reference if x != 'generic'],
            )
            yield (mention_start, mention_end), ref

        else:
            if not mention_label:
                print(
                    f'warn: mention has no ID (start: {mention_start}, mention: {xmi_text[mention_start:mention_end]!r})')
                continue

            if mention_label not in entities.keys():
                print(f'warn: mention with label {mention_label!r} (start: {mention_start}, mention: {xmi_text[mention_start:mention_end]!r}) not found in entity table')
                continue

            entity = entities[mention_label]
            ref = Reference(
                entity=entity,
                borderline_reference=borderline_reference,
                specialcase_reference=specialcase_reference,
            )
            yield (mention_start, mention_end), ref



def extract_and_align_references(xmi: Element, tokens: pandas.Series, entity_label_map: Dict[str, Entity], generic_entity_factory: Optional = None) -> Iterable[Tuple[List[int], Reference]]:
    xmi_text = xmi.find('.//{http:///uima/cas.ecore}Sofa').get('sofaString')
    character_alignment = get_character_alignment(tokens, xmi_text=xmi_text)

    for (begin_char, end_char), ref in extract_references(xmi, entity_label_map, generic_entity_factory):
        assert 'generic' not in ref.specialcase_reference and 'generic' not in ref.borderline_reference, \
            "references cannot be generic!"
        token_idx = list(sorted(set(character_alignment[i] for i in range(begin_char, end_char)) - {None}))
        yield token_idx, ref

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

def split_generics_into_singletons(all_references, generic_entity_factory):
    references_per_entity = collections.defaultdict(list)
    for _, ref in all_references:
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


def group_references_to_mentions(all_references) -> Dict[int, Mention]:
    mention_counter = 0
    all_mentions: Dict[int, Mention] = {}
    # group references to mentions
    keyfn = lambda x: tuple(x[0])
    for token_idx, group in itertools.groupby(sorted(all_references, key=keyfn), key=keyfn):
        references = [x[1] for x in group]
        mention_counter += 1
        all_mentions[mention_counter] = Mention(
            id=mention_counter,
            token_idx=list(token_idx),
            references=references,
        )
    return all_mentions

def check_for_overlapping_mehtions(mentions: Dict[int, Mention], token_ser: Optional[pandas.Series] = None):
    interval_tree = intervaltree.IntervalTree()
    for mention in mentions.values():
        interval_tree.add(intervaltree.Interval(mention.token_idx[0], mention.token_idx[-1] + 1, mention.id))

    handled = set()
    for mention_id, mention in mentions.items():
        if mention_id in handled: continue

        overlap = interval_tree.overlap(mention.token_idx[0], mention.token_idx[-1] + 1)
        if len(overlap) <= 1: continue

        overlapping_mention_ids = [interval.data for interval in overlap]
        handled.update(overlapping_mention_ids)

        # verify no interleaving
        for a, b in itertools.combinations(overlapping_mention_ids, 2):
            b1 = mentions[a].token_idx[0]
            e1 = mentions[a].token_idx[-1]
            b2 = mentions[b].token_idx[0]
            e2 = mentions[b].token_idx[-1]
            if b1 < b2 < e1 < e2:
                print(f"overlapping mention {a} ({b1} to {e1}) and {b} ({b2} to {e2})")

        if all(mentions[i].token_idx == mentions[overlapping_mention_ids[0]].token_idx for i in overlapping_mention_ids):
            # all spans identical, pass
            continue
        else:
            print(f"mentions {overlapping_mention_ids} overlap but consists of different spans")
            for i in overlapping_mention_ids:
                tokens = ' '.join(token_ser.loc[mentions[i].token_idx])
                entities = ', '.join(f"{r.entity.fullname!r}" for r in mentions[i].references)
                print(f"  tokens {tokens!r}, idx {mentions[i].token_idx!r} annotated as {entities}")


def convert_xmi_annotations(xmi_path: Path, tokens: pandas.Series) -> List[Mention]:
    print(f"processing {xmi_path}")
    xmi = ElementTree.parse(xmi_path).getroot()

    entities = gather_entities(xmi)

    generic_entity_factory = make_generic_entity_factory()
    all_references = list(
            extract_and_align_references(xmi,
                                         tokens=tokens,
                                         generic_entity_factory=generic_entity_factory,
                                         entity_label_map=entities))

    # ensure that generics are singletons
    split_generics_into_singletons(all_references, generic_entity_factory)

    all_mentions = group_references_to_mentions(all_references)
    check_for_overlapping_mehtions(all_mentions, tokens)
    return list(all_mentions.values())


def convert_booklevel_annotations(annotations_dir: Path, source_df: pandas.DataFrame) -> List[Mention]:
    entity_table_path = annotations_dir / "Figurenverzeichnis.csv"
    entity_table_df = pandas.read_csv(entity_table_path, sep=';')
    entity_table = read_entity_table(entity_table_df)

    chapter_id_row = source_df['is_section_start'].cumsum()

    generic_entity_factory = make_generic_entity_factory()


    all_references: List[Tuple[List[int], Reference]] = []
    for chapter_id, chapter in source_df.groupby(chapter_id_row):
        xmi_path = list(annotations_dir.glob(f"*chapter_{chapter_id}.xmi"))[0]
        print(f"processing chapter {xmi_path}")
        xmi = ElementTree.parse(xmi_path).getroot()

        all_references.extend(list(
            extract_and_align_references(xmi,
                                         tokens=chapter['text'],
                                         generic_entity_factory=generic_entity_factory,
                                         entity_label_map=entity_table[chapter_id])))

    # ensure that generics are singletons
    split_generics_into_singletons(all_references, generic_entity_factory)

    all_mentions = group_references_to_mentions(all_references)
    check_for_overlapping_mehtions(all_mentions, source_df['text'])
    return list(all_mentions.values())



def main():
    parser = argparse.ArgumentParser(description="Extract and align mentions from XMI annotations to source TSV.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--annotation_dir', type=Path, help='Directory containing XMIs and Figurenverzeichnis.csv')
    group.add_argument('--annotation_xmi', type=Path, help='Single XMI file path')

    parser.add_argument('--source_tsv', type=Path, required=True, help='Path to source text TSV file')
    parser.add_argument('--output_tsv', type=Path, required=True, help='Path to output TSV file')

    args = parser.parse_args()

    source_df = None
    mentions = None


    if args.annotation_dir:
        print(f"Mode: Directory processing\nSource: {args.source_tsv}\nAnnotations: {args.annotation_dir}")

        source_df = pandas.read_csv(args.source_tsv, sep='\t', keep_default_na=False, index_col='i')
        mentions = convert_booklevel_annotations(args.annotation_dir, source_df=source_df)
    elif args.annotation_xmi:
        print(f"Mode: Single file processing\nSource: {args.source_tsv}\nXMI: {args.annotation_xmi}")

        source_df = pandas.read_csv(args.source_tsv, sep='\t')
        mentions = convert_xmi_annotations(args.annotation_xmi, tokens=source_df['text'])

    output_df = source_df.copy()
    output_df['gold'] = ''

    for mention in mentions:
        for k in mention.token_idx:
            assert k in output_df.index
            if k in output_df.index:
                output_df.loc[k, 'gold'] = output_df.loc[k, 'gold'] + str(mention)

    print(f"Saving output to {args.output_tsv}")
    output_df.to_csv(args.output_tsv, sep='\t', quoting=3)


if __name__ == "__main__":
    main()
