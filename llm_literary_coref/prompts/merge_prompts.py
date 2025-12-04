import collections
import copy
import itertools
import json
import logging
from abc import ABC
from typing import List, Tuple, Counter, Dict

import pandas

from llm_literary_coref.mention import Mention, Reference, Entity
from llm_literary_coref.prompts.prompt import Prompt

logger = logging.getLogger(__name__)

BASIC_MERGE_PROMPT = """
# Konsolidierung von Figurenannotationen über mehrere Kapitel

## Aufgabe
Fasse die kapitelweisen Figurenannotationen eines Romans zu einer konsistenten, werkübergreifenden Figurenliste zusammen. Identifiziere dazu Figuren, die in unterschiedlichen Kapiteln unter verschiedenen Namen oder Beschreibungen (`Figurenname_Kapitel`) auftauchen, aber dieselbe Person, Gruppe oder Entität darstellen.

## Input
Du erhältst eine Liste von JSON-Objekten, wobei jedes Objekt eine in einem bestimmten Kapitel identifizierte Figur repräsentiert. Die Felder sind:
- `"Nummer"`: Eine durchlaufende Nummer aller Objekte.
- `"Kapitel"`: Die Nummer des Kapitels.
- `"Figurenname_Kapitel"`: Die Bezeichnung, die der Figur innerhalb dieses einen Kapitels gegeben wurde.
- `"Erwähnungen"`: Eine Liste von Wörtern/Phrasen, mit denen diese Figur im Text erwähnt wurde. Diese Liste dient als wichtiger Kontext für deine Entscheidung.
- `"Figurenname_Vollständig"`: Ein leeres Feld, das du mit der finalen, konsistenten Bezeichnung füllen sollst.

## Annotationsregeln
1.  **Identifiziere zusammengehörige Figuren:** Analysiere die gesamte Liste und identifiziere, welche Einträge (`Figurenname_Kapitel` aus verschiedenen Kapiteln) sich auf dieselbe Figur beziehen.
2.  **Vergib einen kanonischen Namen:** Weise jeder eindeutigen Figur im gesamten Werk EINEN konsistenten, werkübergreifenden Namen zu und trage diesen in das Feld `"Figurenname_Vollständig"` ein.
3.  **Namenswahl:** Wähle als werkübergreifenden Namen die **vollständigste und formellste Bezeichnung**, die für die Figur im gesamten Werk verfügbar ist.
    - Bevorzuge volle Namen gegenüber Teilnamen (z.B. "Sir Kaelan" statt nur "Kaelan").
    - Bevorzuge Eigennamen gegenüber Beschreibungen (z.B. "Königin Elara" statt "die Königin").
4.  **Kontext nutzen:** Nutze die Liste der `"Erwähnungen"` als entscheidende Hinweise. Wenn `"Der Fremde"` die Erwähnung `"die Gestalt"` enthält und `"Lord Valerius"` ebenfalls die Erwähnung `"die Gestalt"`, gehören sie wahrscheinlich zur selben Figur.
5.  **Vollständigkeit:** Stelle sicher, dass **JEDER** Eintrag in der Eingabeliste einen Wert im Feld `"Figurenname_Vollständig"` erhält. Auch Figuren, die nur in einem Kapitel vorkommen, erhalten ihren eigenen Namen als Wert.
6.  **WICHTIGSTE REGEL: Absolute Konsistenz.** Die einmal gewählte Bezeichnung für eine Figur (`Figurenname_Vollständig`) MUSS für alle ihre Vorkommen über alle Kapitel hinweg **exakt identisch** sein.
7.  **Familien und Gruppen:** Behandle Familien (z.B. "das Haus Arkon") und unspezifische Gruppen (z.B. "die Wachen", "Dorfbewohner") ebenfalls als eigene Entitäten mit konsistenten Namen.

## Ausgabeformat
- Gib NUR die aktualisierte Liste zurück
- Stelle sicher, dass wirklich jeder Eintrag der Input-Liste auch in der Ausgabe vorkommt
- Das Feld `"Erwähnungen"` soll in der Ausgabe ausgelassen werden.
- Ein JSON-Objekt pro Zeile (keine Liste, keine code fences)
- Keine zusätzlichen Erklärungen oder Kommentare

## Beispiel

**Input:**
```json
{{"Nummer": 0, "Kapitel": 1, "Figurenname_Kapitel": "Kaelan", "Erwähnungen": ["Kaelan", "der junge Ritter"], "Figurenname_Vollständig": ""}}
{{"Nummer": 1, "Kapitel": 1, "Figurenname_Kapitel": "die Königin", "Erwähnungen": ["die Königin", "Ihre Majestät"], "Figurenname_Vollständig": ""}}
{{"Nummer": 2, "Kapitel": 1, "Figurenname_Kapitel": "Dorfbewohner", "Erwähnungen": ["Dorfbewohner", "Leute"], "Figurenname_Vollständig": ""}}
{{"Nummer": 3, "Kapitel": 2, "Figurenname_Kapitel": "Sir Kaelan", "Erwähnungen": ["Sir Kaelan", "der Ritter"], "Figurenname_Vollständig": ""}}
{{"Nummer": 4, "Kapitel": 2, "Figurenname_Kapitel": "Königin Elara", "Erwähnungen": ["Königin Elara", "Elara"], "Figurenname_Vollständig": ""}}
{{"Nummer": 5, "Kapitel": 2, "Figurenname_Kapitel": "Der Fremde", "Erwähnungen": ["Der Fremde", "die Gestalt"], "Figurenname_Vollständig": ""}}
{{"Nummer": 6, "Kapitel": 3, "Figurenname_Kapitel": "Elara", "Erwähnungen": ["Elara", "die Königin"], "Figurenname_Vollständig": ""}}
{{"Nummer": 7, "Kapitel": 3, "Figurenname_Kapitel": "Lord Valerius", "Erwähnungen": ["Lord Valerius", "der Fremde"], "Figurenname_Vollständig": ""}}
```

**Erwartete Ausgabe:**
```json
{{"Nummer": 0, "Kapitel": 1, "Figurenname_Kapitel": "Kaelan", "Figurenname_Vollständig": "Sir Kaelan"}}
{{"Nummer": 1, "Kapitel": 1, "Figurenname_Kapitel": "die Königin", "Figurenname_Vollständig": "Königin Elara"}}
{{"Nummer": 2, "Kapitel": 1, "Figurenname_Kapitel": "Dorfbewohner", "Figurenname_Vollständig": "Dorfbewohner"}}
{{"Nummer": 3, "Kapitel": 2, "Figurenname_Kapitel": "Sir Kaelan", "Figurenname_Vollständig": "Sir Kaelan"}}
{{"Nummer": 4, "Kapitel": 2, "Figurenname_Kapitel": "Königin Elara", "Figurenname_Vollständig": "Königin Elara"}}
{{"Nummer": 5, "Kapitel": 2, "Figurenname_Kapitel": "Der Fremde", "Figurenname_Vollständig": "Lord Valerius"}}
{{"Nummer": 6, "Kapitel": 3, "Figurenname_Kapitel": "Elara", "Figurenname_Vollständig": "Königin Elara"}}
{{"Nummer": 7, "Kapitel": 3, "Figurenname_Kapitel": "Lord Valerius", "Figurenname_Vollständig": "Lord Valerius"}}
```

---

**Hier ist die zu annotierende Figurenliste:**
{incomplete_response}
"""

# PRON = set("wer denen diese unsrer solche welche unsres unsern ich du er sie es wir ihr sie mich dich ihn sie es uns euch sie mir dir ihm ihr ihm uns euch ihnen mein dein sein ihr sein unser euer ihr meiner deiner seiner ihrer seiner unserer eurer ihrer meins deins seins ihres seins unseres eures ihres meinem deinem seinem ihrem seinem unserem eurem ihrem meinen deinen seinen ihren seinen unseren euren ihren meine deine seine ihre seine unsere eure ihre meinen deinen seinen ihren seinen unseren euren ihren sich man der die das die dem den dessen deren dessen deren welcher welche welches welchen welchem welcher welches welchen deren dessen dem den jener jene jenes jene jenem jener jenes jenen niemand jemand etwas nichts alle einige manche mehrere viele wenige andere beide jeder jedes jeden jedem jedes ein eine ein einer einem einer eins welcher welche welches worin worauf womit wofür wogegen worüber woran woraus wozu womit wer wem wen wessen was dessen das den".split())

def merge_entities(new_id, new_fullname, entities: List[Entity]) -> Entity:
    attribute_counter = collections.defaultdict(collections.Counter)
    for e in entities:
        attribute_counter['gender'].update(e.gender)
        attribute_counter['specialcase_entity'].update([tuple(sorted(e.specialcase_entity))])
        attribute_counter['borderline_entity'].update([tuple(sorted(e.borderline_entity))])

    merged_entity = Entity(
        id=new_id,
        fullname=new_fullname,
        gender=attribute_counter['gender'].most_common()[0][0],
        specialcase_entity=attribute_counter['specialcase_entity'].most_common()[0][0],
        borderline_entity=attribute_counter['borderline_entity'].most_common()[0][0]
    )

    return merged_entity


class MergePrompt(Prompt[List[Mention]], ABC):
    def __init__(self, tokens: pandas.Series, is_section_start: pandas.Series, mentions: List[Mention]):
        self.tokens = tokens
        self.is_section_start = is_section_start
        self.mentions = copy.deepcopy(mentions)
        self.section_id_ser = self.is_section_start.cumsum()


class BasicMergePrompt(MergePrompt):

    def __init__(self, tokens: pandas.Series, is_section_start: pandas.Series, mentions: List[Mention]):
        super().__init__(tokens, is_section_start, mentions)

        self.entities: Dict[Tuple[int, str], List[Tuple[Mention, Reference]]] = self.setup_entities()

        self.annotation_rows = None

    def setup_entities(self) -> Dict[Tuple[int, str], List[Tuple[Mention, Reference]]]:
        entities = collections.defaultdict(list)
        for mention in self.mentions:
            section_id = self.section_id_ser[mention.token_idx[0]]
            for ref in mention.references:
                entities[(int(section_id), ref.entity.id)].append((mention, ref))

        return dict(entities)

    def format_prompt(self) -> str:
        json_input_lines = []
        annotation_rows = []

        i = 0
        for (section_id, _),  references in sorted(self.entities.items(), key=lambda x: x[0][0]):
            entity = references[0][1].entity

            # drop any generic entities since these cannot co-refer anyway
            if 'generic' in entity.specialcase_entity:
                continue

            str_references = [
                ' '.join(self.tokens[mention.token_idx])
                for mention, reference in references
            ]

            entity_name = references[0][1].entity.fullname
            entity_id = references[0][1].entity.id

            # refs_with_count = Counter(s for s in str_references if s.lower() not in PRON).most_common()
            refs_with_count = Counter(s for s in str_references).most_common()
            json_input_lines.append(json.dumps({
                "Nummer": i,
                "Kapitel": section_id,
                "Figurenname_Kapitel": entity_name,
                "Erwähnungen": [x[0] for x in refs_with_count],
                "Figurenname_Vollständig": ""
            }, ensure_ascii=False))

            i = i + 1

            annotation_rows.append((section_id, entity_id))


        self.annotation_rows = annotation_rows

        incomplete_response = "\n".join(json_input_lines)
        return BASIC_MERGE_PROMPT.format(incomplete_response=incomplete_response)


    def decode(self, json_lines: list) -> List[Mention]:
        entity_translation_map: Dict[(int, str), str] = self._decode(json_lines)

        new_entities = collections.defaultdict(list)
        merged_entities = set()
        for (section_id, old_entity_id), entity_global_name in entity_translation_map.items():
            new_entities[entity_global_name].append((section_id, old_entity_id))
            merged_entities.add((section_id, old_entity_id))


        entity_counter = 0
        for entity_global_name, chapter_entities in new_entities.items():
            entity_counter += 1
            for (section_id, old_entity_id) in chapter_entities:
                entries = self.entities[(section_id, old_entity_id)]

                merged_entity = merge_entities(f'figur_{entity_counter:04d}', entity_global_name, [reference.entity for _, reference in entries])
                for _, reference in entries:
                    reference.entity = merged_entity

        for (section_id, old_entity_id) in self.entities.keys() - merged_entities:
            entity_counter += 1
            entries = self.entities[(section_id, old_entity_id)]
            for _, reference in entries:
                reference.entity.id = f'figur_{entity_counter:04d}'


        return self.mentions


    def _decode(self, json_lines: list):
        entity_translation_map = {}
        for line in json_lines:
            try:
                id_ = line['Nummer']
                entity_global_name = line['Figurenname_Vollständig']

                section_id, entity_id = self.annotation_rows[id_]

                entity_translation_map[(section_id, entity_id)] = entity_global_name
            except Exception as e:
                logger.warning(f'Error decoding line {line!r}: {e}')

        return entity_translation_map

