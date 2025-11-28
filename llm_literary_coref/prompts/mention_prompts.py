import itertools
import logging
from abc import abstractmethod
from typing import Literal, Optional, List, Tuple, Dict, Generic, TypeVar

import pandas
from pydantic import BaseModel, TypeAdapter

from llm_literary_coref.mention import Mention, Entity, Reference
from llm_literary_coref.prompts.prompt import Prompt

logger = logging.getLogger(__name__)

BASIC_MENTION_PROMPT = """
# Entitätsannotation in literarischen Texten

## Aufgabe
Annotiere JEDE markierte Entitäts-Erwähnung im Text. Für jede Erwähnung (jede Zeile im Input B) musst du das "Annotation"-Feld als ein Array von Objekten befüllen. Halte dich exakt an die vorgegebenen Kernprinzipien, die Struktur und die Logik der Sonderfälle.

## Zentrales Konzept: Die Entität
Eine **Entität** ist das konzeptionelle Ding, auf das im Text verwiesen wird. Für diese Aufgabe gibt es zwei Arten von Entitäten:
1.  **Einzelfigur:** Ein einzelner menschlicher oder menschenähnlicher Charakter.
2.  **Gruppe als Einheit:** Eine Gruppe, die im Text als ein zusammenhängender Akteur handelt ODER als eine spezifische, aber nicht in ihre Mitglieder aufgelöste Menge von Personen erscheint (siehe "Fall 1" bei den Gruppenregeln).

## Input
Du erhältst:
1.  Den vollständigen Text (A).
2.  Eine Liste der zu annotierenden Erwähnungen (B) als JSON Lines. Jede Erwähnung ist ein einzelnes Wort.

## Kernprinzipien der Annotation
1.  **Konsistenz der `entity_id`:** Die wichtigste Regel! Verwende für dieselbe Entität (egal ob Einzelfigur oder Gruppe als Einheit) über den gesamten Text hinweg **IMMER** die exakt gleiche `entity_id`.
2.  **Namenswahl für `entity_id`:**
    -   Bei Einzelfiguren mit Eigennamen: Verwende den Vornamen.
    -   Bei Einzelfiguren ohne Namen: Erstelle eine kurze, präzise, textnahe Umschreibung (z.B. "der alte Mann", "die Tante").
    -   Bei Gruppen als Einheit (Fall 1): Verwende eine passende Gruppenbezeichnung (z.B. "Familie Armbruster", "die Wachen").
3.  **Referenz-Logik:** Jede Erwähnung referenziert eine oder mehrere Entitäten. Deine Aufgabe ist es, diese im "Annotation"-Array vollständig und korrekt abzubilden.

## Attribute im Annotationsobjekt
-   `gender`: (String, nur bei erster Nennung) `m`, `f`, `nb` (nicht-binär), `u` (unbekannt).
-   `specialcase_entity`: (**Array von Strings**, nur bei erster Nennung) Sonderfälle, die die Natur der Entität beschreiben. Alle diese Attribute sind unabhängig voneinander und können parallel auftreten.
    -   `nonfact`: nicht-faktische Entität, siehe unten.
    -   `group`: Gruppe, siehe unten.
    -   `generic`: Generische Entität, siehe unten.
    -   Falls keine dieser Attribute zutrifft, dann setze bei der ersten Nennung `"specialcase_entity": []`. Beachte, dass diese in der ersten Nennung angegebenen Attribute auch für alle folgenden Nennungen gelten müssen, da diese Attribute sich auf die Entität beziehen.
-   `specialcase_mention`: (**Array von Strings**, opt.) Sonderfälle, die die **Referenz auf die Entität** beschreiben.
    -   `figurative`: Uneigentliche Rede (Metapher, Vergleich).
    -   `part`: Es wird über ein Körperteil auf die Figur referenziert.
    
### Nicht-faktische Entitäten

- Frage dich: Existiert diese Entität zum Zeitpunkt der Erwähnung *real* in der erzählten Welt? Oder ist sie nur **hypothetisch, möglich, gewünscht** oder wird sie **negiert**?
    -   *Beispiele:* "für einen **Dritten** wäre Platz", "Er wünschte sich eine **Tochter**", "keinen besseren **Schüler**".
    -   **Falls JA (nicht real):** Füge `"nonfact"` zum `specialcase_entity`-Array hinzu.

### Generische Entitäten

- Frage dich: Bezieht sich die Erwähnung auf eine konkrete Person, welche in der erzählten Welt existiert?
  - Falls JA: Dann ist die Entität *kein* `generic`.
- Frage dich: Bezieht sich die Erwähnung auf eine *abstrakte Kategorie* oder eine *allgemeine, gesetzmäßige Aussage* über eine Art von Mensch anstatt auf ein spezifisches, individuelles Wesen der erzählten Welt?
  Lässt sich die Entität in der Form "XXX im Allgemeinen" beschreiben?
  - Beispiel: "Männer [im Allgemeinen] denken mehr auf das Einzelne...", "Ein Ritter sagt [im Allgemeinen] immer die Wahrheit"
  - Falls JA: Dann ist die Entität *eine* `generic`. Füge `generic` zum Array hinzu und benenne die Figur als "XXX im Allgemeinen". Diese ist oft durch ein Plural verbalisiert, wie im Fall "die Männer", in diesem Fall also `specialcase_entity: ["generic", "group"]`.
    Beachte, dass auch diese generischen Entitäten ein Gender zugewiesen werden muss.
- Frage dich: Bezieht sich die Erwähnung auf eine *konkrete, aber anonyme oder nicht weiter aufgeschlüsselte Menge von Personen* in der Welt der Erzählung?
  - Beispiel: "meine Freunde", "die Pächter", "die Wachen am Tor"
  - falls JA:  Dann ist die Entität *kein* `generic`. Sie ist einfach eine `group`. `specialcase_entity: ["group"]`. Sie repräsentiert eine reale, aber unbestimmte Ansammlung von Leuten im Text.
  
Beachte, dass generische Entitäten nicht koreferent sein können. Markiere eine Entität nur als `generic` wenn du dir ganz sicher bist.
Beachte, dass sich diese Entscheidung auf die gesamte Entität bezieht. Überprüfe daher vor deiner Entscheidung alle Erwähnungen der Entität.

  

### Umgang mit Gruppen-Erwähnungen
**Fall 1: Nicht-Auflösbare Gruppe (z.B. "Freunde", "Landleute", "Familie")**
-   **Logik:** Dies ist der Standardfall für Plurale, die sich **nicht** auf direkt im Kontext genannte Individuen zurückführen lassen. Es ist eine spezifische, aber anonyme Menge von Personen.
-   **Umsetzung:** Das `Annotation`-Array enthält **ein einziges Objekt**.
    -   Dieses Objekt bekommt eine eigene Gruppen-`entity_id` (z.B. "manche tätige Freunde").
    -   Bei ihrer ersten Nennung, setze `specialcase_entity: ["group"]`.
    -   Wenn diese Gruppe zusätzlich eine allgemeine Kategorie darstellt (siehe Test oben), dann setze `specialcase_entity: ["generic", "group"]`.

**Fall 2: Auflösbare Gruppe (z.B. plurale Pronomen wie "sie", "ihnen" oder "beide")**
-   **Logik:** Die Erwähnung verweist klar auf mehrere, **individuelle Entitäten, die im unmittelbaren Kontext bekannt sind**.
-   **Umsetzung:** Das `Annotation`-Array enthält **mehrere Objekte**, eines für jede referenzierte Entität.
    -   Jedes Objekt enthält die `entity_id` des jeweiligen Mitglieds.
    -   Erfinde keine neuen Einzelfiguren, wenn die Mitglieder nicht aus dem Text hervorgehen. Wenn du die Mitglieder nicht kennst, ist es Fall 1.

## Ausgabeformat
-   Gib EXAKT so viele Zeilen aus, wie Du im Input (B) erhalten hast. KEINE Zeilen überspringen.
-   Das Format ist JSON Lines (ein komplettes JSON-Objekt pro Zeile).
-   Gib NUR die aktualisierte Liste B zurück.
-   Keine Code-Fences, Kommentare oder Erklärungen in der Ausgabe.

## Beispiele

**Beispiel-Text:**
In der Frühe fragte [Madlen][1] [ihre][2] [Tochter][3] . Vor dem Schloss warteten die [Wachen][4] . Danach kamen die zwei [Jungen][5] . Gemeinsam gingen [sie][6] ins Haus . Charlotte sagte : " [Männer][7] sind so . "

**Input B:**
{{"ID": 1, "Position": 4, "Text": "Madlen", "Annotation": []}}
{{"ID": 2, "Position": 5, "Text": "ihre", "Annotation": []}}
{{"ID": 3, "Position": 6, "Text": "Tochter", "Annotation": []}}
{{"ID": 4, "Position": 13, "Text": "Wachen", "Annotation": []}}
{{"ID": 5, "Position": 19, "Text": "Jungen", "Annotation": []}}
{{"ID": 6, "Position": 23, "Text": "sie", "Annotation": []}}
{{"ID": 7, "Position": 31, "Text": "Männer", "Annotation": []}}

**Erwartete Ausgabe:**
{{"ID": 1, "Position": 4, "Text": "Madlen", "Annotation": [{{"entity_id": "Madlen", "gender": "f"}}]}}
{{"ID": 2, "Position": 5, "Text": "ihre", "Annotation": [{{"entity_id": "Madlen"}}]}}
{{"ID": 3, "Position": 6, "Text": "Tochter", "Annotation": [{{"entity_id": "Tochter_Madlens", "gender": "f"}}]}}
{{"ID": 4, "Position": 13, "Text": "Wachen", "Annotation": [{{"entity_id": "die Wachen", "gender": "u", "specialcase_entity": ["group"]}}]}}
{{"ID": 5, "Position": 19, "Text": "Jungen", "Annotation": [{{"entity_id": "Junge1", "gender": "m"}}, {{"entity_id": "Junge2", "gender": "m"}}]}}
{{"ID": 6, "Position": 23, "Text": "sie", "Annotation": [{{"entity_id": "Junge1"}}, {{"entity_id": "Junge2"}}]}}
{{"ID": 7, "Position": 31, "Text": "Männer", "Annotation": [{{"entity_id": "Männer im Allgemeinen", "gender": "m", "specialcase_entity": ["group", "generic"]}}]}}

---

**Hier ist der zu annotierende Text:**
Input A:
{doc_formatted}

Input B:
{incomplete_response}
"""

T = TypeVar('T')


class MentionPrompt(Generic[T], Prompt[List[Mention]]):

    def __init__(self, tokens: pandas.Series, mention_spans: List[Mention]):
        self.tokens = tokens
        self.mention_spans = mention_spans
        self.formatted_input = None
        self.mention_span_map = None
        self.span_annotations = None

    def prepare_input(self, tokens: pandas.Series, mention_spans: List[Mention]) -> Tuple[str, dict[int, Mention], list]:
        input_text_df = tokens.copy()

        mention_span_map = {}
        span_annotations = []

        index_mapper = pandas.Series(range(len(tokens)), index=input_text_df.index)
        for id_, mention_span in enumerate(sorted(mention_spans, key=lambda x: x.token_idx[0])):
            begin = min(mention_span.token_idx)
            end = max(mention_span.token_idx)
            input_text_df.loc[begin] = '[' + input_text_df.loc[begin]
            input_text_df.loc[end] = str(input_text_df.loc[end]) + f'][{id_ + 1}]'
            pos = index_mapper.loc[begin]

            mention_span_map[id_ + 1] = mention_span
            span_annotations.append({
                "ID": id_ + 1,
                "Position": int(pos),
                "Text": " ".join(tokens.loc[begin:end]),
                "Annotation": [],
            })

        formatted_input = ' '.join(input_text_df)

        return formatted_input, mention_span_map, span_annotations

    def format_prompt(self):
        (formatted_input,
         self.mention_span_map,
         span_annotations) = self.prepare_input(self.tokens, self.mention_spans)

        return self.prompt_template(formatted_input, span_annotations)

    @abstractmethod
    def prompt_template(self, formatted_input: str, span_annotations: list) -> str:
        raise NotImplementedError

    def decode(self, json_lines: list) -> List[Mention]:
        extended_output: List[Tuple[Mention, T]] = []
        for line in json_lines:
            try:
                id_ = line['ID']
                annotated_span = self.mention_span_map[id_]

                annotation_raw = line['Annotation']
                annotation_parsed = self.decode_annotation(annotation_raw)
                extended_output.append((annotated_span, annotation_parsed))
            except Exception as e:
                logger.warning(f'Error decoding line {line!r}: {e}')

        return self.decode_annotations(extended_output)

    @abstractmethod
    def decode_annotation(self, annotation_raw: any) -> T:
        pass

    @abstractmethod
    def decode_annotations(self, extended_output: List[Tuple[Mention, T]]) -> List[Mention]:
        pass


class BasicAnnotationReference(BaseModel):
    entity_id: str
    gender: Optional[Literal["m", "f", "nb", "o", "u"]] = None
    specialcase_entity: Optional[List[Literal["group", "generic", "nonfact", "possible_identity"]]] = None
    specialcase_mention: Optional[List[Literal["figurative", "part"]]] = None

BasicAnnotationObject = TypeAdapter(list[BasicAnnotationReference])


class MentionPromptBasic(MentionPrompt[List[BasicAnnotationReference]]):

    entity_fields = ['gender', 'specialcase_entity']

    def __init__(self, tokens: pandas.Series, mention_spans: List[Mention]):
        super().__init__(tokens, mention_spans)
        self.entities = None

    def prompt_template(self, formatted_input: str, span_annotations: list) -> str:
        return BASIC_MENTION_PROMPT.format(doc_formatted=formatted_input, incomplete_response=span_annotations)

    def decode_annotations(self, parsed_output: List[Tuple[Mention, List[Tuple[Mention, BasicAnnotationObject]]]]) -> List[Mention]:
        entities = self.get_entities(parsed_output)
        mentions = list(self.get_mentions(parsed_output, entities))
        return mentions

    def decode_annotation(self, annotation_raw: any) -> BasicAnnotationObject:
        return BasicAnnotationObject.validate_python(annotation_raw)

    def get_entities(self, llm_output: List[Tuple[Mention, BasicAnnotationObject]]) -> Dict[str, Entity]:
        entities = {}

        all_entity_annotations = []
        for mention, annotations in llm_output:
            for annotation in annotations:
                all_entity_annotations.append((mention, annotation))

        entity_id_counter = 0
        for entity_id, entity_annotations in itertools.groupby(sorted(all_entity_annotations, key=lambda x: x[1].entity_id),
                                                              key=lambda x: x[1].entity_id):
            entity_annotations = list(sorted(entity_annotations, key=lambda x: x[0].token_idx[0]))

            fields = {}
            for entity_field in self.entity_fields:
                values = [getattr(anno, entity_field) for _, anno in entity_annotations]
                values = [value for value in values if value is not None]
                value = values[0] if values else None

                if value is None:
                    if entity_field == 'gender':
                        logger.warning(f'for response entity {entity_id!r} no gender was specified! Defaulting to u')
                        fields[entity_field] = 'u'
                    elif entity_field == 'specialcase_entity':
                        logger.warning(f'for response entity {entity_id!r} no specialcase_entity was specified! Defaulting to empty list')
                        fields[entity_field] = []
                else:
                    fields[entity_field] = value

            entity = Entity(
                id=(entity_id_counter := entity_id_counter + 1),
                fullname=entity_id,
                gender=fields['gender'],
                specialcase_entity=fields.get('specialcase_entity', []),
                borderline_entity=[]
            )

            entities[entity_id] = entity

        return entities

    def get_mentions(self, llm_output: List[Tuple[Mention, BasicAnnotationObject]], entities: Dict[str, Entity]):
        for line in llm_output:
            mention, annotations = line
            references = []
            for annotation in annotations:
                entity_id = annotation.entity_id
                entity = entities[entity_id]
                ref = Reference(
                    entity=entity,
                    specialcase_reference=annotation.specialcase_mention or [],
                    borderline_reference=[],
                )
                references.append(ref)

            if len(references) == 0:
                continue

            yield mention.with_references(references)
