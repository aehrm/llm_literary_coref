import itertools
import json
import logging
import re
from abc import abstractmethod
from typing import Literal, Optional, List, Tuple, Dict, Generic, TypeVar

import pandas
from pydantic import BaseModel, TypeAdapter

from llm_literary_coref.mention import Mention, Entity, Reference
from llm_literary_coref.prompts.prompt import Prompt
from llm_literary_coref.util import make_generic_entity_factory, split_generics_into_singletons

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
-   `gender`: (String, nur bei erster Nennung) `m`, `f`, `nb` (nicht-binär), `u` (unbekannt), `mf` (nur bei Gruppen, falls diese aus männlichen und weiblichen Mitgliedern besteht).
-   `specialcase_entity`: (**Array von Strings**, nur bei erster Nennung) Sonderfälle, die die Natur der Entität beschreiben. Alle diese Attribute sind unabhängig voneinander und können parallel auftreten.
    -   `nonfact`: nicht-faktische Entität, siehe unten.
    -   `group`: Gruppe, siehe unten.
    -   `generic`: Generische Entität, siehe unten.
    -   Falls keine dieser Attribute zutrifft, dann setze bei der ersten Nennung `"specialcase_entity": []`. Beachte, dass diese in der ersten Nennung angegebenen Attribute auch für alle folgenden Nennungen gelten müssen, da diese Attribute sich auf die Entität beziehen.
-   `specialcase_mention`: (**Array von Strings**, opt.) Sonderfälle, die die **Referenz auf die Entität** beschreiben.
    -   `figurative`: Uneigentliche Rede (Metapher, Vergleich).
    -   `part`: Es wird über ein Teil des Körpers auf die Figur referenziert (inklusive "Stimme", Umschreibung wie "Figur", "Gestalt", "Seele").
    
## Logik der Sonderfälle: Eine Entscheidungshierarchie

**WICHTIG:** Gehe bei jeder Erwähnung diese Schritte in der angegebenen Reihenfolge durch. Annotiere gemäß der **ersten zutreffenden Regel** und ignoriere die folgenden für diese eine Erwähnung, außer es wird explizit anders angegeben (wie bei dualen Referenzen). Die korrekte Unterscheidung von spezifischen, nicht-faktischen und generischen Entitäten ist entscheidend.

### Schritt 1: Prüfung auf spezifische Referenz (Höchste Priorität)
*Frage dich:* Kann diese Erwähnung auf eine **konkrete, bereits bekannte Figur** im Text zurückgeführt werden, selbst wenn ein allgemeiner Begriff (wie "der Mann", "der Schuldige") oder das Pronomen "man" verwendet wird?
-   **Typischer Fall:** Eine Figur spricht über sich selbst oder eine andere Figur in verallgemeinernder Weise.
-   **Beispiel 1:** Hans reflektiert über seine eigene Schuld und sagt: "...zeigt sich **dem Schuldigen** wie eine strenge Richterin". Hier ist "dem Schuldigen" **keine** generische Entität, sondern eine Referenz auf `Hans`.
-   **Beispiel 2:** Eine Figur sagt "wenn **man** glaubte...". Prüfe scharf, ob `man` hier nicht für "ich" (also die sprechende Figur) steht. 
-   **Anweisung:** Wenn ja, löse die Koreferenz zur spezifischen Figur auf. Diese Entität ist **NICHT `generic` und NICHT `nonfact`**.

### Schritt 2: Prüfung auf nicht-faktische Entität (Mittlere Priorität)
*Frage dich:* Wenn die Erwähnung sich nicht auf eine bekannte Figur bezieht, handelt es sich dann um eine **hypothetische, gewünschte, negierte oder als Beispiel genannte Einzelfigur oder eine kleine, definierte Gruppe?** Existiert diese Entität zum Zeitpunkt der Erwähnung nicht *real* in der erzählten Welt, wird aber als individuelles Konzept behandelt?
-   **Typischer Fall:** Gedankenexperimente, Wünsche, Pläne, Warnungen.
-   **Beispiel 1:** "Anna wünschte sich eine **Tochter**". Diese Tochter existiert nicht, ist aber eine hypothetische Einzelperson. -> `specialcase_entity: ["nonfact"]`.
-   **Beispiel 2:** In einem philosophischen Dialog wird gesagt: "was würde ein **Grieche**... einem **Chineser** antworten?". Diese Personen sind keine allgemeinen Kategorien, sondern **hypothetische Beispiele** in einer Argumentation. -> `specialcase_entity: ["nonfact"]`.
-   **Anweisung:** Wenn ja, erstelle eine neue Entität und füge `"nonfact"` zum `specialcase_entity`-Array hinzu. Diese Entität ist **NICHT `generic`**.

### Schritt 3: Prüfung auf generische Entität (Niedrigste Priorität)
*Frage dich:* Nur wenn Schritt 1 und 2 NICHT zutreffen: Bezieht sich die Erwähnung auf eine **abstrakte Kategorie von Menschen** oder wird eine allgemeingültige, gesetzmäßige Aussage über eine Art von Mensch gemacht? Lässt sich die Entität als "XXX im Allgemeinen" beschreiben, ohne dass sie ein spezifisches oder hypothetisches Individuum im Kontext meint?
-   **Typischer Fall:** Sentenzen, allgemeine Aussagen, die sich nicht auf den unmittelbaren Kontext zurückführen lassen.
-   **Beispiel:** "Ein **König** muss stets gerecht sein." (Wenn dies als allgemeine Regel und nicht bezogen auf einen bestimmten König gesagt wird).
-   **Anweisung:** Nur in diesem Fall, füge `"generic"` zum `specialcase_entity`-Array hinzu. Der Name der Entität muss von der Form "XXX im Allgemeinen" sein.

### Sonderfall: Duale Referenz (Generisch + Spezifisch)
*Frage dich:* Spricht eine Figur eine allgemeine Gruppe an ("ihr Frauen", "ihr Elenden"), meint damit aber **gleichzeitig** und erkennbar auch eine **spezifische Figur** im Raum?
-   **Typischer Fall:** Eine Apostrophe (Anrede), die eine allgemeine Aussage mit einer direkten, persönlichen Ansprache verbindet.
-   **Beispiel:** Gustav sagt zu Sophie: "**Ihr** unglücklichen **Weiber**! wie könnt **ihr** so thöricht seyn...".
-   **Anweisung:** In diesem seltenen Fall soll die Erwähnung **mehrere Annotationsobjekte** im Array erhalten, da es sich um eine Gruppe von Fall 2 handelt: Eines für die generische Entität ("Weiber im Allgemeinen") und eines für jede spezifische Figur, die mitgemeint ist (hier: "Sophie").

### Handhabung von Gruppen (leicht überarbeitet für Klarheit)

**Fall 1: Gruppe als Einheit (Eine Annotation)**
-   **Logik:** Erwähnungen im Plural, die sich auf eine **konkrete, aber anonyme oder nicht weiter aufgeschlüsselte Menge von Personen** in der Welt der Erzählung beziehen. Auch wenn die Mitglieder unbekannt sind, ist es eine spezifische Gruppe im Text. *Beispiele: "meine Freunde", "die Pächter", "die Wachen am Tor".*
-   **Umsetzung:** Das `Annotation`-Array enthält **ein einziges Objekt**.
    -   Dieses Objekt bekommt eine eigene Gruppen-`entity_id` (z.B. "manche tätige Freunde").
    -   Bei ihrer ersten Nennung, setze `specialcase_entity: ["group"]`.

**Fall 2: Auflösbare Gruppe (Mehrere Annotationen)**
-   **Logik:** Die Erwähnung (oft ein Pronomen wie "sie", "ihnen" oder "beide") verweist klar auf **mehrere, individuelle Entitäten, die im unmittelbaren Kontext bekannt sind**.
-   **Umsetzung:** Das `Annotation`-Array enthält **mehrere Objekte**, eines für jede referenzierte Entität.
    -   Jedes Objekt enthält die `entity_id` des jeweiligen Mitglieds.
    -   Erfinde keine neuen Einzelfiguren. Wenn die Mitglieder nicht klar aus dem Kontext hervorgehen, ist es Fall 1.
    - Beachte, dass in diesem Fall das Feld "Annotation" aus mehreren Objekten besteht. Auf keinen Fall darf das Feld "entity_id" aus einer Liste von Namen bestehen.

## Ausgabeformat
-   Gib EXAKT so viele Zeilen aus, wie Du im Input (B) erhalten hast. KEINE Zeilen überspringen.
-   Das Format ist JSON Lines (ein komplettes und valides JSON-Objekt pro Zeile).
-   Gib NUR die aktualisierte Liste B zurück.
-   Keine Code-Fences, Kommentare oder Erklärungen in der Ausgabe.

## Beispiele

**Beispiel-Text:**
[Hans][1] und [Anna][2] saßen im Garten. [Er][3] nannte [sie][4] liebevoll seine [Sonne][5]. Anna wünschte sich eine [Tochter][6]. Hans meinte, ein [König][7] müsse stets gerecht sein. Später gingen [sie][8] gemeinsam ins Haus.

**Input B:**
{{"ID": 1, "Position": 0, "Text": "Hans", "Annotation": []}}
{{"ID": 2, "Position": 2, "Text": "Anna", "Annotation": []}}
{{"ID": 3, "Position": 7, "Text": "Er", "Annotation": []}}
{{"ID": 4, "Position": 9, "Text": "sie", "Annotation": []}}
{{"ID": 5, "Position": 12, "Text": "Sonne", "Annotation": []}}
{{"ID": 6, "Position": 18, "Text": "Tochter", "Annotation": []}}
{{"ID": 7, "Position": 23, "Text": "König", "Annotation": []}}
{{"ID": 8, "Position": 31, "Text": "sie", "Annotation": []}}

**Erwartete Ausgabe:**
{{"ID": 1, "Position": 0, "Text": "Hans", "Annotation": [{{"entity_id": "Hans", "gender": "m"}}]}}
{{"ID": 2, "Position": 2, "Text": "Anna", "Annotation": [{{"entity_id": "Anna", "gender": "f"}}]}}
{{"ID": 3, "Position": 7, "Text": "Er", "Annotation": [{{"entity_id": "Hans"}}]}}
{{"ID": 4, "Position": 9, "Text": "sie", "Annotation": [{{"entity_id": "Anna"}}]}}
{{"ID": 5, "Position": 12, "Text": "Sonne", "Annotation": [{{"entity_id": "Anna", "specialcase_mention": ["figurative"]}}]}}
{{"ID": 6, "Position": 18, "Text": "Tochter", "Annotation": [{{"entity_id": "gewünschte Tochter", "gender": "f", "specialcase_entity": ["nonfact"]}}]}}
{{"ID": 7, "Position": 23, "Text": "König", "Annotation": [{{"entity_id": "ein König im Allgemeinen", "gender": "m", "specialcase_entity": ["nonfact", "generic"]}}]}}
{{"ID": 8, "Position": 31, "Text": "sie", "Annotation": [{{"entity_id": "Hans"}}, {{"entity_id": "Anna"}}]}}

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

        annotations_in_json = '\n'.join(json.dumps(s, ensure_ascii=True) for s in span_annotations)

        return self.prompt_template(formatted_input, annotations_in_json)

    @abstractmethod
    def prompt_template(self, formatted_input: str, span_annotations: str) -> str:
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
    gender: Optional[Literal["m", "f", "nb", "o", "u", "mf"]] = None
    specialcase_entity: Optional[List[Literal["group", "generic", "nonfact", "possible_identity"]]] = None
    specialcase_mention: Optional[List[Literal["figurative", "part"]]] = None

BasicAnnotationObject = TypeAdapter(list[BasicAnnotationReference])


class MentionPromptBasic(MentionPrompt[List[BasicAnnotationReference]]):

    entity_fields = ['gender', 'specialcase_entity']

    def __init__(self, tokens: pandas.Series, mention_spans: List[Mention]):
        super().__init__(tokens, mention_spans)
        self.entities = None

    def prompt_template(self, formatted_input: str, span_annotations: str) -> str:
        return BASIC_MENTION_PROMPT.format(doc_formatted=formatted_input, incomplete_response=span_annotations)

    def decode_annotations(self, parsed_output: List[Tuple[Mention, List[Tuple[Mention, BasicAnnotationObject]]]]) -> List[Mention]:
        entities = self.get_entities(parsed_output)
        mentions = list(self.get_mentions(parsed_output, entities))

        # split generic entities
        generic_entity_factory = make_generic_entity_factory()
        split_generics_into_singletons(mentions, generic_entity_factory)

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
        for entity_name, entity_annotations in itertools.groupby(sorted(all_entity_annotations, key=lambda x: x[1].entity_id),
                                                              key=lambda x: x[1].entity_id):
            entity_annotations = list(sorted(entity_annotations, key=lambda x: x[0].token_idx[0]))

            fields = {}
            for entity_field in self.entity_fields:
                values = [getattr(anno, entity_field) for _, anno in entity_annotations]
                values = [value for value in values if value is not None]
                value = values[0] if values else None

                if value is None:
                    if entity_field == 'gender':
                        logger.warning(f'for response entity {entity_name!r} no gender was specified! Defaulting to u')
                        fields[entity_field] = 'u'
                    elif entity_field == 'specialcase_entity':
                        logger.warning(f'for response entity {entity_name!r} no specialcase_entity was specified! Defaulting to empty list')
                        fields[entity_field] = []
                else:
                    fields[entity_field] = value

            entity = Entity(
                id=(entity_id_counter := entity_id_counter + 1),
                fullname=re.sub(r'[^\w\s]', '', entity_name),
                gender=fields['gender'],
                specialcase_entity=fields.get('specialcase_entity', []),
                borderline_entity=[]
            )

            entities[entity_name] = entity

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
