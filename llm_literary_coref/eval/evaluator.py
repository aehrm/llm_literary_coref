import collections
import io
from contextlib import redirect_stdout
from typing import Optional, Literal, List

from scipy.optimize import linear_sum_assignment
import numpy as np

from llm_literary_coref.eval.metrics import mention_metric, lea_metric, bcubed_metric, ceafe_metric, ceafm_metric, \
    attribute_metric, Clusters, EvalMention
from llm_literary_coref.mention import Mention



def mentions_to_clusters(mentions: List[Mention], doc_id: str,
                                   ignore_spans: Optional[set] = None) -> Clusters:
    """Convert mentions to clusters, ignoring specified spans."""
    clusters = collections.defaultdict(list)
    for mention in mentions:
        span = (mention.token_idx[0], mention.token_idx[-1])
        if ignore_spans is not None and span in ignore_spans:
            continue
        eval_m = EvalMention(doc_id, mention.token_idx[0], mention.token_idx[-1])
        for ref in mention.references:
            clusters[ref.entity.id].append(eval_m)
    return dict(clusters)



def get_mention_assignments(inp_clusters: Clusters, out_clusters: Clusters) -> dict[EvalMention, List[str]]:
    """Map mentions to their entity IDs in out_clusters."""
    out_dic = collections.defaultdict(list)
    for eid, mentions in out_clusters.items():
        for m in mentions:
            out_dic[m].append(eid)

    return {m: out_dic[m] for mentions in inp_clusters.values()
            for m in mentions if m in out_dic}


def get_self_assignments(clusters: Clusters) -> dict[EvalMention, List[str]]:
    """Map each mention to its own entity IDs."""
    result = collections.defaultdict(list)
    for eid, mentions in clusters.items():
        for m in mentions:
            result[m].append(eid)
    return dict(result)


def compute_entity_mapping(key_clusters: Clusters, sys_clusters: Clusters) -> dict[str, str]:
    """Compute optimal entity mapping using phi4 similarity. Returns entity_id -> entity_id."""
    if not key_clusters or not sys_clusters:
        return {}

    key_ids = list(key_clusters.keys())
    sys_ids = list(sys_clusters.keys())

    phi4 = lambda c1, c2: 2 * len(set(c1) & set(c2)) / (len(c1) + len(c2)) if c1 and c2 else 0
    scores = np.zeros((len(key_ids), len(sys_ids)))
    for i, kid in enumerate(key_ids):
        for j, sid in enumerate(sys_ids):
            scores[i, j] = phi4(key_clusters[kid], sys_clusters[sid])

    row_ind, col_ind = linear_sum_assignment(-scores)
    return {key_ids[i]: sys_ids[j] for i, j in zip(row_ind, col_ind) if scores[i, j] > 0}


def get_generic_spans(mentions: List[Mention]) -> set:
    return {(m.token_idx[0], m.token_idx[-1]) for m in mentions
            for ref in m.references if 'generic' in ref.entity.specialcase_entity}


def get_singleton_spans(mentions: List[Mention]) -> set:
    entity_size = collections.Counter(ref.entity.id for m in mentions for ref in m.references)
    return {(m.token_idx[0], m.token_idx[-1]) for m in mentions
            for ref in m.references if entity_size[ref.entity.id] == 1}



class Scorer:

    def __init__(self, beta: float = 1.0):
        self.p_num = self.p_den = self.r_num = self.r_den = 0
        self.beta = beta
        self.doc_scores: dict[str, tuple[float, float, float, float]] = {}

    def update(self, scores: tuple[float, float, float, float], doc_id: Optional[str] = None):
        pn, pd, rn, rd = scores
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd
        if doc_id is not None:
            self.doc_scores[doc_id] = scores

    @property
    def precision(self) -> float:
        return self.p_num / self.p_den if self.p_den else 0

    @property
    def recall(self) -> float:
        return self.r_num / self.r_den if self.r_den else 0

    @property
    def f1(self) -> float:
        p, r, b = self.precision, self.recall, self.beta
        return (1 + b * b) * p * r / (b * b * p + r) if (p + r) else 0

    def support_key(self) -> float:
        return self.r_den

    def support_response(self) -> float:
        return self.p_den

    @staticmethod
    def _compute_prf(pn, pd, rn, rd, beta=1.0) -> dict:
        p = pn / pd if pd else 0
        r = rn / rd if rd else 0
        f = (1 + beta * beta) * p * r / (beta * beta * p + r) if (p + r) else 0
        return {'precision': p, 'recall': r, 'f1': f, 'support_key': rd, 'support_response': pd}

    def results_aggregated(self) -> dict:
        return self._compute_prf(self.p_num, self.p_den, self.r_num, self.r_den, self.beta)

    def results_by_doc(self) -> dict[str, dict]:
        return {doc_id: self._compute_prf(*scores, self.beta) 
                for doc_id, scores in self.doc_scores.items()}

    def results(self):
        return {'aggregated': self.results_aggregated()} | self.results_by_doc()



CLUSTER_METRICS = {
    'mentions': mention_metric,
    'lea': lea_metric,
    'bcub': bcubed_metric,
    'ceafe': ceafe_metric,
    'ceafm': ceafm_metric,
}

ENTITY_ATTRIBUTES = {
    'gender:m': lambda e: e.gender == 'm',
    'gender:f': lambda e: e.gender == 'f',
    'gender:nb': lambda e: e.gender == 'nb',
    'gender:mf': lambda e: e.gender == 'mf',
    'gender:u': lambda e: e.gender == 'u',
    'generic': lambda e: 'generic' in e.specialcase_entity,
    'group': lambda e: 'group' in e.specialcase_entity,
    'nonfact': lambda e: 'nonfact' in e.specialcase_entity,
}


MENTION_ATTRIBUTES = {
    'part': lambda refs: any('part' in r.specialcase_reference for r in refs),
    'figurative': lambda refs: any('figurative' in r.specialcase_reference for r in refs),
    'generic': lambda refs: any('generic' in r.entity.specialcase_entity for r in refs),
}



class Evaluator:

    def __init__(self):
        self.scorers = {
            f"clusters_{subset}": {
                k: Scorer() for k in CLUSTER_METRICS.keys()
            } for subset in ["all", "nogeneric", "nosingletons"]
        } | {
            f"entity_attributes_{subset}_{restrict_on_matches}": {
                k: Scorer() for k in ENTITY_ATTRIBUTES.keys()
            } for subset in ["all", "nogroup", "nogroupnosingletons"]
            for restrict_on_matches in ["unrestricted", "restrictonmatch"]
        } | {
            "mention_attributes": {
                k: Scorer() for k in MENTION_ATTRIBUTES.keys()
            }
        }

    def report(self, as_dict=False, print_individual_doc_scores=False, print_support: Optional[Literal["all", "key", "response"]] = None) -> dict | str:
        if as_dict:
            return {group_name: {scorer_name: scorer.results() for scorer_name, scorer in group.items()} for group_name, group in self.scorers.items()}

        f = io.StringIO()
        with redirect_stdout(f):
            def _print_scorer_results(scorer_dict):
                support_str = {
                    "all": "Support (key/sys)",
                    "key": "Support (key)",
                    "response": "Support (sys)",
                }[print_support] if print_support else None

                if support_str:
                    print(f"{'metric':12s} {'P':>7s} {'R':>7s} {'F1':>7s}  {support_str:>20s}")
                else:
                    print(f"{'metric':12s} {'P':>7s} {'R':>7s} {'F1':>7s}")

                docs = ["aggregated"]
                if print_individual_doc_scores:
                    docs.extend(list(scorer_dict.values())[0].results_by_doc().keys())
                for doc in docs:
                    if len(docs) > 1:
                        print(f" {doc}")
                    for name, scorer in scorer_dict.items():
                        vals = scorer.results()[doc]
                        if support_str:
                            support_val = {
                                "all": f"{vals['support_key']}/{vals['support_response']}",
                                "key": str(vals['support_key']),
                                "response": str(vals['support_response']),
                            }[print_support]
                            print(f"  {name:10s} {vals['precision'] * 100:7.2f} {vals['recall'] * 100:7.2f} {vals['f1'] * 100:7.2f}  {support_val:>20s}")
                        else:
                            print(
                                f"  {name:10s} {vals['precision'] * 100:7.2f} {vals['recall'] * 100:7.2f} {vals['f1'] * 100:7.2f}")


            print('=== CLUSTER METRICS (all) ===')
            _print_scorer_results(self.scorers['clusters_all'])
            print()
            print('=== CLUSTER METRICS (no generic) ===')
            _print_scorer_results(self.scorers['clusters_nogeneric'])
            print()
            print('=== CLUSTER METRICS (no singletons) ===')
            _print_scorer_results(self.scorers['clusters_nosingletons'])
            print()
            print('=== ENTITY ATTRIBUTES (all) ===')
            _print_scorer_results(self.scorers['entity_attributes_all_unrestricted'])
            print()
            print('=== ENTITY ATTRIBUTES (all, restrict on matches) ===')
            _print_scorer_results(self.scorers['entity_attributes_all_restrictonmatch'])
            print()
            print('=== ENTITY ATTRIBUTES (no group) ===')
            _print_scorer_results(self.scorers['entity_attributes_nogroup_unrestricted'])
            print()
            print('=== ENTITY ATTRIBUTES (no group, restrict on matches) ===')
            _print_scorer_results(self.scorers['entity_attributes_nogroup_restrictonmatch'])
            print()
            print('=== ENTITY ATTRIBUTES (no singletons/groups) ===')
            _print_scorer_results(self.scorers['entity_attributes_nogroupnosingletons_unrestricted'])
            print()
            print('=== ENTITY ATTRIBUTES (no singletons/groups, restrict on matches) ===')
            _print_scorer_results(self.scorers['entity_attributes_nogroupnosingletons_restrictonmatch'])
            print()
            print('=== MENTION ATTRIBUTES ===')
            _print_scorer_results(self.scorers['mention_attributes'])

        return f.getvalue()



    def add_document(self, key_mentions: List[Mention], sys_mentions: List[Mention],
                     doc_id: str = "doc"):
        for subset in ["all", "nogeneric", "nosingletons"]:
            self._update_cluster_metrics(key_mentions, sys_mentions, subset, doc_id)
        for subset in ["all", "nogroup", "nogroupnosingletons"]:
            for restrict_matches in [False, True]:
                self._update_entity_attribute_metrics(key_mentions, sys_mentions, subset, restrict_to_matches=restrict_matches, doc_id=doc_id)

        self._update_mention_attribute_metrics(key_mentions, sys_mentions, doc_id)


    def _update_cluster_metrics(self, key_mentions: List[Mention], sys_mentions: List[Mention],
                     mention_filter: Literal["all", "nogeneric", "nosingletons"], doc_id: str = "doc"):
        if mention_filter == "nogeneric":
            filter_key = get_generic_spans(key_mentions)
            filter_response = get_generic_spans(sys_mentions)
        elif mention_filter == "nosingletons":
            filter_key = get_singleton_spans(key_mentions)
            filter_response = get_singleton_spans(sys_mentions)
        else:
            filter_key = set()
            filter_response = set()

        mentions_to_filter = filter_key | filter_response

        key_clusters = mentions_to_clusters(key_mentions, doc_id, ignore_spans=mentions_to_filter)
        sys_clusters = mentions_to_clusters(sys_mentions, doc_id, ignore_spans=mentions_to_filter)

        key_mention_sys = get_mention_assignments(key_clusters, sys_clusters)
        sys_mention_key = get_mention_assignments(sys_clusters, key_clusters)
        key_mention_key = get_self_assignments(key_clusters)
        sys_mention_sys = get_self_assignments(sys_clusters)

        kwargs = dict(key_clusters=key_clusters, sys_clusters=sys_clusters,
                      key_mention_sys=key_mention_sys, sys_mention_key=sys_mention_key,
                      key_mention_key=key_mention_key, sys_mention_sys=sys_mention_sys)
        for name, metric_fn in CLUSTER_METRICS.items():
            res = metric_fn(**kwargs)
            self.scorers[f"clusters_{mention_filter}"][name].update(res, doc_id)

    def _update_entity_attribute_metrics(self, key_mentions: List[Mention], sys_mentions: List[Mention],
                                         entity_filter: Literal["all", "nogroup", "nogroupnosingletons"],
                                         restrict_to_matches=False, doc_id: str = "doc"):
        key_clusters = mentions_to_clusters(key_mentions, doc_id)
        sys_clusters = mentions_to_clusters(sys_mentions, doc_id)


        key_entities = {ref.entity.id: ref.entity for m in key_mentions for ref in m.references}
        sys_entities = {ref.entity.id: ref.entity for m in sys_mentions for ref in m.references}

        entity_mapping = compute_entity_mapping(key_clusters, sys_clusters)
        inverse_entity_mapping = {v: k for k, v in entity_mapping.items()}

        if entity_filter == "nogroup":
            key_entities_to_remove = {k for k, e in key_entities.items() if 'group' in e.specialcase_entity}
            sys_entities_to_remove = {k for k, e in sys_entities.items() if 'group' in e.specialcase_entity}
        elif entity_filter == "nogroupnosingletons":
            key_entities_to_remove = {k for k, e in key_entities.items() if len(key_clusters[k]) <= 1 or 'group' in e.specialcase_entity}
            sys_entities_to_remove = {k for k, e in sys_entities.items() if len(sys_clusters[k]) <= 1 or 'group' in e.specialcase_entity}
        else:
            key_entities_to_remove = set()
            sys_entities_to_remove = set()

        key_filter = lambda k: k in key_entities_to_remove or entity_mapping.get(k) in sys_entities_to_remove
        sys_filter = lambda k: k in sys_entities_to_remove or inverse_entity_mapping.get(k) in key_entities_to_remove

        key_entities = {k: e for k, e in key_entities.items() if not key_filter(k) and (not restrict_to_matches or k in entity_mapping.keys())}
        sys_entities = {k: e for k, e in sys_entities.items() if not sys_filter(k) and (not restrict_to_matches or k in inverse_entity_mapping.keys())}

        print(f"---- {entity_filter}, {restrict_to_matches=} ----")
        for k in key_entities.keys() & entity_mapping.keys():
            print(f"{key_entities[k].fullname:<30} {key_entities[k].gender}  ->  {sys_entities[entity_mapping[k]].gender} {sys_entities[entity_mapping[k]].fullname}")
        for k in key_entities.keys() - entity_mapping.keys():
            print(f"{key_entities[k].fullname:<30} {key_entities[k].gender}  ->  None")
        for r in sys_entities.keys() - inverse_entity_mapping.keys():
            print(f"{"None":<30}    ->  {sys_entities[r].gender} {sys_entities[r].fullname}")

        restrictkey = "restrictonmatch" if restrict_to_matches else "unrestricted"
        scorer_key = f'entity_attributes_{entity_filter}_{restrictkey}'
        for name in self.scorers[scorer_key].keys():
            attribute_fn = ENTITY_ATTRIBUTES[name]
            res = attribute_metric(key_entities, sys_entities, entity_mapping, attribute_fn)
            self.scorers[scorer_key][name].update(res, doc_id)

    def _update_mention_attribute_metrics(self, key_mentions: List[Mention], sys_mentions: List[Mention], doc_id: str = "doc"):
        key_spans = {(m.token_idx[0], m.token_idx[-1]): m.references for m in key_mentions}
        sys_spans = {(m.token_idx[0], m.token_idx[-1]): m.references for m in sys_mentions}

        # Identity mapping for overlapping spans
        mapping = {s: s for s in set(key_spans) & set(sys_spans)}

        for name, attribute_fn in MENTION_ATTRIBUTES.items():
            res = attribute_metric(key_spans, sys_spans, mapping, attribute_fn)
            self.scorers['mention_attributes'][name].update(res, doc_id)
