from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class EvalMention:
    """A mention for evaluation purposes, hashable by document and span."""
    doc_id: str
    start: int
    end: int

    def __hash__(self):
        return hash((self.doc_id, self.start, self.end))

    def __eq__(self, other):
        return (self.doc_id, self.start, self.end) == (other.doc_id, other.start, other.end)


Clusters = dict[str, List[EvalMention]]
MentionAssignment = dict[EvalMention, List[str]]


def mention_recall(source_clusters: Clusters, mention_to_target: MentionAssignment, **_) -> tuple:
    source_mentions = {m for mentions in source_clusters.values() for m in mentions}
    num = len(source_mentions & set(mention_to_target.keys()))
    den = len(source_mentions)
    return num, den


def muc_recall(source_clusters: Clusters, mention_to_target: MentionAssignment, **_) -> tuple:
    num, den = 0, 0
    for c in source_clusters.values():
        if len(c) == 1:
            continue

        den += len(c) - 1
        tp = len(c)
        linked = set()

        # for the partition of the response entity w.r.t. the key entities, we ignore plural mentions in the key
        for m in c:
            if m in mention_to_target:
                if len(mention_to_target[m]) > 1:
                    continue
                else:
                    linked.add(mention_to_target[m][0])
            else:
                tp -= 1
        tp -= len(linked)
        num += tp

    return num, den


def lea_recall(source_clusters: Clusters, target_clusters: Clusters,
               mention_to_target: MentionAssignment, **_) -> tuple:
    num, den = 0, 0
    for c in source_clusters.values():
        if len(c) == 1:
            all_links, common_links = 1, 0
            if c[0] in mention_to_target:
                for tid in mention_to_target[c[0]]:
                    if len(target_clusters[tid]) == 1:
                        common_links = 1
                        break
        else:
            common_links = 0
            all_links = len(c) * (len(c) - 1) / 2.0
            for i, m in enumerate(c):
                if m in mention_to_target:
                    for m2 in c[i + 1:]:
                        if m2 in mention_to_target:
                            if set(mention_to_target[m]) & set(mention_to_target[m2]):
                                common_links += 1
        num += len(c) * common_links / all_links
        den += len(c)
    return num, den


def bcubed_recall(source_clusters: Clusters, target_clusters: Clusters,
                  mention_to_target: MentionAssignment, mention_to_source: MentionAssignment, **_) -> tuple:
    num, den = 0, 0
    for m in mention_to_source:
        den += 1
        target_mentions = set()
        if m in mention_to_target:
            for tid in mention_to_target[m]:
                target_mentions |= set(target_clusters[tid])
        source_mentions = set()
        for cid in mention_to_source[m]:
            source_mentions |= set(source_clusters[cid])
        num += len(target_mentions & source_mentions) / len(source_mentions)
    return num, den


def _ceaf_recall(source_clusters: Clusters, target_clusters: Clusters,
                 phi_fn: Callable, den_fn: Callable) -> tuple:
    """CEAF helper computing similarity with phi and denominator functions."""
    source_list = list(source_clusters.values())
    target_list = list(target_clusters.values())

    if not source_list or not target_list:
        return 0, den_fn(source_list)

    scores = np.zeros((len(source_list), len(target_list)))
    for i, kc in enumerate(source_list):
        for j, sc in enumerate(target_list):
            scores[i, j] = phi_fn(kc, sc)
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()

    return similarity, den_fn(source_list)


def ceafe_recall(source_clusters: Clusters, target_clusters: Clusters, **_) -> tuple:
    """CEAF entity-based (phi4)."""
    phi4 = lambda c1, c2: 2 * len(set(c1) & set(c2)) / (len(c1) + len(c2)) if c1 and c2 else 0
    return _ceaf_recall(source_clusters, target_clusters, phi4, len)


def ceafm_recall(source_clusters: Clusters, target_clusters: Clusters, **_) -> tuple:
    """CEAF mention-based (phi3)."""
    phi3 = lambda c1, c2: len(set(c1) & set(c2))
    return _ceaf_recall(source_clusters, target_clusters, phi3, lambda cs: sum(len(c) for c in cs))


def attribute_recall(source_items: dict, target_items: dict,
                     mapping: dict, extract_fn: Callable) -> tuple:
    """
    Compute attribute agreement scores specifically looking to isolate recall alignment.
    "Item" refers to mention or entity.

    Args:
        source_items: {id: item, ...} from source boundary (key / sys relative execution)
        target_items: {id: item, ...} from target boundary
        mapping: source_id -> target_id alignment
        extract_fn: item -> bool, whether item has the attribute

    Returns:
        (num, den) bounds required for fraction evaluation.
    """
    num = den = 0
    for sid, item in source_items.items():
        if extract_fn(item):
            den += 1
            target_item = target_items.get(mapping.get(sid))
            if target_item and extract_fn(target_item):
                num += 1

    return num, den
