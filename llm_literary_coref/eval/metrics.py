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

def mention_metric(key_clusters: Clusters, sys_clusters: Clusters,
                   key_mention_sys: MentionAssignment, sys_mention_key: MentionAssignment, **_) -> tuple:
    sys_mentions = {m for mentions in sys_clusters.values() for m in mentions}
    key_mentions = {m for mentions in key_clusters.values() for m in mentions}
    p_num = len(sys_mentions & set(sys_mention_key.keys()))
    p_den = len(sys_mentions)
    r_num = len(key_mentions & set(key_mention_sys.keys()))
    r_den = len(key_mentions)
    return p_num, p_den, r_num, r_den


def muc_metric(key_clusters: Clusters, sys_clusters: Clusters,
               key_mention_sys: MentionAssignment, sys_mention_key: MentionAssignment, **_) -> tuple:
    def _muc_score(input_clusters: Clusters,
                   mention_to_gold: MentionAssignment):
        num, den = 0, 0
        for c in input_clusters.values():
            if len(c) == 1:
                continue

            den += len(c) - 1
            tp = len(c)
            linked = set()

            # for the partition of the response entity w.r.t. the key entities, we ignore plural mentions in the key
            for m in c:
                if m in mention_to_gold:
                    if len(mention_to_gold[m]) > 1:
                        continue
                    else:
                        linked.add(mention_to_gold[m][0])
                else:
                    tp = tp - 1
            tp -= len(linked)
            num += tp

        return num, den

    p_num, p_den = _muc_score(sys_clusters, sys_mention_key)
    r_num, r_den = _muc_score(key_clusters, key_mention_sys)
    return p_num, p_den, r_num, r_den




def lea_metric(key_clusters: Clusters, sys_clusters: Clusters,
               key_mention_sys: MentionAssignment, sys_mention_key: MentionAssignment, **_) -> tuple:

    def _lea_score(input_clusters: Clusters, output_clusters: Clusters,
                   mention_to_gold: MentionAssignment) -> tuple[float, float]:
        num, den = 0, 0
        for c in input_clusters.values():
            if len(c) == 1:
                all_links, common_links = 1, 0
                if c[0] in mention_to_gold:
                    for gid in mention_to_gold[c[0]]:
                        if len(output_clusters[gid]) == 1:
                            common_links = 1
                            break
            else:
                common_links = 0
                all_links = len(c) * (len(c) - 1) / 2.0
                for i, m in enumerate(c):
                    if m in mention_to_gold:
                        for m2 in c[i + 1:]:
                            if m2 in mention_to_gold:
                                if set(mention_to_gold[m]) & set(mention_to_gold[m2]):
                                    common_links += 1
            num += len(c) * common_links / all_links
            den += len(c)
        return num, den

    p_num, p_den = _lea_score(sys_clusters, key_clusters, sys_mention_key)
    r_num, r_den = _lea_score(key_clusters, sys_clusters, key_mention_sys)
    return p_num, p_den, r_num, r_den




def bcubed_metric(key_clusters: Clusters, sys_clusters: Clusters,
                  key_mention_sys: MentionAssignment, sys_mention_key: MentionAssignment,
                  key_mention_key: MentionAssignment, sys_mention_sys: MentionAssignment) -> tuple:
    def _bcubed_score(input_clusters: Clusters, output_clusters: Clusters,
                      mention_to_output: MentionAssignment, mention_to_input: MentionAssignment) -> tuple[float, float]:
        num, den = 0, 0
        for m in mention_to_input:
            den += 1
            gold_mentions = set()
            if m in mention_to_output:
                for gid in mention_to_output[m]:
                    gold_mentions |= set(output_clusters[gid])
            pred_mentions = set()
            for cid in mention_to_input[m]:
                pred_mentions |= set(input_clusters[cid])
            num += len(gold_mentions & pred_mentions) / len(pred_mentions)
        return num, den

    p_num, p_den = _bcubed_score(sys_clusters, key_clusters, sys_mention_key, sys_mention_sys)
    r_num, r_den = _bcubed_score(key_clusters, sys_clusters, key_mention_sys, key_mention_key)
    return p_num, p_den, r_num, r_den


def _ceaf_score(sys_clusters: Clusters, key_clusters: Clusters,
                phi_fn: Callable, den_fn: Callable) -> tuple:
    """CEAF helper with phi and denominator functions."""
    sys_list = list(sys_clusters.values())
    key_list = list(key_clusters.values())

    if not key_list or not sys_list:
        return 0, den_fn(sys_list), 0, den_fn(key_list)

    scores = np.zeros((len(key_list), len(sys_list)))
    for i, kc in enumerate(key_list):
        for j, sc in enumerate(sys_list):
            scores[i, j] = phi_fn(kc, sc)
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()
    return similarity, den_fn(sys_list), similarity, den_fn(key_list)


def ceafe_metric(key_clusters: Clusters, sys_clusters: Clusters, **_) -> tuple:
    """CEAF entity-based (phi4)."""
    phi4 = lambda c1, c2: 2 * len(set(c1) & set(c2)) / (len(c1) + len(c2)) if c1 and c2 else 0
    return _ceaf_score(sys_clusters, key_clusters, phi4, len)


def ceafm_metric(key_clusters: Clusters, sys_clusters: Clusters, **_) -> tuple:
    """CEAF mention-based (phi3)."""
    phi3 = lambda c1, c2: len(set(c1) & set(c2))
    return _ceaf_score(sys_clusters, key_clusters, phi3, lambda cs: sum(len(c) for c in cs))


def attribute_metric(gold_items: dict, pred_items: dict,
                     mapping: dict, extract_fn: Callable) -> tuple:
    """
    Compute attribute agreement scores. "Item" refers to mention or entity.

    Args:
        gold_items: {id: item, ...} from gold/key
        pred_items: {id: item, ...} from prediction/system
        mapping: gold_id -> pred_id alignment
        extract_fn: item -> bool, whether item has the attribute

    Returns:
        (p_num, p_den, r_num, r_den)
    """
    r_num = r_den = 0
    for gid, item in gold_items.items():
        if extract_fn(item):
            r_den += 1
            pred_item = pred_items.get(mapping.get(gid))
            if pred_item and extract_fn(pred_item):
                r_num += 1

    # symmetric for pred
    p_num = p_den = 0
    inv_mapping = {v: k for k, v in mapping.items()}
    for pid, item in pred_items.items():
        if extract_fn(item):
            p_den += 1
            gold_item = gold_items.get(inv_mapping.get(pid))
            if gold_item and extract_fn(gold_item):
                p_num += 1

    return p_num, p_den, r_num, r_den