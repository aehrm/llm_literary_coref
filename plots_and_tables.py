#%%

import collections
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from cycler import cycler
from scipy.stats import linregress
from tqdm import tqdm

from llm_literary_coref.eval.evaluator import (
    Evaluator,
    mentions_to_clusters,
    get_mention_assignments,
    compute_entity_mapping,
    get_self_assignments,
)
from llm_literary_coref.mention import parse_mentions

#%%
# =============================================================================
# Configuration: Paths, Plotting Style, Document Metadata
# =============================================================================

FIGURE_DIR = Path("/tmp/jcls_figures")
FIGURE_DIR.mkdir(exist_ok=True)

try:
    ROOT_DIR = Path(__file__).parent
except NameError:
    ROOT_DIR = Path(".").parent

ANNOTATIONS_DIR = ROOT_DIR / "gerfun_corpus"
MODEL_OUTPUT_DIR = ROOT_DIR / "llm_outputs"

# Document ordering and display titles
DOC_TITLE_PAIRS = [
    ("Fischer_Gustav", "Gustavs Verirrungen"),
    ("Heimburg_Trudchen", "Trudchens Heirat"),
    ("Wolff_Wildfangrecht", "Wildfangrecht"),
    ("Kürnberger_Amerika", "Amerika-Müde"),
    ("Goethe_Wahlverwandtschaften", "Wahlverwandtschaften"),
]
ORDERING = [x[0] for x in DOC_TITLE_PAIRS]
DOC_TITLE = dict(DOC_TITLE_PAIRS)

CLUSTER_VARIANTS = {
    "all": "full",
    "replaceplural": "plurals replaced",
    "nogeneric": "w/o generics",
    "nosingletons": "w/o singletons",
}
ENTITY_VARIANTS = {
    "all": "full",
    "nogroup": "w/o groups",
    "nogroupnosingletons": "w/o groups, singletons",
}
CLUSTER_METRICS = ["mentions", "muc", "bcub", "ceafe", "conll", "ceafm", "lea"]
ENTITY_METRICS = [
    "gender:m", "gender:f", "gender:u", "gender:nb", "gender:mf",
    "generic", "group", "nonfact",
]
MENTION_METRICS = ["part", "figurative", "generic"]

MODEL_LABELS = {
    "google--gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "google--gemini-2.5-flash": "Gemini 2.5 Flash",
    "google--gemini-3-flash-preview": "Gemini 3 Flash",
    "iaa": "Human vs. Human",
}
METRIC_LABELS = {"conll": "CoNLL F1", "lea": "LEA F1"}

GERFUN_MODELS = [
    "google--gemini-2.5-flash-lite",
    "google--gemini-3-flash-preview",
]

# Matplotlib style
matplotlib.rcParams["font.family"] = "Fira Sans"
matplotlib.rcParams["axes.prop_cycle"] = cycler(
    color=[
        (0.400, 0.761, 0.647),
        (0.988, 0.553, 0.384),
        (0.553, 0.627, 0.796),
        (0.906, 0.541, 0.765),
        (0.651, 0.847, 0.329),
        (1.000, 0.851, 0.184),
        (0.898, 0.769, 0.580),
        (0.702, 0.702, 0.702),
    ]
)

SMALL_SIZE = 7
MEDIUM_SIZE = 8

plt.rcParams.update(
    {
        "font.size": SMALL_SIZE,
        "axes.labelsize": SMALL_SIZE,
        "axes.titlesize": SMALL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "lines.linewidth": 0.8,
        "lines.markersize": 3,
        "grid.linewidth": 0.3,
        "figure.titlesize": MEDIUM_SIZE,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
    }
)

#%%



def log_number_fmt(x, pos=None):
    if 1 <= x <= 100:
        return f"{x:.0f}"
    elif 100 < x <= 1_000_000:
        return f"{x / 1000:.0f}K"
    elif x > 1_000_000:
        return f"{x / 1_000_000:.0f}M"
    elif 0.01 <= x < 1:
        return f"{x}"
    else:
        return f"{x:.2g}"


def percent_format(x, pos=None):
    if x > 0.01:
        return f"{x * 100:.0f}%"
    else:
        return f"{x * 100:.3g}%"


def save_figure(fig, name):
    path = FIGURE_DIR / name
    fig.savefig(path)
    print(f"  -> Figure saved to: {path}")


def print_section(title):
    print(f"\n\n\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


#%%
# =============================================================================
# Load Corpus Data
# =============================================================================

print_section("Loading corpus data")

document_files = (ANNOTATIONS_DIR / "annotated_tsv").glob("*.tsv")

documents = {}
document_mentions = {}
document_entities = {}

for doc_path in sorted(document_files):
    doc_id = doc_path.stem
    gold = pd.read_csv(
        doc_path, sep="\t", index_col="i", keep_default_na=False
    )["gold"].sort_index()

    mentions = list(parse_mentions(gold))

    source_path = doc_path.parent.parent / "sources" / doc_path.name
    df = pd.read_csv(
        source_path, sep="\t", index_col="i", keep_default_na=False, na_values=""
    ).sort_index()
    df["gold"] = gold

    documents[doc_id] = df
    document_mentions[doc_id] = mentions

    entities = collections.defaultdict(list)
    for mention in mentions:
        for ref in mention.references:
            entities[ref.entity.id].append((mention, ref))
    document_entities[doc_id] = entities

print(f"  Loaded {len(documents)} documents: {list(documents.keys())}")


#%%
# =============================================================================
# Table: Basic Corpus Statistics
# =============================================================================

def get_mentions_by_tag(document_id, tag):
    df = documents[document_id]
    return [
        m
        for m in document_mentions[document_id]
        if any(df.loc[m.token_idx, "tag"] == tag)
    ]



print_section("Basic corpus statistics")

rows = []
for doc_id in ORDERING:
    df = documents[doc_id]
    mentions = document_mentions[doc_id]
    entities = document_entities[doc_id]
    rows.append(
        {
            "doc": doc_id,
            "Num. tokens": len(df),
            "Num. sentences": df["is_sent_start"].sum(),
            "Num. sections": df["is_section_start"].sum(),
            "Num. mentions": len(mentions),
            "Num. references": sum(len(m.references) for m in mentions),
            "Num. entities": len(entities),
        }
    )

stats_df = pd.DataFrame(rows).set_index("doc").rename(index=DOC_TITLE)
num_docs = len(stats_df)
stats_df.loc["total"] = stats_df.sum()
stats_df.loc["average"] = stats_df.loc["total"] / num_docs

display_cols = [
    "Num. tokens",
    "Num. sentences",
    "Num. mentions",
    "Num. references",
    "Num. entities",
]
print(stats_df[display_cols].to_string(na_rep=""))

#%%
# =============================================================================
# Table: Mention-level Statistics
# =============================================================================

print_section("Mention-level statistics")

total_mentions = sum(len(m) for m in document_mentions.values())
ne_mentions = sum(len(get_mentions_by_tag(d, "NE")) for d in documents)
nn_mentions = sum(len(get_mentions_by_tag(d, "NN")) for d in documents)

mention_stats = [
    ("Num. mentions", total_mentions),
    (
        "Num. plural mentions",
        sum(
            1
            for mentions in document_mentions.values()
            for m in mentions
            if len(m.references) > 1
        ),
    ),
    ("Num. proper noun mentions", ne_mentions),
    ("Num. nominal noun mentions", nn_mentions),
    ("Num. pronominal mentions", total_mentions - ne_mentions - nn_mentions),
    (
        "Num. mentions with figurative reference",
        sum(
            1
            for mentions in document_mentions.values()
            for m in mentions
            if any("figurative" in r.specialcase_reference for r in m.references)
        ),
    ),
    (
        "Num. mentions with part reference",
        sum(
            1
            for mentions in document_mentions.values()
            for m in mentions
            if any("part" in r.specialcase_reference for r in m.references)
        ),
    ),
]

mention_stats_df = pd.DataFrame(mention_stats, columns=["label", "count"]).set_index("label")
mention_stats_df["average"] = mention_stats_df["count"] / len(documents)
mention_stats_df["proportion"] = mention_stats_df["count"] / total_mentions

print(mention_stats_df.to_string())

#%%
# =============================================================================
# Table: Entity-level Statistics
# =============================================================================

print_section("Entity-level statistics")


def _count_entities(predicate):
    return sum(
        1
        for entities in document_entities.values()
        for refs in entities.values()
        if predicate(refs)
    )


total_entities = _count_entities(lambda _: True)

entity_stats = [
    ("Num. entities", total_entities),
    ("Num. non-singleton entities", _count_entities(lambda r: len(r) > 1)),
    (
        "Num. non-singleton non-specialcase entities",
        _count_entities(
            lambda r: len(r) > 1 and r[0][1].entity.specialcase_entity == []
        ),
    ),
    ("Num. singleton entities", _count_entities(lambda r: len(r) == 1)),
    (
        "Num. non-generic singleton entities",
        _count_entities(
            lambda r: len(r) == 1
            and "generic" not in r[0][1].entity.specialcase_entity
        ),
    ),
    (
        "Num. generic entities",
        _count_entities(lambda r: "generic" in r[0][1].entity.specialcase_entity),
    ),
    (
        "Num. group entities",
        _count_entities(lambda r: "group" in r[0][1].entity.specialcase_entity),
    ),
    (
        "Num. group singleton entities",
        _count_entities(
            lambda r: len(r) == 1 and "group" in r[0][1].entity.specialcase_entity
        ),
    ),
    (
        "Num. nonfact entities",
        _count_entities(lambda r: "nonfact" in r[0][1].entity.specialcase_entity),
    ),
    (
        "Num. nonfact singleton entities",
        _count_entities(
            lambda r: len(r) == 1 and "nonfact" in r[0][1].entity.specialcase_entity
        ),
    ),
]

entity_stats_df = pd.DataFrame(entity_stats, columns=["label", "count"]).set_index("label")
entity_stats_df["average"] = entity_stats_df["count"] / len(documents)
entity_stats_df["proportion"] = entity_stats_df["count"] / total_entities

print(entity_stats_df.to_string(na_rep=""))


#%%
# =============================================================================
# Table: Entity Size Breakdown
# =============================================================================

print_section("Entity size breakdown")

entity_size_rows = []
for doc_id, entities in document_entities.items():
    for entity_id, mentions in entities.items():
        entity = mentions[0][1].entity
        entity_size_rows.append(
            {
                "doc": doc_id,
                "entity": entity.fullname,
                "num_references": len(mentions),
                "generic": "generic" in entity.specialcase_entity,
                "group": "group" in entity.specialcase_entity,
                "specialcase": entity.specialcase_entity != [],
            }
        )

entity_sizes = pd.DataFrame(entity_size_rows)

total_num_entities = len(entity_sizes)
total_num_references = entity_sizes["num_references"].sum()

summary_rows = []
for label, mask in [
    ("All entities", entity_sizes.index == entity_sizes.index),
    ("Core entities", ~entity_sizes["specialcase"] & (entity_sizes["num_references"] > 1)),
    ("Group entities", entity_sizes["group"]),
    ("Generic singletons", entity_sizes["generic"]),
    ("Non-generic singletons", ~entity_sizes["generic"] & (entity_sizes["num_references"] == 1)),
]:
    subset = entity_sizes[mask]
    summary_rows.append(
        {
            "label": label,
            "number entities": len(subset),
            "total number references": subset["num_references"].sum(),
        }
    )

summary_df = pd.DataFrame(summary_rows).set_index("label")
summary_df["proportion of entities"] = summary_df["number entities"] / total_num_entities
summary_df["proportion of references"] = summary_df["total number references"] / total_num_references

print(summary_df.to_string())


#%%
# =============================================================================
# Table: Gender Distribution of Entities
# =============================================================================

print_section("Gender distribution of entities")

gender_rows = []
for entities in document_entities.values():
    for refs in entities.values():
        e = refs[0][1].entity
        gender_rows.append(
            {
                "gender": e.gender,
                "generic": "generic" in e.specialcase_entity,
                "group": "group" in e.specialcase_entity,
                "num_references": len(refs),
            }
        )

gender_df = pd.DataFrame(gender_rows)
gender_df.loc[gender_df["gender"].apply(lambda g: "o" in g or "u" in g), "gender"] = "u"

gender_pivot = pd.pivot_table(
    gender_df, index="gender", columns=["generic", "group"], aggfunc=len
)
total_count = gender_pivot.sum().sum()
gender_pivot["Overall"] = gender_pivot.sum(axis=1)

gender_pivot = pd.concat(
    {"count": gender_pivot, "percent": 100 * gender_pivot / total_count}, axis=1
)
gender_pivot = gender_pivot.reorder_levels([1, 2, 3, 0], axis=1).sort_index(
    level=[1, 2], axis=1
)

print(gender_pivot.to_string(na_rep=0))


#%%
# =============================================================================
# Compute Entity Sizes (used by several later cells)
# =============================================================================

print_section("Entity size breakdown")

entity_size_rows = []
for doc_id, entities in document_entities.items():
    for entity_id, mentions in entities.items():
        entity = mentions[0][1].entity
        entity_size_rows.append(
            {
                "doc": doc_id,
                "entity": entity.fullname,
                "num_references": len(mentions),
                "generic": "generic" in entity.specialcase_entity,
                "group": "group" in entity.specialcase_entity,
                "specialcase": entity.specialcase_entity != [],
            }
        )

entity_sizes = pd.DataFrame(entity_size_rows)

summary_rows = []
for label, mask in [
    ("All entities", entity_sizes.index == entity_sizes.index),  # all True
    ("Core entities", ~entity_sizes["specialcase"] & (entity_sizes["num_references"] > 1)),
    ("Group entities", entity_sizes["group"]),
    ("Generic singletons", entity_sizes["generic"]),
    ("Non-generic singletons", ~entity_sizes["generic"] & (entity_sizes["num_references"] == 1)),
]:
    subset = entity_sizes[mask]
    summary_rows.append(
        {
            "label": label,
            "number entities": len(subset),
            "total number references": subset["num_references"].sum(),
        }
    )

summary_df = pd.DataFrame(summary_rows).set_index("label")
summary_df["proportion of entities"] = (
    summary_df["number entities"] / summary_df.iloc[0]["number entities"]
)
summary_df["proportion of references"] = (
    summary_df["total number references"]
    / summary_df.iloc[0]["total number references"]
)

print(summary_df.to_string()) 

#%%
# =============================================================================
# Figure: Zipf Plot (Entity Size Distribution)
# =============================================================================

print_section("Figure: Zipf / entity size distribution")

fig, (ax_rank, ax_cumul) = plt.subplots(ncols=2, figsize=(5.4, 3.0), dpi=300)

# --- Left panel: rank vs. count with power-law fits ---
color_cycle = iter(plt.rcParams["axes.prop_cycle"])
for doc_id, doc_entities_df in sorted(
    entity_sizes.groupby("doc"), key=lambda x: ORDERING.index(x[0])
):
    sel = ~doc_entities_df["group"] & (doc_entities_df["num_references"] > 1)
    ranked = doc_entities_df.loc[sel].sort_values("num_references", ascending=False).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    x, y = ranked["rank"].values, ranked["num_references"].values
    color = next(color_cycle)["color"]

    ax_rank.scatter(
        x, y, marker="o", facecolors="none", edgecolors=color,
        linewidth=plt.rcParams["lines.linewidth"], alpha=0.5,
    )

    slope, intercept, *_ = linregress(np.log(x), np.log(y))
    x_line = np.array([x.min(), x.max()])
    y_line = np.exp(intercept + slope * np.log(x_line))
    ax_rank.plot(
        x_line, y_line, "--", color=color, alpha=0.7,
        label=f"${np.exp(intercept)/1000:.2f}\\cdot 10^3 \\cdot x^{{{slope:.3g}}}$",
    )

ax_rank.set_xscale("log")
ax_rank.set_yscale("log")
ax_rank.set_xlabel("Entity rank")
ax_rank.set_ylabel("Number of references")
ax_rank.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
ax_rank.yaxis.set_major_formatter(FuncFormatter(log_number_fmt))
ax_rank.legend(frameon=False)

# --- Right panel: cumulative proportion ---
color_cycle = iter(plt.rcParams["axes.prop_cycle"])
for doc_id, doc_entities_df in sorted(
    entity_sizes.groupby("doc"), key=lambda x: ORDERING.index(x[0])
):
    sel = ~doc_entities_df["group"] & (doc_entities_df["num_references"] > 1)
    ranked = doc_entities_df.loc[sel].sort_values("num_references", ascending=False).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    x = ranked["rank"].values
    y = ranked["num_references"].cumsum().values / doc_entities_df["num_references"].sum()

    color = next(color_cycle)["color"]
    ax_cumul.scatter(
        x, y, marker="o", facecolors="none", edgecolors=color,
        label=DOC_TITLE[doc_id], linewidth=plt.rcParams["lines.linewidth"], alpha=0.5,
    )

ax_cumul.set_xscale("log")
ax_cumul.set_yscale("log")
ax_cumul.set_xlabel("Entity rank")
ax_cumul.set_ylabel("Cumulative Proportion of references")
ax_cumul.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
ax_cumul.yaxis.set_major_formatter(FuncFormatter(percent_format))
ax_cumul.yaxis.set_minor_formatter(FuncFormatter(percent_format))
ax_cumul.legend(frameon=False)

plt.tight_layout()
save_figure(fig, "entity_sizes.pdf")
plt.show()


#%%
# =============================================================================
# Compute Entity Spread (used by plot and summary)
# =============================================================================

print_section("Entity spread statistics")

spread_rows = []
for doc_id, entities in document_entities.items():
    for entity_id, mentions in entities.items():
        entity = mentions[0][1].entity
        token_positions = [i for m, _ in mentions for i in m.token_idx]
        spread_rows.append(
            {
                "doc": doc_id,
                "entity": entity.fullname,
                "num_references": len(mentions),
                "spread": max(token_positions) - min(token_positions),
                "generic": "generic" in entity.specialcase_entity,
                "group": "group" in entity.specialcase_entity,
            }
        )

spreads = pd.DataFrame(spread_rows)

print(
    spreads[spreads["num_references"] > 1]["spread"]
    .describe(percentiles=[0.5, 0.90])
    .to_string()
)

#%%
# =============================================================================
# Figure: Entity Spread vs. Number of References
# =============================================================================

print_section("Figure: Entity spread vs. number of references")

n_docs = len(ORDERING)
nrows = (n_docs + 1) // 2
fig, ax_grid = plt.subplots(
    ncols=2, nrows=nrows, sharex=True, sharey=True, figsize=(5.3, 1.65 * nrows),
)
axs = ax_grid.flatten()

color_cycle = iter(plt.rcParams["axes.prop_cycle"])
for i, doc_id in enumerate(ORDERING):
    ax = axs[i]
    doc_spread = spreads[spreads["doc"] == doc_id]
    sel = doc_spread["num_references"] > 1
    g = doc_spread[sel]

    color = next(color_cycle)["color"]
    ax.scatter(
        g["num_references"], g["spread"],
        marker="o", facecolors="none", edgecolors=color, alpha=0.7,
    )
    ax.axhline(len(documents[doc_id]), ls="--", color=color, alpha=0.5, linewidth=1.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(log_number_fmt))
    ax.xaxis.set_major_formatter(FuncFormatter(log_number_fmt))
    ax.set_title(DOC_TITLE[doc_id])
    ax.set_ylabel("Spread in Tokens")
    ax.set_xlabel("Number of References")

# Remove unused subplots and ensure bottom-row labels are visible
for j in range(n_docs, len(axs)):
    fig.delaxes(axs[j])
for j in range(n_docs - 2, n_docs):
    axs[j].tick_params(labelbottom=True)

plt.tight_layout()
save_figure(fig, "entity_spread.pdf")
plt.show()

#%%
# =============================================================================
# Compute Mention Distances
# =============================================================================

print_section("Computing mention distances")

total_mention_count = sum(
    len(refs) for entities in document_entities.values() for refs in entities.values()
)

mention_distance_frames = []

with tqdm(total=total_mention_count, desc="Mention distances") as pbar:
    for doc_id, entities in document_entities.items():
        doc_df = documents[doc_id]

        for entity_id, refs in entities.items():
            sorted_mentions = sorted(
                (m for m, _ in refs), key=lambda m: m.token_idx[0]
            )

            positions = [m.token_idx[0] for m in sorted_mentions]
            types = []
            for m in sorted_mentions:
                tags = list(doc_df.loc[m.token_idx, "tag"])
                if "NE" in tags:
                    types.append("NE")
                elif "NN" in tags:
                    types.append("NN")
                else:
                    types.append(doc_df.loc[m.token_idx[0], "tag"])

            dist_df = pd.DataFrame(
                {"position": positions, "type": types, "doc": doc_id, "entity": entity_id}
            )
            pbar.update(len(dist_df))

            if len(dist_df) <= 1 or not (dist_df["type"] == "NE").any():
                continue

            ne_positions = (
                dist_df[dist_df["type"] == "NE"]
                .reindex(dist_df.index)["position"]
            )

            dist_df["distance_to_previous_mention"] = dist_df["position"].diff()
            dist_df["distance_to_previous_ne"] = dist_df["position"] - ne_positions.ffill()
            dist_df["distance_to_next_mention"] = -dist_df["position"].diff(-1)
            dist_df["distance_to_next_ne"] = ne_positions.bfill() - dist_df["position"]

            dist_df["dist_to_mention"] = dist_df[
                ["distance_to_previous_mention", "distance_to_next_mention"]
            ].min(axis=1)
            dist_df["dist_to_ne"] = dist_df[
                ["distance_to_previous_ne", "distance_to_next_ne"]
            ].min(axis=1)

            mention_distance_frames.append(dist_df)

mention_distances = pd.concat(mention_distance_frames, ignore_index=True)

#%%
# =============================================================================
# Table: Mention Distance Descriptive Statistics (non-NE mentions)
# =============================================================================

print_section("Mention distance descriptive statistics (non-NE mentions)")

non_ne_distances = mention_distances.loc[mention_distances["type"] != "NE"].dropna()
print(non_ne_distances.describe(percentiles=[0.5, 0.99]).to_string())

#%%
# =============================================================================
# Figure: Mention Distance Survival Curves
# =============================================================================

print_section("Figure: Mention distance survival curves")

distance_thresholds = [1, 2, 5] + list(
    np.logspace(np.log10(10), np.log10(30000), 25)
)

fig, ax = plt.subplots(figsize=(5, 2))
color_cycle = iter(plt.rcParams["axes.prop_cycle"])

for doc_id, g in sorted(
    non_ne_distances.groupby("doc"), key=lambda x: ORDERING.index(x[0])
):
    surv = pd.DataFrame(index=distance_thresholds)
    for col in ["dist_to_mention", "dist_to_ne"]:
        surv[col] = [(g[col] >= t).mean() for t in distance_thresholds]
    surv = surv.replace(0, np.nan)

    color = next(color_cycle)["color"]
    ax.plot(
        distance_thresholds, surv["dist_to_mention"],
        "-", label=DOC_TITLE[doc_id], alpha=0.8, lw=1.4, color=color,
    )
    ax.plot(
        distance_thresholds, surv["dist_to_ne"],
        "--", alpha=0.8, lw=1.4, color=color,
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Distance in Tokens")
ax.set_ylabel("Prop. Mentions")
ax.xaxis.set_major_formatter(FuncFormatter(log_number_fmt))
ax.yaxis.set_major_formatter(FuncFormatter(percent_format))
ax.legend(frameon=False)

plt.tight_layout()
save_figure(fig, "mention_distance.pdf")
plt.show()

#%%
# =============================================================================
# Load IAA Report
# =============================================================================

print_section("Inter-Annotator Agreement (IAA)")

iaa_report_path = ANNOTATIONS_DIR / "evaluation_reports" / "iaa_report.json"
with open(iaa_report_path) as f:
    iaa_report = json.load(f)

print(f"  Loaded IAA report from: {iaa_report_path}")

#%%
# =============================================================================
# Table: IAA Cluster Metrics
# =============================================================================

print_section("IAA cluster-level metrics")

iaa_cluster_table = pd.DataFrame(
    index=list(CLUSTER_VARIANTS.keys()),
    columns=CLUSTER_METRICS,
)

for variant in iaa_cluster_table.index:
    report = iaa_report[f"clusters_{variant}"]
    for metric, scores in report.items():
        iaa_cluster_table.loc[variant, metric] = scores["aggregated"]["f1"]

iaa_cluster_table["conll"] = iaa_cluster_table[["muc", "bcub", "ceafe"]].astype(float).mean(axis=1)

print(
    iaa_cluster_table.rename(index=CLUSTER_VARIANTS).to_string(
        float_format=lambda x: f"{x * 100:.2f}"
    )
)

#%%
# =============================================================================
# Table: IAA Entity Attribute Metrics
# =============================================================================

print_section("IAA entity attribute metrics")


iaa_entity_table = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [list(ENTITY_VARIANTS.keys()), ["f1", "count"]]
    ),
    columns=ENTITY_METRICS,
)

for variant in ENTITY_VARIANTS:
    report = iaa_report[f"entity_attributes_{variant}_restrictonmatch"]
    for metric, scores in report.items():
        counts = sorted(
            [scores["aggregated"]["support_key"], scores["aggregated"]["support_response"]]
        )
        if sum(counts) > 0:
            iaa_entity_table.loc[(variant, "f1"), metric] = scores["aggregated"]["f1"]
            iaa_entity_table.loc[(variant, "count"), metric] = "+".join(map(str, counts))

print(
    iaa_entity_table.rename(index=ENTITY_VARIANTS).to_string(
        na_rep="--", float_format=lambda x: f"{x * 100:.2f}"
    )
)

#%%
# =============================================================================
# Table: IAA Mention Attribute Metrics
# =============================================================================

print_section("IAA mention attribute metrics")

iaa_mention_table = pd.DataFrame(
    index=MENTION_METRICS, columns=["f1", "count"]
)

for metric in MENTION_METRICS:
    scores = iaa_report["mention_attributes"][metric]["aggregated"]
    counts = sorted([scores["support_key"], scores["support_response"]])
    if sum(counts) > 0:
        iaa_mention_table.loc[metric, "count"] = "+".join(map(str, counts))
        iaa_mention_table.loc[metric, "f1"] = scores["f1"]

print(
    iaa_mention_table.to_string(na_rep="--", float_format=lambda x: f"{x * 100:.2f}")
)

#%%
# =============================================================================
# Table: Difference to Silver Pre-annotation
# =============================================================================

print_section("Difference to silver pre-annotation (per chapter)")

num_chapters = sum(df["is_section_start"].sum() for df in documents.values())

pre_annotation_scores = []
with tqdm(total=num_chapters, desc="Pre-annotation evaluation") as pbar:
    for doc_id, doc_df in documents.items():
        source_df = pd.read_csv(
            ANNOTATIONS_DIR / "sources" / (doc_id + ".tsv"),
            sep="\t", keep_default_na=False,
        )
        pre_annotations = source_df["llm_pre_annotation"]
        chap_ids = source_df["is_section_start"].cumsum()

        for chap_id, chap in doc_df.groupby(chap_ids):
            gold_mentions = list(parse_mentions(chap["gold"]))
            pre_mentions = list(parse_mentions(pre_annotations.loc[chap.index]))

            evaluator = Evaluator()
            evaluator.add_document(
                key_mentions=gold_mentions, sys_mentions=pre_mentions
            )
            report = evaluator.report(as_dict=True)
            pre_annotation_scores.append(
                (doc_id, chap_id, report["clusters_all"])
            )
            pbar.update(1)

#%%
# =============================================================================
# Table: Aggregated Pre-annotation Scores
# =============================================================================

print_section("Aggregated pre-annotation scores (median with IQR)")

pre_scores_df = pd.DataFrame(
    index=pd.MultiIndex.from_tuples([(d, c) for d, c, _ in pre_annotation_scores]),
    columns=pd.MultiIndex.from_product([CLUSTER_METRICS, ["precision", "recall", "f1"]]),
)

for doc_id, chap_id, report in pre_annotation_scores:
    for metric in CLUSTER_METRICS:
        if metric == "conll":
            continue
        for part in ["precision", "recall", "f1"]:
            pre_scores_df.loc[(doc_id, chap_id), (metric, part)] = report[metric]["doc"][part]
    pre_scores_df.loc[(doc_id, chap_id), ("conll", "f1")] = np.mean(
        [report[m]["doc"]["f1"] for m in ["muc", "bcub", "ceafe"]]
    )

agg = pre_scores_df.aggregate(
    ["median", lambda x: x.quantile(0.25) - x.median(), lambda x: x.quantile(0.75) - x.median()]
)
agg.index = ["median", "25%", "75%"]

agg_display = agg.T.unstack().reorder_levels([1, 0], axis=1)
agg_display = agg_display.loc[CLUSTER_METRICS, ["precision", "recall", "f1"]]

print(
    agg_display.to_string(na_rep="--", float_format=lambda x: f"{x * 100:.2f}")
)

#%%
# =============================================================================
# Evaluate DROC Test Set: LLM Models + Human IAA
# =============================================================================

print_section("DROC test set evaluation")

DROC_MODELS = ["google--gemini-2.5-flash-lite", "google--gemini-2.5-flash"]

droc_scores = []

# LLM model predictions
print("  Evaluating LLM vs. human...")
for model in DROC_MODELS:
    model_dir = MODEL_OUTPUT_DIR / "droc_test_set" / model
    all_docs = list(model_dir.glob("*.tsv"))
    print(f"  Evaluating model '{model}' on {len(all_docs)} documents...")

    for doc_path in all_docs:
        # rate LLM inference of RATER1 mentions against RATE2 file
        if "RATER1" in doc_path.name:
            other_path = doc_path.parent / doc_path.name.replace("RATER1", "RATER2")
        else:
            other_path = doc_path.parent / doc_path.name.replace("RATER2", "RATER1")

        pred_df = pd.read_csv(doc_path, sep="\t", keep_default_na=False)
        gold_df = pd.read_csv(other_path, sep="\t", keep_default_na=False)

        evaluator = Evaluator()
        evaluator.add_document(
            key_mentions=list(parse_mentions(gold_df["gold"])),
            sys_mentions=list(parse_mentions(pred_df["pred"])),
        )
        report = evaluator.report(as_dict=True)
        droc_scores.append((model, doc_path.name, report["clusters_all"]))

# Human vs Human IAA
print("  Evaluating human vs. human IAA...")
droc_dir = ROOT_DIR / "droc_test_set"
for doc_path in sorted(droc_dir.glob("*.tsv")):
    if "RATER1" not in doc_path.name:
        continue
    other_path = doc_path.parent / doc_path.name.replace("RATER1", "RATER2")

    df1 = pd.read_csv(doc_path, sep="\t", keep_default_na=False)
    df2 = pd.read_csv(other_path, sep="\t", keep_default_na=False)

    evaluator = Evaluator()
    evaluator.add_document(
        key_mentions=list(parse_mentions(df1["gold"])),
        sys_mentions=list(parse_mentions(df2["gold"])),
    )
    report = evaluator.report(as_dict=True)
    droc_scores.append(("iaa", doc_path.name, report["clusters_all"]))

#%%
# =============================================================================
# Build DROC Scores DataFrame
# =============================================================================

droc_scores_df = pd.DataFrame(
    index=pd.MultiIndex.from_tuples(
        [(m, d) for m, d, _ in droc_scores], names=["model", "doc"]
    ),
    columns=pd.MultiIndex.from_product([CLUSTER_METRICS, ["precision", "recall", "f1"]]),
)

for model, doc_name, report in droc_scores:
    for metric in CLUSTER_METRICS:
        if metric == "conll":
            continue
        for part in ["precision", "recall", "f1"]:
            if part != "f1" and model == "iaa":
                continue
            droc_scores_df.loc[(model, doc_name), (metric, part)] = report[metric]["doc"][part]
    droc_scores_df.loc[(model, doc_name), ("conll", "f1")] = np.mean(
        [report[m]["doc"]["f1"] for m in ["muc", "bcub", "ceafe"]]
    )

#%%
# =============================================================================
# Table: Aggregated DROC Scores (median with IQR)
# =============================================================================

print_section("Aggregated DROC scores per model (median F1 with IQR)")

droc_agg = droc_scores_df.loc[:, (slice(None), 'f1')].groupby(level="model").agg(
    ["median", lambda x: x.quantile(0.25) - x.median(), lambda x: x.quantile(0.75) - x.median()]
)
droc_agg.columns = droc_agg.columns.set_levels(
    ["median", "25%", "75%"], level=-1
)

droc_agg_display = (
    droc_agg
    .stack(level=0)
)

print(
    droc_agg_display.to_string(na_rep="--", float_format=lambda x: f"{x * 100:.2f}")
)

#%%
# =============================================================================
# Figure: DROC Performance Boxplot
# =============================================================================

print_section("Figure: DROC performance boxplot")

DROC_PLOT_METRICS = ["conll", "lea"]
DROC_PLOT_MODELS = ["google--gemini-2.5-flash-lite", "google--gemini-2.5-flash", "iaa"]

maverick_baseline = pd.Series({"conll": 0.7375, "lea": 0.6777})

boxplot_kwargs = dict(
    patch_artist=True,
    boxprops=dict(fc="black", edgecolor="black"),
    widths=0.07,
    medianprops=dict(solid_capstyle="projecting", color="white"),
    showcaps=False,
    flierprops=dict(markersize=4, markeredgewidth=0.6),
)

fig, axs = plt.subplots(nrows=len(DROC_PLOT_METRICS), figsize=(4, 3.0), dpi=300)

for ax, metric in zip(axs, DROC_PLOT_METRICS):
    ticks = np.arange(len(DROC_PLOT_MODELS))
    data = [
        100 * droc_scores_df.loc[m, (metric, "f1")].values
        for m in DROC_PLOT_MODELS
    ]
    ax.boxplot(data, positions=ticks, orientation="horizontal", **boxplot_kwargs)
    ax.axvline(np.median(data[-1]), ls="--", color="black")
    ax.axvline(100 * maverick_baseline[metric], ls="--", color="#ee7777")
    ax.yaxis.set_inverted(True)
    ax.set_xlabel(METRIC_LABELS[metric])
    ax.set_yticks(ticks, [MODEL_LABELS[m] for m in DROC_PLOT_MODELS])

plt.tight_layout(h_pad=3)
save_figure(fig, "droc_performance.pdf")
plt.show()

#%%
# =============================================================================
# Load LLM Evaluation Reports (GerFun corpus)
# =============================================================================

print_section("Loading LLM evaluation reports (GerFun)")

llm_eval_reports = {}
for model_dir in (MODEL_OUTPUT_DIR / "merged").iterdir():
    if not model_dir.is_dir():
        continue
    report_path = model_dir / "evaluation_report.json"
    if not report_path.is_file():
        print(f"  WARNING: no evaluation file for {model_dir.name}")
        continue
    with open(report_path) as f:
        llm_eval_reports[model_dir.name] = json.load(f)

print(f"  Loaded reports for: {list(llm_eval_reports.keys())}")

#%%
# =============================================================================
# Table: LLM Cluster Metrics on GerFun
# =============================================================================

print_section("LLM cluster metrics on GerFun corpus")

cluster_index = pd.MultiIndex.from_product(
    [list(CLUSTER_VARIANTS.keys()), GERFUN_MODELS, ORDERING],
    names=["variant", "model", "doc"],
)
cluster_columns = pd.MultiIndex.from_product([CLUSTER_METRICS, ["precision", "recall", "f1"]])

cluster_table = pd.DataFrame(index=cluster_index.sortlevel()[0], columns=cluster_columns.sortlevel()[0])

for variant, model, doc_id in cluster_table.index:
    report = llm_eval_reports[model][f"clusters_{variant}"]
    for metric, part in cluster_table.columns:
        if metric == "conll":
            continue
        if doc_id in report.get(metric, {}):
            cluster_table.loc[(variant, model, doc_id), (metric, part)] = report[metric][doc_id][part]
    cluster_table.loc[(variant, model, doc_id), ("conll", "f1")] = (
        cluster_table.loc[(variant, model, doc_id), (["muc", "bcub", "ceafe"], "f1")]
        .astype(float)
        .mean()
    )

summary_df = cluster_table.loc[cluster_index, cluster_columns].groupby(level=[0, 1])\
    .mean()\
    .rename(index=CLUSTER_VARIANTS)\
    .rename(columns={"precision": "P", "recall": "R", "f1": "F1"})

print(summary_df.to_string(na_rep="--", float_format=lambda x: f"{x * 100:.1f}"))

#%%
# =============================================================================
# Figure: GerFun Performance per Document
# =============================================================================

print_section("Figure: GerFun performance per document")

fig, axs = plt.subplots(
    ncols=len(DROC_PLOT_METRICS), nrows=2,
    figsize=(5.0, 2.5), dpi=300, height_ratios=[1, 0],
)

handles = []
for ax, metric in zip(axs[0], DROC_PLOT_METRICS):
    tick_pos = []
    for j, model in enumerate(GERFUN_MODELS):
        color_cycle = iter(plt.rcParams["axes.prop_cycle"])
        ax.set_ylabel(METRIC_LABELS[metric])
        start_x = (len(ORDERING) + 0.3) * j

        for i, doc_id in enumerate(ORDERING):
            color = next(color_cycle)["color"]
            val = 100 * float(cluster_table.loc[("all", model, doc_id), (metric, "f1")])
            h = ax.scatter(start_x + i, val, label=DOC_TITLE[doc_id], color=color)
            handles.append(h)

        avg = (
            100
            * cluster_table.loc[("all", model, ORDERING), (metric, "f1")]
            .astype(float)
            .mean()
        )
        (h_line,) = ax.plot(
            [start_x, start_x + len(ORDERING)], [avg, avg],
            color="black", ls="--", label="avg", alpha=0.5,
        )
        handles.append(h_line)
        tick_pos.append(start_x + len(ORDERING) / 2)

    ax.set_xticks(tick_pos, [MODEL_LABELS[m] for m in GERFUN_MODELS])
    ax.set_ylim((30, 100))

# Shared legend in bottom row
gs = axs[0, 0].get_gridspec()
for ax in axs[1, :]:
    ax.remove()

ax_legend = fig.add_subplot(gs[1, :])
for spine in ax_legend.spines.values():
    spine.set_visible(False)
ax_legend.set_xticks([])
ax_legend.set_yticks([])
ax_legend.legend(
    frameon=False, ncol=2, loc="upper center",
    bbox_to_anchor=(0.5, -0.75),
    handles=handles[: len(ORDERING) + 1],
)

plt.tight_layout()
save_figure(fig, "gerfun_performance.pdf")
plt.show()

#%%
# =============================================================================
# Table: Per-segment GerFun performance (median F1 with IQR)
# =============================================================================

print_section("Per-segment GerFun performance (median F1 with IQR)")

segment_scores = []
for model in GERFUN_MODELS:
    report = llm_eval_reports[model]["clusters_all"]
    segment_ids = sorted(k for k in report["lea"] if "segment" in k)

    for segment_id in segment_ids:
        lea_f1 = report["lea"][segment_id]["f1"]
        muc_f1 = report["muc"][segment_id]["f1"]
        bcub_f1 = report["bcub"][segment_id]["f1"]
        ceafe_f1 = report["ceafe"][segment_id]["f1"]
        conll_f1 = np.mean([muc_f1, bcub_f1, ceafe_f1])

        segment_scores.append(
            {
                "model": model,
                "segment": segment_id,
                "conll": conll_f1,
                "lea": lea_f1,
            }
        )

segment_scores_df = pd.DataFrame(segment_scores)

segment_summary = (
    segment_scores_df
    .groupby("model")[["conll", "lea"]]
    .agg([
        "median",
        lambda x: x.quantile(0.25) - x.median(),
        lambda x: x.quantile(0.75) - x.median(),
    ])
)
segment_summary.columns = segment_summary.columns.set_levels(
    ["median", "25%", "75%"], level=-1
)

print(segment_summary.stack(level=0).to_string( float_format=lambda x: f"{x * 100:.2f}"))

#%%
# =============================================================================
# Table: Inference Cost
# =============================================================================

print_section("Inference cost overview")

cost_rows = []
for model in llm_eval_reports:
    pred_dirs = [
        MODEL_OUTPUT_DIR / "predicted_sections" / model.replace("/", "--"),
        MODEL_OUTPUT_DIR / "merged" / model.replace("/", "--"),
    ]
    for pred_dir in pred_dirs:
        if not pred_dir.exists():
            continue
        for pred_json in pred_dir.glob("*.json"):
            if "evaluation_report" in pred_json.stem:
                continue
            with open(pred_json) as f:
                obj = json.load(f)

            is_merge = "merged" in str(pred_json)
            tsv_path = pred_json.parent / (pred_json.stem + ".tsv")
            segment_length = len(pd.read_csv(tsv_path, sep="\t"))

            for i, det in enumerate(obj["generation_details"]):
                usage = det["usage"]
                if usage is None:
                    print(f"  WARNING: no usage info for {pred_json}, request {i}")
                cost_rows.append(
                    {
                        "model": model,
                        "file": pred_json.stem,
                        "request": i,
                        "cost": usage["cost"] if usage else np.nan,
                        "segment_length": segment_length if usage and not is_merge else np.nan,
                        "prompt_tokens": usage["prompt_tokens"] if usage else np.nan,
                        "completion_tokens": usage["completion_tokens"] if usage else np.nan,
                    }
                )

cost_df = pd.DataFrame(cost_rows).set_index(["model", "file", "request"]).astype("float32")

total_cost = cost_df.groupby(level=0).sum()
print("Total cost per model:")
print(total_cost.to_string())
print("\nCost normalized by segment length:")
print(
    total_cost.apply(lambda row: row / row["segment_length"], axis=1).to_string(
        float_format="%.4g"
    )
)
print("\nOverall tokens per annotated token:")
totals = cost_df[["prompt_tokens", "completion_tokens"]].sum()
print((totals / cost_df["segment_length"].sum()).to_string(float_format="%.4g"))

#%%
# =============================================================================
# Table: LLM Entity Attribute Metrics
# =============================================================================

print_section("LLM entity attribute metrics")

LLM_ENTITY_METRICS = ["gender:m", "gender:f", "gender:u", "group", "nonfact"]

llm_entity_table = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [list(ENTITY_VARIANTS.keys()), GERFUN_MODELS]
    ),
    columns=pd.MultiIndex.from_product([LLM_ENTITY_METRICS, ["precision", "recall", "f1"]]),
)

for variant, model in llm_entity_table.index:
    for metric, part in llm_entity_table.columns:
        report = llm_eval_reports[model][f"entity_attributes_{variant}_restrictonmatch"][metric]
        scores = [report[doc_id][part] for doc_id in ORDERING]
        llm_entity_table.loc[(variant, model), (metric, part)] = np.mean(scores)

print(
    llm_entity_table.rename(index=ENTITY_VARIANTS)
    .rename(columns={"precision": "P", "recall": "R", "f1": "F1"})
    .to_string(na_rep="--", float_format=lambda x: f"{x * 100:.1f}")
)

#%%
# =============================================================================
# Table: LLM Mention Attribute Metrics
# =============================================================================

print_section("LLM mention attribute metrics (generic)")

LLM_MENTION_METRICS = ["generic"]

llm_mention_table = pd.DataFrame(
    index=GERFUN_MODELS,
    columns=pd.MultiIndex.from_product([LLM_MENTION_METRICS, ["precision", "recall", "f1"]]),
)

for model in llm_mention_table.index:
    for metric, part in llm_mention_table.columns:
        report = llm_eval_reports[model]["mention_attributes"][metric]
        scores = [report[doc_id][part] for doc_id in ORDERING]
        llm_mention_table.loc[model, (metric, part)] = np.mean(scores)

print(
    llm_mention_table.to_string(na_rep="--", float_format=lambda x: f"{x * 100:.1f}")
)

#%%
# =============================================================================
# Compute LEA Scores per Entity (for detailed plot)
# =============================================================================

print_section("Computing per-entity LEA recall scores")


def _lea_score(input_clusters, output_clusters, mention_to_gold):
    """Compute LEA recall numerator/denominator and per-cluster resolution scores."""
    num, den = 0, 0
    score_per_cluster = {}

    for cluster_id, c in input_clusters.items():
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
                    for m2 in c[i + 1 :]:
                        if m2 in mention_to_gold and (
                            set(mention_to_gold[m]) & set(mention_to_gold[m2])
                        ):
                            common_links += 1

        num += len(c) * common_links / all_links
        den += len(c)
        score_per_cluster[cluster_id] = (common_links / all_links, len(c))

    return num, den, score_per_cluster


LEA_PLOT_DOCS = ["Goethe_Wahlverwandtschaften", "Kürnberger_Amerika"]
LEA_MODEL = "google--gemini-3-flash-preview"

key_entities_map = {}
for doc_id in LEA_PLOT_DOCS:
    key_entities_map[doc_id] = {
        k: v[0][1].entity for k, v in document_entities[doc_id].items()
    }

entity_recall = {}
for doc_id in LEA_PLOT_DOCS:
    key_mentions = document_mentions[doc_id]
    pred_series = pd.read_csv(
        MODEL_OUTPUT_DIR / "merged" / LEA_MODEL / (doc_id + ".tsv"),
        sep="\t", index_col="i", keep_default_na=False,
    )["pred"].sort_index()
    sys_mentions = list(parse_mentions(pred_series))

    key_clusters = mentions_to_clusters(key_mentions, doc_id="doc")
    sys_clusters = mentions_to_clusters(sys_mentions, doc_id="doc")
    key_mention_sys = get_mention_assignments(key_clusters, sys_clusters)

    _, _, per_entity = _lea_score(key_clusters, sys_clusters, key_mention_sys)
    entity_recall[doc_id] = per_entity

print(f"  Computed LEA entity recall for: {LEA_PLOT_DOCS}")

#%%
# =============================================================================
# Figure: LEA Decomposition Plot
# =============================================================================

print_section("Figure: LEA decomposition plot")

MIN_LABEL_RELEVANCE = {
    "Goethe_Wahlverwandtschaften": 200,
    "Kürnberger_Amerika": 500,
    #"Fischer_Gustav": 100,
}
SMALL_GROUPS = {
    "Goethe_Wahlverwandtschaften": [range(2, 21), range(21, 101)],
    "Kürnberger_Amerika": [range(2, 21), range(21, 101), range(101, 251)],
    #"Fischer_Gustav": [range(2, 11), range(11, 51)],
}

fig, axs = plt.subplots(ncols=len(LEA_PLOT_DOCS), figsize=(7, 4))

for ax, doc_id in zip(axs, LEA_PLOT_DOCS):
    color_cycle = itertools.cycle(iter(plt.rcParams["axes.prop_cycle"]))
    recall = entity_recall[doc_id]
    doc_ents = key_entities_map[doc_id]

    generic_ids = {eid for eid in doc_ents if "generic" in doc_ents[eid].specialcase_entity}
    singleton_ids = {eid for eid in doc_ents if recall[eid][1] == 1}
    group_ids = {
        eid for eid in doc_ents if "group" in doc_ents[eid].specialcase_entity
    } - generic_ids
    core_ids = doc_ents.keys() - singleton_ids - group_ids

    # Build bar data
    bar_x, bar_width, bar_height, bar_color, bar_labels = [], [], [], [], []
    cur = 0

    # Large individual entities
    large_ids = {
        eid for eid in core_ids
        if all(recall[eid][1] not in r for r in SMALL_GROUPS[doc_id])
    }
    for eid in sorted(large_ids, key=lambda e: recall[e][1], reverse=True):
        resolution, relevance = recall[eid]
        label = (
            doc_ents[eid].fullname
            if relevance > MIN_LABEL_RELEVANCE[doc_id]
            else ""
        )
        bar_x.append(cur)
        bar_width.append(resolution)
        bar_height.append(relevance)
        bar_labels.append(label)
        bar_color.append(next(color_cycle)["color"])
        cur += relevance

    # Grouped small entities
    for ran in sorted(SMALL_GROUPS[doc_id], key=lambda r: r.start, reverse=True):
        small_ids = {eid for eid in core_ids if recall[eid][1] in ran}
        if not small_ids:
            continue
        relevance = sum(recall[eid][1] for eid in small_ids)
        resolution = (
            sum(recall[eid][0] * recall[eid][1] for eid in small_ids) / relevance
        )
        bar_x.append(cur)
        bar_width.append(resolution)
        bar_height.append(relevance)
        bar_labels.append(f"entities w/ {ran.start}-{ran.stop - 1} mentions")
        bar_color.append("#aaaaaa")
        cur += relevance

    # Non-generic singletons
    ng_singletons = singleton_ids - generic_ids
    if ng_singletons:
        relevance = sum(recall[eid][1] for eid in ng_singletons)
        resolution = sum(recall[eid][0] * recall[eid][1] for eid in ng_singletons) / relevance
        bar_x.append(cur)
        bar_width.append(resolution)
        bar_height.append(relevance)
        bar_labels.append("non-generic singletons")
        bar_color.append("#aaaaaa")
        cur += relevance

    # Generic singletons
    if generic_ids:
        relevance = sum(recall[eid][1] for eid in generic_ids)
        resolution = sum(recall[eid][0] * recall[eid][1] for eid in generic_ids) / relevance
        bar_x.append(cur)
        bar_width.append(resolution)
        bar_height.append(relevance)
        bar_labels.append("generic singletons")
        bar_color.append("#f0a3a3")
        cur += relevance

    # Group entities (non-singleton)
    group_nonsingle = group_ids - singleton_ids
    if group_nonsingle:
        relevance = sum(recall[eid][1] for eid in group_nonsingle)
        resolution = sum(recall[eid][0] * recall[eid][1] for eid in group_nonsingle) / relevance
        bar_x.append(cur)
        bar_width.append(resolution)
        bar_height.append(relevance)
        bar_labels.append("group entities")
        bar_color.append("#a0cfe0")
        cur += relevance

    ax.barh(
        bar_x, height=bar_height, width=bar_width,
        color=bar_color, align="edge", edgecolor="white", linewidth=0.6,
    )
    ax.set_yticks(
        np.array(bar_x) + np.array(bar_height) / 2, bar_labels
    )
    ax.tick_params(axis="y", length=0, labelsize=5)
    ax.set_xlabel("LEA Resolution")
    ax.set_xlim((0, 1))
    ax.set_ylim((cur, 0))
    ax.set_title(DOC_TITLE[doc_id])
    ax.xaxis.set_major_formatter(FuncFormatter(percent_format))

    if ax is axs[0]:
        ax.set_ylabel("LEA Relevance = Entity Size")

plt.tight_layout()
save_figure(fig, "llm_lea_plot.pdf")
plt.show()

#%%
# =============================================================================
# Compute Plural Mention Resolution
# =============================================================================

print_section("Computing plural mention resolution")

plural_rows = []
total_to_process = len(GERFUN_MODELS) * sum(len(m) for m in document_mentions.values())

with tqdm(total=total_to_process, desc="Plural resolution") as pbar:
    for model in GERFUN_MODELS:
        for doc_id, key_mentions in document_mentions.items():
            pred_series = pd.read_csv(
                MODEL_OUTPUT_DIR / "merged" / model / (doc_id + ".tsv"),
                sep="\t", index_col="i", keep_default_na=False,
            )["pred"].sort_index()

            sys_mentions = list(parse_mentions(pred_series))
            key_clusters = mentions_to_clusters(key_mentions, doc_id="doc")
            sys_clusters = mentions_to_clusters(sys_mentions, doc_id="doc")
            entity_mapping = compute_entity_mapping(sys_clusters, key_clusters)

            key_mention_sys = get_mention_assignments(key_clusters, sys_clusters)
            key_mentions_key = get_self_assignments(key_clusters)

            for key_mention, key_cluster_ids in key_mentions_key.items():
                sys_cluster_ids = set(key_mention_sys.get(key_mention, []))
                mapped = {
                    entity_mapping.get(x, f"orphan_{x}") for x in sys_cluster_ids
                }
                plural_rows.append(
                    {
                        "model": model,
                        "doc_id": doc_id,
                        "begin": key_mention.start,
                        "end": key_mention.end,
                        "num_key": len(key_cluster_ids),
                        "num_sys": len(sys_cluster_ids),
                        "num_errors": len(mapped - set(key_cluster_ids)),
                    }
                )
                pbar.update(1)

plural_resolution = pd.DataFrame(plural_rows)

#%%
# =============================================================================
# Table: Plural Resolution Accuracy
# =============================================================================

print_section("Table: Plural mention resolution accuracy")

bins = pd.IntervalIndex.from_breaks([1, 2, 3, 4, np.inf], closed="left")
bin_labels = [
    f"{int(b.left)} ref." if b.right != np.inf else f"≥{int(b.left)} ref."
    for b in bins
]

plural_report_table = plural_resolution.pivot_table(
    values='num_errors',
    aggfunc=lambda x: (x==0).mean(),
    index='model',
    columns=pd.cut(plural_resolution['num_key'], bins)
)

print(
    plural_report_table
        .rename(columns=dict(zip(bins, bin_labels)))
        .to_string()
)

#%%
# =============================================================================
# Figure: Plural Resolution Accuracy
# =============================================================================

print_section("Figure: Plural mention resolution accuracy")

fig, axs = plt.subplots(nrows=2, figsize=(3, 2.15), sharex=True)

for model, accuracy in plural_report_table.iterrows():
    axs[0].plot(accuracy.values, "-o", label=MODEL_LABELS[model])

axs[0].legend(frameon=False)
axs[0].set_xticks(range(len(bins)), bin_labels)
axs[0].set_ylim((0, 1))
axs[0].set_ylabel("Proportion correctly\n resolved in response")
axs[0].yaxis.set_major_formatter(FuncFormatter(percent_format))

# Use last group for the count histogram (all models share same key mentions)
last_group = plural_resolution[plural_resolution["model"] == GERFUN_MODELS[-1]]
counts = last_group.groupby(pd.cut(last_group["num_key"], bins)).size().values
bars = axs[1].bar(range(len(bins)), height=counts, color="#bbbbbb")
axs[1].set_xticks(range(len(bins)), bin_labels)
axs[1].bar_label(bars, label_type="edge")
axs[1].set_ylim((0, 70_000))
axs[1].set_ylabel("Num. key mentions")
axs[1].yaxis.set_major_formatter(FuncFormatter(log_number_fmt))

plt.tight_layout()
save_figure(fig, "llm_plurals.pdf")
plt.show()
