#%%
import collections
import csv
import re

import numpy as np
import warnings
import pandas
import json
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, LogFormatter
import matplotlib
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress
from tqdm import tqdm
import itertools


from llm_literary_coref.eval.evaluator import Evaluator, mentions_to_clusters, get_mention_assignments, compute_entity_mapping, get_self_assignments
from llm_literary_coref.mention import parse_mentions

#%%

matplotlib.rcParams['font.family'] = 'Fira Sans'

from cycler import cycler
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=
[(0.4, 0.7607843137254902, 0.6470588235294118), (0.9882352941176471, 0.5529411764705883, 0.3843137254901961), (0.5529411764705883, 0.6274509803921569, 0.796078431372549), (0.9058823529411765, 0.5411764705882353, 0.7647058823529411), (0.6509803921568628, 0.8470588235294118, 0.32941176470588235), (1.0, 0.8509803921568627, 0.1843137254901961), (0.8980392156862745, 0.7686274509803922, 0.5803921568627451), (0.7019607843137254, 0.7019607843137254, 0.7019607843137254)]
)

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rcParams.update({
    'font.size': SMALL_SIZE,            # Default font size
    'axes.labelsize': SMALL_SIZE,       # Font size for x and y labels
    'axes.titlesize': SMALL_SIZE,      # Font size for main title
    'xtick.labelsize': SMALL_SIZE,      # Font size for x-axis tick labels
    'ytick.labelsize': SMALL_SIZE,      # Font size for y-axis tick labels
    'legend.fontsize': SMALL_SIZE,      # Font size for legend
    'lines.linewidth': 0.8,    # Default line width
    'lines.markersize': 3,     # Default marker size
    'grid.linewidth': 0.3,     # Line width for grid
    'figure.titlesize': MEDIUM_SIZE,    # Font size for `plt.suptitle`
    'axes.linewidth': 0.5,     # Set spine width globally
    'xtick.major.width': 0.5,  # Set x-axis major tick width
    'ytick.major.width': 0.5,  # Set y-axis major tick width
    'xtick.minor.width': 0.3,  # Set x-axis minor tick width
    'ytick.minor.width': 0.3,  # Set y-axis minor tick width
})


#%%


Path('/tmp/jcls_figures').mkdir(exist_ok=True)
try:
    ROOT_DIR = Path(__file__).parent
except NameError:
    ROOT_DIR = Path('.').parent



#%%

doc_title = [
    ('Fischer_Gustav', 'Gustavs Verirrungen'),
    ('Heimburg_Trudchen', 'Trudchens Heirat'),
    ('Wolff_Wildfangrecht', 'Wildfangrecht'),
    ('Goethe_Wahlverwandtschaften', 'Wahlverwandtschaften'),
    ('Kürnberger_Amerika', 'Amerika-Müde'),
]
ordering = [x[0] for x in doc_title]
doc_title = dict(doc_title)
#%%

# load files


annotations_dir = ROOT_DIR / "gerfun_corpus"
document_files = (annotations_dir / "annotated_tsv").glob('*.tsv')

documents = {}
document_mentions = {}
document_entities = {}
for doc in document_files:
    gold = pandas.read_csv(doc, sep='\t', index_col='i', keep_default_na=False)['gold'].sort_index()
    mentions = list(parse_mentions(gold))

    df = pandas.read_csv(doc.parent.parent / "sources" / doc.name, sep='\t', index_col='i', keep_default_na=False, na_values="").sort_index()
    df['gold'] = gold
    documents[doc.stem] = df
    document_mentions[doc.stem] = mentions

    document_entities[doc.stem] = collections.defaultdict(list)
    for mention in mentions:
        for ref in mention.references:
            document_entities[doc.stem][ref.entity.id].append((mention, ref))


#%%


def get_mentions_by_tag(document_id, tag):
    df = documents[document_id]
    out = []
    for mention in document_mentions[document_id]:
        if any(df.loc[mention.token_idx, 'tag'] == tag):
            out.append(mention)
    return out

#%%

## basic statistics

statistics = []
for doc, df in sorted(documents.items(), key=lambda x: len(x[1])):
    statistics.append([doc, 'Num. tokens', len(df)])
    statistics.append([doc, 'Num. sentences', df['is_sent_start'].sum()])
    statistics.append([doc, 'Num. sections', df['is_section_start'].sum()])
    statistics.append([doc, 'Num. mentions', len(document_mentions[doc])])
    statistics.append([doc, 'Num. references', sum(1 for mention in document_mentions[doc] for ref in mention.references)])
    statistics.append([doc, 'Num. entities', len(document_entities[doc])])

df = pandas.DataFrame(statistics, columns=['doc', 'label', 'count']).pivot(index='doc', columns='label').droplevel(0, axis=1)
df = df.rename(doc_title)
df.loc['total'] = df.sum()
df.loc['average'] = df.sum() / len(df)
print(df[['Num. tokens', 'Num. sentences', 'Num. mentions', 'Num. references', 'Num. entities']].to_string(na_rep=''))

#%%

mention_statistics = []
mention_statistics.append(['Num. mentions', sum(len(mentions) for mentions in document_mentions.values())])
mention_statistics.append(['Num. plural mentions', sum(1 for mentions in document_mentions.values() for mention in mentions if len(mention.references) > 1)])
mention_statistics.append(['Num. proper noun mentions', sum(1 for doc in documents.keys() for mention in get_mentions_by_tag(doc, 'NE'))])
mention_statistics.append(['Num. nominal noun mentions', sum(1 for doc in documents.keys() for mention in get_mentions_by_tag(doc, 'NN'))])
mention_statistics.append(['Num. mentions with figurative reference', sum(1 for mentions in document_mentions.values() for mention in mentions if any('figurative' in ref.specialcase_reference for ref in mention.references))])
mention_statistics.append(['Num. mentions with part reference', sum(1 for mentions in document_mentions.values() for mention in mentions if any('part' in ref.specialcase_reference for ref in mention.references))])

df = pandas.DataFrame(mention_statistics, columns=['label', 'count']).set_index('label')
df['average'] = df['count'] / len(documents)

df['proportion'] = df['count'] / df['count'].iloc[0]

print(df[['count', 'average', 'proportion']].to_string(na_rep=''))

#%%


entity_statistics = []
entity_statistics.append(['Num. entities', sum(len(entities) for entities in document_entities.values())])
entity_statistics.append(['Num. non-singleton entities', sum(1 for entities in document_entities.values() for references in entities.values() if len(references) > 1)])
entity_statistics.append(['Num. non-singleton non-specialcase entities', sum(1 for entities in document_entities.values() for references in entities.values() if len(references) > 1 and references[0][1].entity.specialcase_entity == [])])
entity_statistics.append(['Num. singleton entities', sum(1 for entities in document_entities.values() for references in entities.values() if len(references) == 1)])
entity_statistics.append(['Num. non-generic singleton entities', sum(1 for entities in document_entities.values() for references in entities.values() if len(references) == 1 and 'generic' not in references[0][1].entity.specialcase_entity)])
entity_statistics.append(['Num. generic entities', sum(1 for entities in document_entities.values() for references in entities.values() if 'generic' in references[0][1].entity.specialcase_entity)])
entity_statistics.append(['Num. group entities', sum(1 for entities in document_entities.values() for references in entities.values() if 'group' in references[0][1].entity.specialcase_entity)])
entity_statistics.append(['Num. group singleton entities', sum(1 for entities in document_entities.values() for references in entities.values() if len(references) == 1 and 'group' in references[0][1].entity.specialcase_entity)])
entity_statistics.append(['Num. nonfact entities', sum(1 for entities in document_entities.values() for references in entities.values() if 'nonfact' in references[0][1].entity.specialcase_entity)])
entity_statistics.append(['Num. nonfact singleton entities', sum(1 for entities in document_entities.values() for references in entities.values() if len(references) == 1 and 'nonfact' in references[0][1].entity.specialcase_entity)])

df = pandas.DataFrame(entity_statistics, columns=['label', 'count']).set_index('label')
df['average'] = df['count'] / len(documents)
df['proportion'] = df['count'] / df['count'].iloc[0]

print(df[['count', 'average', 'proportion']].to_string(na_rep=''))

#%%

generic_table = pandas.DataFrame(index=ordering, columns=['generic singletons', 'non-generic singletons'])
for doc_id, entities in document_entities.items():
    num_entities = len(entities)
    generic_table.loc[doc_id, 'generic singletons'] = sum(1 for references in entities.values() if 'generic' in references[0][1].entity.specialcase_entity) / len(entities)
    generic_table.loc[doc_id, 'non-generic singletons'] = sum(1 for references in entities.values() if len(references) == 1 and 'generic' not in references[0][1].entity.specialcase_entity) / len(entities)

print(generic_table.loc[['Fischer_Gustav', 'Goethe_Wahlverwandtschaften', 'Kürnberger_Amerika', 'Heimburg_Trudchen', 'Wolff_Wildfangrecht'],:].to_string())

#%%

gender_statistics = []
for entities in document_entities.values():
    for references in entities.values():
        e = references[0][1].entity
        gender_statistics.append((e.gender, 'generic' in e.specialcase_entity, 'group' in e.specialcase_entity, len(references)))

gender_statistics = pandas.DataFrame(gender_statistics, columns=['gender', 'generic', 'group', 'num_references'])
gender_statistics.loc[gender_statistics['gender'].apply(lambda x: 'o' in x or 'u' in x), 'gender'] = 'u'

gender_by_entity = pandas.pivot_table(gender_statistics, index=['gender'], columns=['generic', 'group'], aggfunc=len)
total_num = gender_by_entity.sum().sum()
gender_by_entity.loc[:, 'Overall'] = gender_by_entity.sum(axis=1)

gender_by_entity = pandas.concat({'count': gender_by_entity, 'percent': 100*gender_by_entity / total_num}, axis=1)
gender_by_entity = gender_by_entity.reorder_levels([1,2,3,0], axis=1).sort_index(level=[1,2], axis=1)
print(gender_by_entity.to_string())


#%%

## calculate entity sizes

entity_sizes = []
for doc, entities in document_entities.items():
    for entity_id, mentions in entities.items():
        entity = mentions[0][1].entity
        entity_sizes.append((doc, entity.fullname, len(mentions), 'generic' in entity.specialcase_entity, 'group' in entity.specialcase_entity))

entity_sizes = pandas.DataFrame(entity_sizes, columns=['doc', 'entity', 'num_references', 'generic', 'group'])

statistics = []
statistics.append(['All entities', len(entity_sizes), sum(entity_sizes['num_references'])])
sel = ~entity_sizes['group']&(entity_sizes['num_references'] > 1)
statistics.append(['Non-singleton individuals', len(entity_sizes.loc[sel]), sum(entity_sizes.loc[sel, 'num_references'])])
sel = (entity_sizes['group'])
statistics.append(['Group entities', len(entity_sizes.loc[sel]), sum(entity_sizes.loc[sel, 'num_references'])])
sel = entity_sizes['generic']
statistics.append(['Generic singletons', len(entity_sizes.loc[sel]), sum(entity_sizes.loc[sel, 'num_references'])])
sel = (~entity_sizes['generic'])&(entity_sizes['num_references'] == 1)
statistics.append(['Non-generic singletons', len(entity_sizes.loc[sel]), sum(entity_sizes.loc[sel, 'num_references'])])

df = pandas.DataFrame(statistics, columns=['label', 'number entities', 'total number references'])

df['proportion of entities'] = df['number entities'] / df.iloc[0]['number entities']
df['proportion of references'] = df['total number references'] / df.iloc[0]['total number references']

print(df.to_string())


#%%

# Zipf Plot


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(5.4, 3.0), dpi=300)

ax = ax1
cycler = iter(plt.rcParams['axes.prop_cycle'])
for doc, g in sorted(entity_sizes.groupby('doc'), key=lambda x: ordering.index(x[0])):
    sel = ~entity_sizes['group'] & (entity_sizes['num_references'] > 1)
    g = g.loc[sel]
    g = g.sort_values('num_references', ascending=False)

    g = g[['entity', 'num_references']]
    g['rank'] = np.arange(1, len(g) + 1)

    x =g['rank']
    y = g['num_references']

    color = next(cycler)['color']
    ax.scatter(x, y, marker='o', facecolors='none', edgecolors=color, linewidth=plt.rcParams["lines.linewidth"], alpha=0.5)

    # Calculate trend line using log-transformed values
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

    # Plot trend line
    x_line = np.array([g['rank'].min(), g['rank'].max()])
    y_line = np.exp(intercept + slope * np.log(x_line))
    ax.plot(x_line, y_line, '--', color=color, alpha=0.7, label=f'${np.exp(intercept)/1000:.2f}\\cdot 10^3 \\cdot x^{{{slope:.3g}}}$')

ax.set_xscale('log')
ax.set_yscale('log')

def log_number_fmt(x, pos=None):
    if 1 <= x <= 100:
        return f"{x:.0f}"
    elif 100 < x <= 1_000_000:
        return f"{x/1000:.0f}K"
    elif 1_000_000 < x:
        return f"{x/1_000_000:.0f}M"
    elif 0.01 <= x < 1:
        return f"{x}"
    else:
        return f"{x:.2g}"
        # fx = np.log10(x)
        # return f"$\mathdefault{{10^{{{fx}}}}}$"

ax.set_xlabel('Entity rank')
ax.set_ylabel('Number of references')
ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
ax.yaxis.set_major_formatter(FuncFormatter(log_number_fmt))
ax.legend(frameon=False)

# Zipf Plot, cumulative and proportional


ax = ax2
cycler = iter(plt.rcParams['axes.prop_cycle'])
for doc, doc_df in entity_sizes.groupby('doc'):
    sel = ~entity_sizes['group'] & (entity_sizes['num_references'] > 1)
    g = doc_df.loc[sel]
    # g = doc_df
    g = g.sort_values('num_references', ascending=False)

    g = g[['entity', 'num_references']]
    g['rank'] = np.arange(1, len(g) + 1)

    x = g['rank']#/(len(doc_df)+1)
    y = g['num_references'].cumsum()/sum(doc_df['num_references'])

    color = next(cycler)['color']
    ax.scatter(x, y, marker='o', facecolors='none', edgecolors=color, label=doc_title[doc], linewidth=plt.rcParams["lines.linewidth"], alpha=0.5)

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('Entity rank')
ax.set_ylabel('Cumulative Proportion of references')


def percent_format(x, pos=None):
    if x > 0.01:
        return f'{x * 100:.0f}%'
    else:
        return f'{x * 100:.3g}%'


# ax.xaxis.set_major_formatter(FuncFormatter(percent_format))
ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
ax.yaxis.set_major_formatter(FuncFormatter(percent_format))
ax.yaxis.set_minor_formatter(FuncFormatter(percent_format))

ax.legend(frameon=False)
plt.tight_layout()
plt.savefig('/tmp/jcls_figures/entity_sizes.pdf')
plt.show()


#%%

## Spread

spreads = []
for doc, entities in document_entities.items():
    for entity_id, mentions in entities.items():
        entity = mentions[0][1].entity
        start = min(i for mention, _ in mentions for i in mention.token_idx)
        end = max(i for mention, _ in mentions for i in mention.token_idx)
        spreads.append((doc, entity.fullname, len(mentions), end-start, 'generic' in entity.specialcase_entity, 'group' in entity.specialcase_entity))

spreads = pandas.DataFrame(spreads, columns=['doc', 'entity', 'num_references', 'spread', 'generic', 'group'])

print(spreads[spreads['num_references'] > 1]['spread'].describe(percentiles=[0.5, 0.90]))

#%%

fig, ax_grid = plt.subplots(ncols=2, nrows=len(ordering)//2+(1 if len(ordering)%2>0 else 0), sharex=True, sharey=True, figsize=(5.3, 2.4*len(ordering)//2))

axs = ax_grid.flatten()

cycler = iter(plt.rcParams['axes.prop_cycle'])
for i, (ax, (doc, doc_df)) in enumerate(zip(axs, sorted(spreads.groupby('doc'), key=lambda x: ordering.index(x[0])))):
    sel = (doc_df['num_references'] > 1)
    g = doc_df.loc[sel]

    x = g['num_references']#/sum(doc_df['num_references'])
    y = g['spread']#/len(documents[doc])
    color = next(cycler)['color']
    ax.scatter(x, y, marker='o', facecolors='none', edgecolors=color, alpha=0.7)
    ax.axhline(len(documents[doc]), ls='--', color=color, alpha=0.5, linewidth=1.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(log_number_fmt))
    ax.xaxis.set_major_formatter(FuncFormatter(log_number_fmt))
    ax.set_title(doc_title[doc])
    ax.set_ylabel('Spread in Tokens')
    ax.set_xlabel('Number of References')

for ax in axs[len(ordering):]:
    fig.delaxes(ax)

for ax in axs[len(ordering)-2:len(ordering)]:
    ax.tick_params(labelbottom=True)


plt.tight_layout()
plt.savefig('/tmp/jcls_figures/entity_spread.pdf')
plt.show()

#%%

## Mention Distance

mention_distances = []

total_mentions = sum(len(mentions) for entities in document_entities.values() for mentions in entities.values())

with tqdm(total=total_mentions) as pbar:
    for doc, entities in document_entities.items():
        for entity_id, mentions in entities.items():
            doc_df = documents[doc]
            # chapter_id = doc_df['is_section_start'].cumsum()
            mentions = list(sorted((mention for mention, _ in mentions), key=lambda x: x.token_idx[0]))
            mention_positions = [mention.token_idx[0] for mention in mentions]
            # mention_chapter = [chapter_id.loc[x] for x in mention_positions]
            mention_type = ['NE' if 'NE' in list(doc_df.loc[mention.token_idx, 'tag']) else
                            'NN' if 'NN' in list(doc_df.loc[mention.token_idx, 'tag']) else
                            doc_df.loc[mention.token_idx[0], 'tag']
                            for mention in mentions]

            distances = pandas.DataFrame({'position': mention_positions, 'type': mention_type})
            distances['doc'] = doc
            distances['entity'] = entity_id
            pbar.update(len(distances))
            if len(distances) == 1:
                continue
            if not (distances['type'] == 'NE').any():
                continue


            distances['distance_to_previous_mention'] = distances['position'].diff()
            distances['distance_to_previous_ne'] = distances['position'] - distances[distances['type'] == 'NE'].reindex(distances.index)['position'].ffill()
            distances['distance_to_next_mention'] = -distances['position'].diff(-1)
            distances['distance_to_next_ne'] =  distances[distances['type'] == 'NE'].reindex(distances.index)['position'].bfill() - distances['position']

            distances['dist_to_mention'] = distances[['distance_to_previous_mention', 'distance_to_next_mention']].min(axis=1)
            distances['dist_to_ne'] = distances[['distance_to_previous_ne', 'distance_to_next_ne']].min(axis=1)

            mention_distances.append(distances)

mention_distances = pandas.concat(mention_distances)

#%%

df = mention_distances[mention_distances['type'] != 'NE'].dropna()

print(df.describe(percentiles=[0.5, 0.9, .95, .99, .999]).to_string())

#%%

x = [1,2,5] + list(np.logspace(np.log10(10), np.log10(30000), 25))

cycler = iter(plt.rcParams['axes.prop_cycle'])
fig, ax = plt.subplots(figsize=(5, 2))
for doc, g in sorted(df.groupby('doc'), key=lambda x: ordering.index(x[0])):
    surv = pandas.DataFrame(index=x)
    surv['dist_to_mention'] = np.nan
    surv['dist_to_ne'] = np.nan
    for x_ in x:
        surv.loc[x_] = (g[['dist_to_mention', 'dist_to_ne']] >= x_).mean()

    surv = surv.replace(0, np.nan)

    color = next(cycler)['color']
    ax.plot(x, surv['dist_to_mention'], '-', label=doc_title[doc], alpha=0.8, lw=1.4, color=color)
    ax.plot(x, surv['dist_to_ne'], '--', alpha=0.8, lw=1.4, color=color)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Distance in Tokens')
ax.set_ylabel('Prop. Mentions')
# ax.legend(frameon=False)
def percent_format(x, pos=None):
    if x > 0.01:
        return f'{x * 100:.0f}%'
    else:
        return f'{x * 100:.3g}%'

ax.xaxis.set_major_formatter(FuncFormatter(log_number_fmt))
ax.yaxis.set_major_formatter(FuncFormatter(percent_format))

ax.legend(frameon=False)
plt.tight_layout()
plt.savefig('/tmp/jcls_figures/mention_distance.pdf')
plt.show()


#%%

## IAA
with open(ROOT_DIR / 'gerfun_corpus' / 'evaluation_reports' / 'iaa_report.json') as f:
    iaa_report = json.load(f)

#%%

cluster_metrics = list(iaa_report['clusters_all'].keys())

cluster_variants = {
    "all": "full",
    "replaceplural": "plurals replaced",
    "nogeneric": "w/o generics",
    "nosingletons": "w/o singletons",
}


iaa_cluster_table = pandas.DataFrame(index=['all', 'replaceplural', 'nogeneric', 'nosingletons'],
                                 columns=['mentions', 'muc', 'bcub', 'ceafe', 'conll', 'ceafm', 'lea'])

for variant in iaa_cluster_table.index:
    report = iaa_report[f'clusters_{variant}']

    for metric, scores in report.items():
        iaa_cluster_table.loc[variant, metric] = scores['aggregated']['f1']

iaa_cluster_table['conll'] = iaa_cluster_table[['muc', 'bcub', 'ceafe']].mean(axis=1)

print(iaa_cluster_table.rename(cluster_variants).to_string(float_format=lambda x: f"{x*100:.2f}"))

#%%

entity_variants = {
    "all": "full",
    "nogroup": "w/o groups",
    "nogroupnosingletons": "w/o groups, singletons",
}

entity_metrics = ['gender:m', 'gender:f', 'gender:u', 'gender:nb', 'gender:mf', 'generic', 'group', 'nonfact']

entity_table = pandas.DataFrame(index=pandas.MultiIndex.from_product([['all', 'nogroup', 'nogroupnosingletons'], ['f1', 'count']]),
                                 columns=entity_metrics)

for variant in entity_table.index.levels[0]:
    report = iaa_report[f'entity_attributes_{variant}_restrictonmatch']

    for metric, scores in report.items():
        counts = list(sorted([scores['aggregated']['support_key'], scores['aggregated']['support_response']]))
        count_str = '+'.join(map(str, counts))

        if sum(counts) > 0:
            entity_table.loc[(variant, 'f1'), metric] = scores['aggregated']['f1']
            entity_table.loc[(variant, 'count'), metric] = count_str

print(entity_table.rename(entity_variants).to_string(na_rep='--', float_format=lambda x: f"{x*100:.2f}"))

#%%

mention_metrics = ['part', 'figurative', 'generic']
mention_table = pandas.DataFrame(index=mention_metrics, columns=['f1', 'count'])
for metric in mention_table.index:
    scores = iaa_report['mention_attributes'][metric]['aggregated']
    counts = list(sorted([scores['support_key'], scores['support_response']]))
    count_str = '+'.join(map(str, counts))

    if sum(counts) > 0:
        mention_table.loc[metric, 'count'] = count_str
        mention_table.loc[metric, 'f1'] = scores['f1']

print(mention_table.to_string(na_rep='--', float_format=lambda x: f"{x*100:.2f}"))

#%%

## Difference to silver pre-annotation

num_chapters = sum(doc_df['is_section_start'].sum() for doc_df in documents.values())

scores = []
with tqdm(total=num_chapters) as pbar:
    for doc, doc_df in documents.items():
        source_df = pandas.read_csv(annotations_dir / "sources" / (doc + '.tsv'), sep='\t', keep_default_na=False)
        pre_annotations = source_df['llm_pre_annotation']
        chap_ids = source_df['is_section_start'].cumsum()
        for chap_id, chap in doc_df.groupby(chap_ids):
            gold_mentions = list(parse_mentions(chap['gold']))
            pre_annotated_mentions = list(parse_mentions(pre_annotations.loc[chap.index]))

            evaluator = Evaluator()
            evaluator.add_document(key_mentions=gold_mentions, sys_mentions=pre_annotated_mentions)
            report = evaluator.report(as_dict=True)
            scores.append((doc, chap_id, report['clusters_replaceplural']))
            pbar.update(1)


#%%

scores_df = pandas.DataFrame(
    index=pandas.MultiIndex.from_tuples(x[:2] for x in scores),
    columns=pandas.MultiIndex.from_product([['mentions', 'muc', 'bcub', 'ceafe', 'conll', 'ceafm', 'lea'], ['precision', 'recall', 'f1']]))
for doc, chap_id, report in scores:
    for (metric, t) in scores_df.columns:
        if metric == 'conll': continue
        scores_df.loc[(doc, chap_id), (metric, t)] = report[metric]['doc'][t]

    scores_df.loc[(doc, chap_id), ('conll', 'f1')] = np.mean([report[m]['doc']['f1'] for m in ['muc', 'bcub', 'ceafe']])

#%%

aggregated_scores_df = scores_df.aggregate(['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
                                           axis=0)
aggregated_scores_df.index = ['median', '25%', '75%']
aggregated_scores_df.loc['25%'] = aggregated_scores_df.loc['25%'] - aggregated_scores_df.loc['median']
aggregated_scores_df.loc['75%'] = aggregated_scores_df.loc['75%'] - aggregated_scores_df.loc['median']
aggregated_scores_df = aggregated_scores_df.T.unstack().reorder_levels([1, 0], axis=1)
print(aggregated_scores_df)

aggregated_scores_df = aggregated_scores_df.loc[['mentions', 'muc', 'bcub', 'ceafe', 'conll', 'ceafm', 'lea'], (['precision', 'recall', 'f1'])]
print(aggregated_scores_df.to_string(na_rep='--', float_format=lambda x: f"{x*100:.2f}"))


#%%

## DROC Performance + IAA
model_output_dir = ROOT_DIR / "llm_outputs"

models = ['google--gemini-2.5-flash-lite', 'google--gemini-2.5-flash']

droc_model_scores = {}

scores = []
for model in models:
    all_docs = list((model_output_dir / "droc_test_set" / model).glob("*.tsv"))

    for doc in all_docs:
        if 'RATER1' in doc.name:
            other_doc = doc.parent / doc.name.replace('RATER1', 'RATER2')
        else:
            other_doc = doc.parent / doc.name.replace('RATER2', 'RATER1')
        doc_df = pandas.read_csv(doc, sep='\t', keep_default_na=False)
        other_doc_df = pandas.read_csv(other_doc, sep='\t', keep_default_na=False)
        gold_mentions = list(parse_mentions(other_doc_df['gold']))
        pred_mentions = list(parse_mentions(doc_df['pred']))

        evaluator = Evaluator()
        evaluator.add_document(key_mentions=gold_mentions, sys_mentions=pred_mentions)
        report = evaluator.report(as_dict=True)
        scores.append((model, doc.name, report['clusters_all']))

#%%

# human vs human
all_docs = list((ROOT_DIR / "droc_test_set").glob("*.tsv"))
for doc in all_docs:
    if 'RATER1' in doc.name:
        other_doc = doc.parent / doc.name.replace('RATER1', 'RATER2')
    else:
        continue
    doc_df = pandas.read_csv(doc, sep='\t', keep_default_na=False)
    other_doc_df = pandas.read_csv(other_doc, sep='\t', keep_default_na=False)

    gold_mentions = list(parse_mentions(doc_df['gold']))
    pred_mentions = list(parse_mentions(other_doc_df['gold']))

    evaluator = Evaluator()
    evaluator.add_document(key_mentions=gold_mentions, sys_mentions=pred_mentions)
    report = evaluator.report(as_dict=True)
    scores.append(('iaa', doc.name, report['clusters_all']))

#%%

scores_df = pandas.DataFrame(
    index=pandas.MultiIndex.from_tuples([x[:2] for x in scores], names=['model', 'doc']),
    columns=pandas.MultiIndex.from_product([['mentions', 'muc', 'bcub', 'ceafe', 'conll', 'ceafm', 'lea'], ['precision', 'recall', 'f1']]))

for model, doc, report in scores:
    for (metric, t) in scores_df.columns:
        if metric == 'conll': continue
        if t != 'f1' and model == 'iaa': continue
        scores_df.loc[(model, doc), (metric, t)] = report[metric]['doc'][t]

    scores_df.loc[(model, doc), ('conll', 'f1')] = np.mean([report[m]['doc']['f1'] for m in ['muc', 'bcub', 'ceafe']])

#%%

def lower_quartile(x): return (x.quantile(0.25) - x.median()) if len(x) > 0 else np.nan
def upper_quartile(x): return (x.quantile(0.75) - x.median()) if len(x) > 0 else np.nan

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    aggregated_scores_df = scores_df.reset_index().groupby('model').apply(lambda x: x.drop('doc', axis=1).agg(['median', lower_quartile, upper_quartile]))

aggregated_scores_df = aggregated_scores_df.unstack().loc[:, (['muc', 'bcub', 'ceafe', 'conll', 'ceafm', 'lea'], 'f1')]
print(aggregated_scores_df.to_string(na_rep='--', float_format=lambda x: f"{x*100:.2f}"))

#%%

boxplot_kwargs = dict(patch_artist=True, boxprops=dict(fc='black', edgecolor='black'), widths=.07,
                      medianprops=dict(solid_capstyle='projecting', color='white'), showcaps=False,
                      flierprops=dict(markersize=4, markeredgewidth=.6))

metrics = ['conll', 'lea']
metric_labels = {'conll': "CoNLL", 'lea': "LEA"}
models = ['google--gemini-2.5-flash-lite', 'google--gemini-2.5-flash', 'iaa']
model_labels = {'google--gemini-2.5-flash-lite': "Gemini 2.5 Flash Lite",
                'google--gemini-2.5-flash': "Gemini 2.5 Flash",
                'iaa': "Human vs. Human"}

maverick_baseline_scores = pandas.Series({
    'conll': .7375,
    'lea': .6777,
})


fig, axs = plt.subplots(nrows=len(metrics), figsize=(4, 3.0), dpi=300)
for ax, metric in zip(axs, metrics):
    ticks = np.arange(len(models))
    X = [100*scores_df.loc[m, (metric, 'f1')].values for m in models]
    ax.boxplot(X, positions=ticks, **boxplot_kwargs, orientation='horizontal')

    ax.axvline(np.median(X[-1]), ls='--', color='black')
    ax.axvline(100*maverick_baseline_scores.loc[metric], ls='--', color='#ee7777')
    ax.yaxis.set_inverted(True)
    ax.set_xlabel(metric_labels[metric])
    ax.set_yticks(ticks, [model_labels[m] for m in models])

plt.tight_layout(h_pad=3)
plt.savefig('/tmp/jcls_figures/droc_performance.pdf')
plt.show()

#%%

## LLM Evaluation


metrics = ['conll', 'lea']
metric_labels = {'conll': "CoNLL", 'lea': "LEA"}
models = [
    # 'qwen--qwen3-30b-a3b-instruct-2507',
    # 'qwen--qwen3-vl-235b-a22b-instruct',
    'google--gemini-2.5-flash-lite',
    # 'google--gemini-2.5-flash',
    'google--gemini-3-flash-preview',
]
model_labels = {'google--gemini-2.5-flash-lite': "Gemini 2.5 Flash Lite",
                # 'google--gemini-2.5-flash': "gemini-2.5-flash",
                'google--gemini-3-flash-preview': "Gemini 3 Flash",
                # 'qwen--qwen3-30b-a3b-instruct-2507': 'qwen3-30b-a3b-instruct',
                # 'qwen--qwen3-vl-235b-a22b-instruct': 'qwen3-vl-235b-a22b-instruct',
                }

llm_eval_reports = {}
for model_dir in (ROOT_DIR / "llm_outputs" / "merged").iterdir():
    if not model_dir.is_dir():
        continue
    if not (model_dir / "evaluation_report.json").is_file():
        print('warn: no evaluation file for', model_dir)
        continue
    model_name = model_dir.name
    with open(model_dir / "evaluation_report.json") as f:
        llm_eval_reports[model_name] = json.load(f)

#%%

cluster_variants = {
    "all": "full",
    "replaceplural": "plurals replaced",
    "nogeneric": "w/o generics",
    "nosingletons": "w/o singletons",
}

# all_document_ids = [ x.replace('_', '-') for x in doc_title.keys() ]

cluster_table = pandas.DataFrame(index=pandas.MultiIndex.from_product([['all', 'nogeneric', 'nosingletons', "replaceplural"], models, ordering], names=['variant', 'model', 'doc']),
                                 columns=pandas.MultiIndex.from_product(
                                     [['mentions', 'muc', 'bcub', 'ceafe', 'conll', 'ceafm', 'lea'],
                                      ['precision', 'recall', 'f1']]))

for (variant, model, doc_id) in cluster_table.index:
    report = llm_eval_reports[model][f'clusters_{variant}']

    for metric, t in cluster_table.columns:
        if metric == 'conll': continue
        if doc_id not in report[metric].keys(): continue
        cluster_table.loc[(variant, model, doc_id), (metric, t)] = report[metric][doc_id][t]

    cluster_table.loc[(variant, model, doc_id), ('conll', 'f1')] = cluster_table.loc[(variant, model, doc_id), (['muc', 'bcub', 'ceafe'], 'f1')].mean()

#%%

print(cluster_table.groupby(level=[0,1]).mean().rename(cluster_variants).rename({'precision': 'P', 'recall': 'R', 'f1': 'F1'}, axis=1).to_string(na_rep='--', float_format=lambda x: f"{x*100:.1f}"))


#%%

handles = []

fig, axs = plt.subplots(ncols=len(metrics), nrows=2, figsize=(5.0, 2.5), dpi=300, height_ratios=[1,0])
for ax, metric in zip(axs[0], metrics):
    tick_pos = []
    for j, model in enumerate(models):
        cycler = iter(plt.rcParams['axes.prop_cycle'])
        ax.set_ylabel(metric_labels[metric])
        start_x = (len(ordering) + 0.3)*j
        for i, doc in enumerate(ordering):
            color = next(cycler)['color']
            X = 100*cluster_table.loc[('all', model, doc), (metric, 'f1')]
            l = ax.scatter(start_x + i, X, label=doc_title[doc], color=color)
            handles.append(l)

        avg = 100 * cluster_table.loc[('all', model, ordering), (metric, 'f1')].mean()
        l, = ax.plot([start_x, start_x + len(ordering)], [avg, avg], color='black', ls='--', label='avg', alpha=0.5)
        handles.append(l)
        tick_pos.append(start_x + len(ordering)/2)
    ax.set_xticks(tick_pos, [model_labels[x] for x in models])
    ax.set_ylim((30, 100))

# axs[1].legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.25),
#              handles=handles[:len(ordering)+1])
gs = axs[0, 0].get_gridspec()
for ax in axs[1,:]:
    ax.remove()

axbig = fig.add_subplot(gs[1,:])
axbig.spines['top'].set_visible(False)
axbig.spines['right'].set_visible(False)
axbig.spines['bottom'].set_visible(False)
axbig.spines['left'].set_visible(False)
axbig.get_xaxis().set_ticks([])
axbig.get_yaxis().set_ticks([])

axbig.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.75),
             handles=handles[:len(ordering)+1])

plt.tight_layout()
plt.savefig('/tmp/jcls_figures/gerfun_performance.pdf')
plt.show()


#%%

# Per-segment performance

segment_lengths = {}
for doc_id in documents.keys():
    for f in (ROOT_DIR / "llm_outputs" / "mention_detection").glob(f'{doc_id}*.tsv'):
        segment_lengths[f.stem] = len(pandas.read_csv(f, sep='\t'))

fig, axs = plt.subplots(ncols=2, sharey=True)

ax = axs[0]
for model in models:
    lea_reports = llm_eval_reports[model]['clusters_all']['lea']
    segments = list(sorted(x for x in lea_reports.keys() if 'segment' in x))
    lea_f_scores = [lea_reports[c]['f1'] for c in segments]

    doc_length = []
    for label in segments:
        doc_length.append(segment_lengths[label])

    # Add linear regression
    slope, intercept, r_value, p_value, std_err = linregress(doc_length, lea_f_scores, alternative='less')
    x_reg = np.array([min(doc_length), max(doc_length)])
    y_reg = slope * x_reg + intercept

    ax.scatter(doc_length, lea_f_scores, label=f"{model_labels.get(model, model)} (p={p_value:.3g})")
    ax.plot(x_reg, y_reg, '--', alpha=0.5)

ax.set_xlabel('Segment Length (tokens)')
ax.set_ylabel('LEA F1 Score')
ax.legend(frameon=False)

ax = axs[1]
for model in models:
    lea_reports = llm_eval_reports[model]['clusters_all']['lea']
    segments = list(sorted(x for x in lea_reports.keys() if 'segment' in x))
    lea_f_scores = [lea_reports[c]['f1'] for c in segments]

    boxplot_data = [lea_f_scores]
    positions = [list(model_labels.keys()).index(model)]

    ax.boxplot(boxplot_data, positions=positions)

ax.set_ylim((0.5, 1))
ax.set_xticks(range(len(models)), [model_labels[m] for m in models])

plt.show()

#%%

df = pandas.DataFrame(index=pandas.MultiIndex.from_product([segment_lengths.keys(), models]), columns=pandas.MultiIndex.from_product([[], []]))
for segment_id, model in df.index:
    reports = llm_eval_reports[model]
    for k, v in reports.items():
        if 'cluster' not in k: continue
        for metric, scores in v.items():
            if segment_id in scores.keys():
                df.loc[(segment_id, model), (k, metric)] = scores[segment_id]['f1']

# df['conll'] = df[['muc', 'bcub', 'ceafe']].mean(axis=1)

print(df.groupby(level=1).apply(lambda x: x.describe()).to_string())


#%%

# Inference Cost

cost_overview = pandas.DataFrame(index=pandas.MultiIndex(levels=[[], [], []], codes=[[], [], []]), columns=['prompt_tokens', 'completion_tokens', 'cost'])

for model in llm_eval_reports.keys():
    for pred_json_file in list((ROOT_DIR / "llm_outputs" / "predicted_sections" / model.replace('/', '--')).glob('*.json'))\
            + list((ROOT_DIR / "llm_outputs" / "merged" / model.replace('/', '--')).glob('*.json')):

        if 'evaluation_report' in pred_json_file.stem: continue
        with open(pred_json_file) as f:
            obj = json.load(f)

        is_merge = 'merged' in str(pred_json_file)

        segment_length = len(pandas.read_csv(pred_json_file.parent / (pred_json_file.stem + ".tsv"), sep='\t'))

        for i, det in enumerate(obj['generation_details']):
            usage = det['usage']
            if usage is None:
                print(f'for {pred_json_file}, request {i}, no usage info')
            cost_overview.loc[(model, pred_json_file.stem, i), 'cost'] = usage['cost'] if usage else np.nan
            cost_overview.loc[(model, pred_json_file.stem, i), 'segment_length'] = segment_length if usage and not is_merge else np.nan
            cost_overview.loc[(model, pred_json_file.stem, i), 'prompt_tokens'] = usage['prompt_tokens'] if usage else np.nan
            cost_overview.loc[(model, pred_json_file.stem, i), 'completion_tokens'] = usage['completion_tokens'] if usage else np.nan

cost_overview = cost_overview.astype('float32')

total_cost_overview = cost_overview.groupby(level=0).sum()
print(total_cost_overview.to_string())
print(total_cost_overview.apply(lambda x: x / x['segment_length'], axis=1).to_string(float_format="%.4g"))
print((cost_overview[['prompt_tokens', 'completion_tokens']].sum() / cost_overview['segment_length'].sum()).to_string(float_format="%.4g"))

#%%

# entity / mention metrics

entity_variants = {
    "all": "full",
    "nogroup": "w/o groups",
    "nogroupnosingletons": "w/o groups, singletons",
}

entity_metrics = ['gender:m', 'gender:f', 'gender:u', 'group', 'nonfact']

entity_table = pandas.DataFrame(index=pandas.MultiIndex.from_product([['all', 'nogroup', 'nogroupnosingletons'], models]),
                                columns=pandas.MultiIndex.from_product([entity_metrics, ['precision', 'recall', 'f1']]))

for variant, model in entity_table.index:
    for metric, part in entity_table.columns:
        report = llm_eval_reports[model][f'entity_attributes_{variant}_restrictonmatch'][metric]
        scores = [report[x][part] for x in ordering]
        entity_table.loc[(variant, model), (metric, part)] = np.mean(scores)


print(entity_table.rename(entity_variants).rename({'precision': 'P', 'recall': 'R', 'f1': 'F1'}, axis=1).to_string(na_rep='--', float_format=lambda x: f"{x*100:.1f}"))

#%%

mention_metrics = ['generic']
mention_table = pandas.DataFrame(index=models, columns=pandas.MultiIndex.from_product([mention_metrics, ['precision', 'recall', 'f1']]))
for model in mention_table.index:
    for metric, part in mention_table.columns:
        report = llm_eval_reports[model]['mention_attributes'][metric]
        scores = [report[x][part] for x in ordering]
        mention_table.loc[model, (metric, part)] = np.mean(scores)

print(mention_table.to_string(na_rep='--', float_format=lambda x: f"{x*100:.1f}"))


#%%

## LEA Plot

def _lea_score(input_clusters, output_clusters,
               mention_to_gold):
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
                    for m2 in c[i + 1:]:
                        if m2 in mention_to_gold:
                            if set(mention_to_gold[m]) & set(mention_to_gold[m2]):
                                common_links += 1
        num += len(c) * common_links / all_links
        den += len(c)
        score_per_cluster[cluster_id] = (common_links / all_links, len(c))
    return num, den, score_per_cluster


to_plot = ['Goethe_Wahlverwandtschaften', 'Kürnberger_Amerika']
key_entities = {}
for doc_id in to_plot:
    key_entities[doc_id] = {k: v[0][1].entity for k, v in document_entities[doc_id].items()}

lea_model = "google--gemini-3-flash-preview"

entity_recall = {}
for doc_id in to_plot:
    key_mentions = document_mentions[doc_id]
    df = pandas.read_csv(ROOT_DIR / "llm_outputs" / "merged" / lea_model / (doc_id + ".tsv"), sep='\t', index_col='i', keep_default_na=False)['pred'].sort_index()
    sys_mentions = list(parse_mentions(df))
    key_clusters = mentions_to_clusters(key_mentions, doc_id='doc')
    sys_clusters = mentions_to_clusters(sys_mentions, doc_id='doc')

    key_mention_sys = get_mention_assignments(key_clusters, sys_clusters)
    sys_mention_key = get_mention_assignments(sys_clusters, key_clusters)

    p_num, p_den, per_entity = _lea_score(key_clusters, sys_clusters, key_mention_sys)
    entity_recall[doc_id] = per_entity

#%%

min_label_relevance = {
    'Goethe_Wahlverwandtschaften': 200,
    'Fischer_Gustav': 100,
    'Kürnberger_Amerika': 500,
}
small_groups = {
    'Goethe_Wahlverwandtschaften': [range(2, 21), range(21, 101)],
    'Kürnberger_Amerika': [range(2, 21), range(21, 101), range(101, 251)],
    'Fischer_Gustav': [range(2, 11), range(11, 51)],
}

fig, axs = plt.subplots(ncols=len(to_plot), figsize=(7, 4))

for ax, doc_id in zip(axs, to_plot):
    cycler = itertools.cycle(iter(plt.rcParams['axes.prop_cycle']))
    recall = entity_recall[doc_id]
    doc_entities = key_entities[doc_id]

    generic_entities = {x for x in doc_entities.keys() if 'generic' in doc_entities[x].specialcase_entity}
    singleton_entities = {x for x in doc_entities.keys() if recall[x][1] == 1}
    group_entities = {x for x in doc_entities.keys() if 'group' in doc_entities[x].specialcase_entity} - generic_entities

    core_entities = doc_entities.keys() - singleton_entities - group_entities

    cur = 0
    labels = []
    X = []
    height = []
    width = []
    color = []
    edgecolor = []

    large_entities = {x for x in core_entities if all(recall[x][1] not in r for r in small_groups[doc_id])}
    for entity_id in sorted(large_entities, key=lambda x: recall[x][1], reverse=True):
        resolution, relevance = recall[entity_id]
        X.append(cur)
        width.append(resolution)
        height.append(relevance)
        cur = cur + relevance
        labels.append(key_entities[doc_id][entity_id].fullname if relevance > min_label_relevance[doc_id] else '')
        color.append(next(cycler)['color'])

    for ran in sorted(small_groups[doc_id], key=lambda x: x.start, reverse=True):
        small_entities = {x for x in core_entities if recall[x][1] in ran}
        relevance = sum(recall[x][1] for x in small_entities)
        resolution = sum(recall[x][0] * recall[x][1] for x in small_entities) / relevance
        X.append(cur)
        width.append(resolution)
        height.append(relevance)
        cur = cur + relevance
        labels.append(f'entities w/ {ran.start}-{ran.stop-1} mentions')
        color.append('#aaaaaa')

    nongeneric_singletons = singleton_entities - generic_entities
    relevance = sum(recall[x][1] for x in nongeneric_singletons)
    resolution = sum(recall[x][0] * recall[x][1] for x in nongeneric_singletons) / relevance
    X.append(cur)
    width.append(resolution)
    height.append(relevance)
    cur = cur + relevance
    labels.append('non-generic singletons')
    color.append('#aaaaaa')

    relevance = sum(recall[x][1] for x in generic_entities)
    resolution = sum(recall[x][0] * recall[x][1] for x in generic_entities) / relevance
    X.append(cur)
    width.append(resolution)
    height.append(relevance)
    cur = cur + relevance
    labels.append('generic singletons')
    color.append('#f0a3a3')

    relevance = sum(recall[x][1] for x in group_entities - singleton_entities)
    resolution = sum(recall[x][0] * recall[x][1] for x in group_entities - singleton_entities) / relevance
    X.append(cur)
    width.append(resolution)
    height.append(relevance)
    cur = cur + relevance
    labels.append('group entities')
    color.append('#a0cfe0')

    ax.barh(X, height=height, width=width, color=color, align='edge', edgecolor='white', linewidth=0.6)

    ax.set_yticks(np.array(X) + np.array(height)/2, labels)
    ax.tick_params(axis='y', length=0, labelsize=5)

    ax.set_xlabel("LEA Resolution")
    ax.set_xlim((0, 1))
    ax.set_ylim((cur, 0))
    if ax == axs[0]:
        ax.set_ylabel("LEA Relevance = Entity Size")
    ax.set_title(doc_title[doc_id])
    ax.xaxis.set_major_formatter(FuncFormatter(percent_format))

    print(doc_id, sum(height), sum(x for _, x in recall.values()), sum(h*w for h, w in zip(height, width))/ sum(height), cluster_table.loc[('all', lea_model, doc_id), ('lea', 'recall')])

plt.tight_layout()
plt.savefig("/tmp/jcls_figures/llm_lea_plot.pdf")
plt.show()

#%%

## Plural Resolution


plural_mention_resolution = []
with tqdm(total=len(models) * sum(len(x) for x in document_mentions.values())) as pbar:
    for model in models:
        for doc_id, key_mentions in document_mentions.items():
            print(model, doc_id)
            inference = pandas.read_csv(ROOT_DIR / "llm_outputs" / "merged" / model / (doc_id + ".tsv"), sep='\t', index_col='i',
                         keep_default_na=False)['pred'].sort_index()
            sys_mentions = list(parse_mentions(inference))
            key_clusters = mentions_to_clusters(key_mentions, doc_id='doc')
            sys_clusters = mentions_to_clusters(sys_mentions, doc_id='doc')
            entity_mapping = compute_entity_mapping(sys_clusters, key_clusters)

            key_mention_sys = get_mention_assignments(key_clusters, sys_clusters)
            key_mentions_key = get_self_assignments(key_clusters)

            for key_mention, key_cluster_ids in key_mentions_key.items():
                sys_cluster_ids = set(key_mention_sys.get(key_mention, []))
                mapped_entities = set(entity_mapping.get(x, f'orphan_{x}') for x in sys_cluster_ids)

                plural_mention_resolution.append((
                  model, doc_id, key_mention.start, key_mention.end, len(key_cluster_ids), len(sys_cluster_ids),
                  len(mapped_entities - set(key_cluster_ids))
                ))
                pbar.update(1)


plural_mention_resolution = pandas.DataFrame(plural_mention_resolution, columns=['model', 'doc_id', 'begin', 'end', 'num_key', 'num_sys', 'num_errors'])

#%%

bins = pandas.IntervalIndex.from_breaks([1, 2, 3, 4, np.inf], closed='left')
labels = [f"{int(x.left)} ref." if x.right != np.inf else f"{int(x.left)}+ ref." for x in bins]

fig, axs = plt.subplots(nrows=2, figsize=(3, 3), sharex=True)
for model, df in plural_mention_resolution.groupby('model'):
    is_correct = df.groupby(pandas.cut(df['num_key'], bins)).apply(lambda x: (x['num_errors'] == 0).mean())
    axs[0].plot(is_correct.values, '-o', label=model_labels[model])

axs[0].legend(frameon=False)
axs[0].set_xticks(range(len(bins)), labels)
axs[0].set_ylim((0, 1))
axs[0].set_ylabel("Proportion correctly\n resolved in response")
axs[0].yaxis.set_major_formatter(FuncFormatter(percent_format))

p = axs[1].bar(x=range(len(bins)), height=df.groupby(pandas.cut(df['num_key'], bins)).size().values, color='#bbbbbb')
axs[1].set_xticks(range(len(bins)), labels)
axs[1].bar_label(p, label_type='edge')
axs[1].set_ylim((0, 70_000))
axs[1].set_ylabel("Num. key mentions")
axs[1].yaxis.set_major_formatter(FuncFormatter(log_number_fmt))

plt.tight_layout()
plt.savefig('/tmp/jcls_figures/llm_plurals.pdf')
plt.show()

