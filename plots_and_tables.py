#%%
import collections
import csv

import numpy as np
import pandas
import json
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import matplotlib
from scipy.stats import linregress
from tqdm import tqdm

from llm_literary_coref.mention import parse_mentions

#%%

from cycler import cycler
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=[
(0.2823529411764706, 0.47058823529411764, 0.8156862745098039), (0.9333333333333333, 0.5215686274509804, 0.2901960784313726), (0.41568627450980394, 0.8, 0.39215686274509803), (0.8392156862745098, 0.37254901960784315, 0.37254901960784315), (0.5843137254901961, 0.4235294117647059, 0.7058823529411765), (0.5490196078431373, 0.3803921568627451, 0.23529411764705882), (0.8627450980392157, 0.49411764705882355, 0.7529411764705882), (0.4745098039215686, 0.4745098039215686, 0.4745098039215686), (0.8352941176470589, 0.7333333333333333, 0.403921568627451), (0.5098039215686274, 0.7764705882352941, 0.8862745098039215)

])

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

#%%

# load files

try:
    document_files = (Path(__file__).parent / "annotations" / "annotated_tsv").glob('*.tsv')
except NameError:
    document_files = (Path('.') / "annotations" / "annotated_tsv").glob('*.tsv')

documents = {}
document_mentions = {}
document_entities = {}
for doc in document_files:
    gold = pandas.read_csv(doc, sep='\t', index_col='i', quoting=csv.QUOTE_NONE)['gold'].sort_index()
    mentions = list(parse_mentions(gold))

    df = pandas.read_csv(doc.parent.parent / "sources" / doc.name, sep='\t', index_col='i', quoting=csv.QUOTE_NONE).sort_index()
    df['gold'] = gold
    documents[doc.stem] = df
    document_mentions[doc.stem] = mentions

    document_entities[doc.stem] = collections.defaultdict(list)
    for mention in mentions:
        for ref in mention.references:
            document_entities[doc.stem][ref.entity.id].append((mention, ref))
            
#%%

ordering = list(documents.keys())
doc_title = {
'Heimburg_Trudchen': 'Trudchens Heirat',
'Fischer_Gustav': 'Gustavs Verirrungen',
'Goethe_Wahlverwandtschaften': 'Wahlverwandtschaften',
'Kürnberger_Amerika': 'Amerika-Müde',
}


#%%


def get_mentions_by_tag(document_id, tag):
    df = documents[document_id]
    out = []
    for mention in document_mentions[document_id]:
        if any(df.loc[mention.token_idx, 'tag'] == tag):
            out.append(mention)
    return out

#%%

# basic statistics

statistics = []
for doc, df in sorted(documents.items(), key=lambda x: len(x[1])):
    statistics.append([doc, 'Num. tokens', len(df)])
    statistics.append([doc, 'Num. sentences', df['is_sent_start'].sum()])
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
entity_statistics.append(['Num. nonfact entities', sum(1 for entities in document_entities.values() for references in entities.values() if 'nonfact' in references[0][1].entity.specialcase_entity)])

df = pandas.DataFrame(entity_statistics, columns=['label', 'count']).set_index('label')
df['average'] = df['count'] / len(documents)
df['proportion'] = df['count'] / df['count'].iloc[0]

print(df[['count', 'average', 'proportion']].to_string(na_rep=''))

#%%

# calculate entity sizes

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
    ax.scatter(x, y, marker='o', facecolors='none', edgecolors=color, linewidth=plt.rcParams["lines.linewidth"])

    # Calculate trend line using log-transformed values
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

    # Plot trend line
    x_line = np.array([g['rank'].min(), g['rank'].max()])
    y_line = np.exp(intercept + slope * np.log(x_line))
    ax.plot(x_line, y_line, '--', color=color, alpha=0.4, label=f'${np.exp(intercept)/1000:.2f}\\cdot 10^3 \\cdot x^{{{slope:.2g}}}$')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('Entity rank')
ax.set_ylabel('Number of references')
ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
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
    ax.scatter(x, y, marker='o', facecolors='none', edgecolors=color, label=doc_title[doc], linewidth=plt.rcParams["lines.linewidth"])

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

# Spread

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

fig, ax = plt.subplots(ncols=2, nrows=len(ordering)//2, sharex=True, sharey=True, figsize=(5.3, 2.4*len(ordering)//2))

axs = ax.flatten()

cycler = iter(plt.rcParams['axes.prop_cycle'])
for ax, (doc, doc_df) in zip(axs, sorted(spreads.groupby('doc'), key=lambda x: ordering.index(x[0]))):
    sel = (doc_df['num_references'] > 1)
    g = doc_df.loc[sel]

    x = g['num_references']#/sum(doc_df['num_references'])
    y = g['spread']#/len(documents[doc])
    color = next(cycler)['color']
    ax.scatter(x, y, marker='o', facecolors='none', edgecolors=color)
    ax.axhline(len(documents[doc]), ls='--', color=color, alpha=0.5, linewidth=1.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(doc_title[doc])
    ax.set_ylabel('Spread in Tokens')
    ax.set_xlabel('Number of References')

plt.tight_layout()
plt.savefig('/tmp/jcls_figures/entity_spread.pdf')
plt.show()

#%%

# Mention Distance

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

x = [1,2,5] + list(np.logspace(np.log10(10), np.log10(30000), 20))

cycler = iter(plt.rcParams['axes.prop_cycle'])
fig, ax = plt.subplots(figsize=(4, 2))
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
ax.set_ylabel('Survival probability')
# ax.legend(frameon=False)
def percent_format(x, pos=None):
    if x > 0.01:
        return f'{x * 100:.0f}%'
    else:
        return f'{x * 100:.3g}%'

def tokencount_format(x, pos=None):
    if x < 1000:
        return f'{x:.0f}'
    else:
        return f'{x//1000:.0f}k'

ax.xaxis.set_major_formatter(FuncFormatter(tokencount_format))
ax.yaxis.set_major_formatter(FuncFormatter(percent_format))

ax.legend(frameon=False)
plt.tight_layout()
plt.savefig('/tmp/jcls_figures/mention_distance.pdf')
plt.show()


#%%

# IAA
with open(Path(__file__).parent / 'annotations' / 'evaluation_reports' / 'iaa_report.json') as f:
    iaa_report = json.load(f)

#%%

cluster_metrics = list(iaa_report['clusters_all'].keys())

cluster_variants = {
    "all": "full",
    "replaceplural": "plurals replaced",
    "nogeneric": "w/o generics",
    "nosingletons": "w/o singletons",
}


cluster_table = pandas.DataFrame(index=['all', 'replaceplural', 'nogeneric', 'nosingletons'],
                                 columns=['mentions', 'muc', 'bcub', 'ceafe', 'conll', 'ceafm', 'lea'])

for variant in cluster_table.index:
    report = iaa_report[f'clusters_{variant}']

    for metric, scores in report.items():
        cluster_table.loc[variant, metric] = scores['aggregated']['f1']

cluster_table['conll'] = cluster_table[['muc', 'bcub', 'ceafe']].mean(axis=1)

print(cluster_table.rename(cluster_variants).to_string(float_format=lambda x: f"{x*100:.2f}"))

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

print(entity_table.rename(cluster_variants).to_string(na_rep='--', float_format=lambda x: f"{x*100:.2f}"))

#%%

mention_metrics = ['part', 'figurative']
mention_table = pandas.DataFrame(index=mention_metrics, columns=['f1', 'count'])
for metric in mention_table.index:
    scores = iaa_report['mention_attributes'][metric]['aggregated']
    counts = list(sorted([scores['support_key'], scores['support_response']]))
    count_str = '+'.join(map(str, counts))

    if sum(counts) > 0:
        mention_table.loc[metric, 'count'] = count_str
        mention_table.loc[metric, 'f1'] = scores['f1']

print(mention_table.to_string(na_rep='--', float_format=lambda x: f"{x*100:.2f}"))

