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

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', markersize=4)

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

# make some statistics

def get_mentions_by_tag(document_id, tag):
    df = documents[document_id]
    out = []
    for mention in document_mentions[document_id]:
        if any(df.loc[mention.token_idx, 'tag'] == tag):
            out.append(mention)
    return out


statistics = []
statistics.append([0, 'Num. documents', len(documents)])
statistics.append([1, 'Num. tokens', sum(len(df) for df in documents.values())])
statistics.append([1, 'Num. sentences', sum(df['is_sent_start'].sum() for df in documents.values())])
statistics.append([1, 'Num. entities', sum(len(entities) for entities in document_entities.values())])

statistics.append([1, 'Num. mentions', sum(len(mentions) for mentions in document_mentions.values())])
statistics.append([1, 'Num. plural mentions', sum(1 for mentions in document_mentions.values() for mention in mentions if len(mention.references) > 1)])
statistics.append([1, 'Num. proper noun mentions', sum(1 for doc in documents.keys() for mention in get_mentions_by_tag(doc, 'NE'))])
statistics.append([1, 'Num. nominal noun mentions', sum(1 for doc in documents.keys() for mention in get_mentions_by_tag(doc, 'NN'))])
statistics.append([1, 'Num. mentions with figurative reference', sum(1 for mentions in document_mentions.values() for mention in mentions if any('figurative' in ref.specialcase_reference for ref in mention.references))])
statistics.append([1, 'Num. mentions with part reference', sum(1 for mentions in document_mentions.values() for mention in mentions if any('part' in ref.specialcase_reference for ref in mention.references))])
# statistics.append([0, 'Mentions per Token', sum(len(mentions) for mentions in document_mentions.values()) / sum(len(df) for df in documents.values())])

statistics.append([1, 'Num. non-singleton entities', sum(1 for entities in document_entities.values() for references in entities.values() if len(references) > 1)])
statistics.append([1, 'Num. singleton entities', sum(1 for entities in document_entities.values() for references in entities.values() if len(references) == 1)])
statistics.append([1, 'Num. generic entities', sum(1 for entities in document_entities.values() for references in entities.values() if 'generic' in references[0][1].entity.specialcase_entity)])
statistics.append([1, 'Num. group entities', sum(1 for entities in document_entities.values() for references in entities.values() if 'group' in references[0][1].entity.specialcase_entity)])
statistics.append([1, 'Num. nonfact entities', sum(1 for entities in document_entities.values() for references in entities.values() if 'nonfact' in references[0][1].entity.specialcase_entity)])

props = {
    'Num. mentions': ['Num. plural mentions', 'Num. mentions with figurative reference', 'Num. mentions with part reference',
                      'Num. proper noun mentions', 'Num. nominal noun mentions'],
    'Num. entities': ['Num. non-singleton entities', 'Num. singleton entities', 'Num. generic entities',
                      'Num. group entities', 'Num. nonfact entities']
}

df = pandas.DataFrame(statistics, columns=['to_average', 'label', 'value'])
df['average'] = np.nan
df.loc[df.to_average == 1, 'average'] = df.loc[df.to_average == 1, 'value'] / len(documents)

df['proportion'] = np.nan
for normalization_row, rows in props.items():
    sel = df.label.isin([normalization_row] + rows)
    df.loc[sel, 'proportion'] = df.loc[sel, 'value'] / df.loc[df.label == normalization_row, 'value'].item()

print(df[['label', 'value', 'average', 'proportion']].to_string(index=False, na_rep=''))

#%%

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

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3.5))

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
    ax.scatter(x, y, marker='o', facecolors='none', edgecolors=color)

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
    ax.scatter(x, y, marker='o', facecolors='none', edgecolors=color, label=doc_title[doc])

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

#%%

fig, ax = plt.subplots(ncols=2, nrows=len(ordering)//2, sharex=True, sharey=True, figsize=(6, 3*len(ordering)//2))

axs = ax.flatten()

cycler = iter(plt.rcParams['axes.prop_cycle'])
for ax, (doc, doc_df) in zip(axs, sorted(spreads.groupby('doc'), key=lambda x: ordering.index(x[0]))):
    sel = ~spreads['group'] & (spreads['num_references'] > 10)
    g = doc_df.loc[sel]

    x = g['num_references']#/sum(doc_df['num_references'])
    y = g['spread']#/len(documents[doc])
    color = next(cycler)['color']
    ax.scatter(x, y, marker='o', facecolors='none', edgecolors=color)
    ax.axhline(len(documents[doc]), ls='--', color=color, alpha=0.3)
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
            pbar.update(len(distances))
            doc_df = documents[doc]
            # chapter_id = doc_df['is_section_start'].cumsum()
            mention_positions = list(sorted(mention.token_idx[0] for mention, _ in mentions))
            # mention_chapter = [chapter_id.loc[x] for x in mention_positions]
            mention_type = [doc_df.loc[x]['tag'] for x in mention_positions]

            distances = pandas.DataFrame({'position': mention_positions, 'type': mention_type})
            # distances['token'] = doc_df.loc[distances.position, 'text'].reset_index(drop=True)

            if len(distances) == 1:
                continue

            distances['distance'] = distances['position'].diff()
            distances['entity'] = entity_id
            distances['doc'] = doc

            mention_distances.append(distances.iloc[1:])

mention_distances = pandas.concat(mention_distances)

#%%
d = mention_distances[~mention_distances['type'].str.match('NN|NE')]['distance']
d = d[d > 0]  # ignore overlapping mentions

print(d.describe(percentiles=[0.5, 0.9, .95, .99, .999]).to_string())