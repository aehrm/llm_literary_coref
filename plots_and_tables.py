#%%
import collections
import csv

import numpy as np
import warnings
import pandas
import json
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import matplotlib
from scipy.stats import linregress
from tqdm import tqdm

from llm_literary_coref.eval.evaluator import Evaluator
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
try:
    ROOT_DIR = Path(__file__).parent
except NameError:
    ROOT_DIR = Path('.').parent


#%%

# load files


annotations_dir = ROOT_DIR / "annotations"
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

print(entity_table.rename(entity_variants).to_string(na_rep='--', float_format=lambda x: f"{x*100:.2f}"))

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

#%%

# Difference to silver pre-annotation

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

# DROC Performance + IAA
model_output_dir = ROOT_DIR / "outputs"

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
metric_labels = {'conll': "CoNNL", 'lea': "LEA"}
models = ['google--gemini-2.5-flash-lite', 'google--gemini-2.5-flash', 'iaa']
model_labels = {'google--gemini-2.5-flash-lite': "gemini-2.5-flash-lite",
                'google--gemini-2.5-flash': "gemini-2.5-flash",
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
    ax.axvline(100*maverick_baseline_scores.loc[metric], ls='--', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3])
    ax.yaxis.set_inverted(True)
    ax.set_xlabel(metric_labels[metric])
    ax.set_yticks(ticks, [model_labels[m] for m in models])

plt.tight_layout(h_pad=3)
plt.savefig('/tmp/jcls_figures/droc_performance.pdf')
plt.show()

