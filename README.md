# Coreference Resolution for Full German Novels using Large Language Models

This is the accompagnying repository to the paper [Coreference Resolution for Full German Novels using Large Language Models](TODO) (2026, JCLS) and contains

1. the GerFuN (German Full Novel) coreference dataset, and
2. a pipeline for detecting character mentions and resolving coreference chains in literary texts using LLMs (via OpenRouter).

## Repository Structure
- `gerfun_corpus/`: German Literary dataset in TSV format (and original INCEpTION XMI source files).
- `llm_outputs/`: Original LLM inferences reported in the paper.
- `llm_literary_coref/`: Python library containing logic for prompts, evaluation metrics, and API interaction.
- `mention_detection.py`: Detects spans of potential character mentions using a BERT-based sequence tagger.
- `llm_coref_annotation.py`: Performs entity resolution on individual text segments using an LLM.
- `llm_coref_merger.py`: Merges entities from different segments into a consistent global entity list for a full novel, again using an LLM.
- `eval_llm_inference.py`: Evaluates prediction quality against gold standards.
- `plots_and_tables.py`: Code to reproduce the plots and tables of the paper.

## Getting Started

This project uses `uv` for dependency management.
Install [uv](https://github.com/astral-sh/uv) and export your OpenRouter API key:

```bash
export OPENROUTER_API_KEY='your_api_key_here'
```


## Inference

To process new text, you must provide a novel in TSV format. Each row represents one token. The TSV file needs the following columns: `i` (numeric token index), `token` (token as string), `is_section_start` (integer, 1 if token starts a new section, 0 otherwise). All other columns (such as `gold`) are copied.

### 1. Mention Detection and Section Splitting

The first step uses a BERT-based model to find character mention spans. This divides your long text into smaller segments along section starts.
```bash
uv run python mention_detection.py \
    --input_files my_novel.tsv \
    --model_id "aehrm/moderngbert-fun-mention-detection" \
    --output_dir llm_outputs/mention_detection
# generates files 
# llm_outputs/mention_detection/my_novel_segment_0000.tsv
# llm_outputs/mention_detection/my_novel_segment_0001.tsv
# ...
```

### 2. LLM Annotation (Per Segment)
```bash
uv run python llm_coref_annotation.py \
    --input_files llm_outputs/mention_detection/my_novel_segment_*.tsv \
    --output_dir llm_outputs/predicted_sections/ \
    --model "google/gemini-2.5-flash-lite"
# generates files 
# llm_outputs/predicted_sections/my_novel_segment_0000.tsv
# llm_outputs/predicted_sections/my_novel_segment_0001.tsv
# ...
```

### 3. Global Entity Merging
```bash
uv run python llm_coref_merger.py \
    --novel_files my_novel.tsv \
    --input_files llm_outputs/predicted_sections/my_novel_segment_*.tsv \
    --output_dir llm_outputs/merged/ \
    --model "google/gemini-2.5-flash-lite"
# generates file
# llm_outputs/merged/my_novel.tsv
```

## Reproduction

### 1. Mention Detection
Run the BERT-based mention detector on your source TSVs. This creates segments with IDs for potential mentions.
```bash
uv run python mention_detection.py \
    --input_files gerfun_corpus/sources/*.tsv \
    --model_id "aehrm/moderngbert-fun-mention-detection" \
    --output_dir llm_outputs/mention_detection
# generates files 
# llm_outputs/mention_detection/Goethe_Wahlverwandtschaften_segment_0000.tsv
# llm_outputs/mention_detection/Goethe_Wahlverwandtschaften_segment_0001.tsv
# ...
```

### 2. LLM Annotation (Per Segment)
Annotate segments using an LLM
```bash
uv run python llm_coref_annotation.py \
    --input_files llm_outputs/mention_detection/* \
    --output_dir llm_outputs/predicted_sections/google--gemini-2.5-flash-lite \
    --model "google/gemini-2.5-flash-lite"
# generates files 
# llm_outputs/predicted_sections/google--gemini-2.5-flash/Goethe_Wahlverwandtschaften_segment_0000.tsv
# llm_outputs/predicted_sections/google--gemini-2.5-flash/Goethe_Wahlverwandtschaften_segment_0001.tsv
# ...
```

### 3. Global Entity Merging
Consolidate the mentions so that characters are linked across segments
```bash
uv run python llm_coref_merger.py \
    --novel_files gerfun_corpus/annotated_tsv/*.tsv \
    --segment_files llm_outputs/predicted_sections/google--gemini-2.5-flash-lite/*.tsv \
    --output_dir llm_outputs/merged/google--gemini-2.5-flash-lite \
    --model "google/gemini-2.5-flash-lite"
# generates file
# llm_outputs/merged/google--gemini-2.5-flash-lite/Goethe_Wahlverwandtschaften.tsv
# llm_outputs/merged/google--gemini-2.5-flash-lite/Fischer_Gustav.tsv
# ...
```

NOTE: This copies all columns (including the gold values) from the novel TSV files into the output files.

### 4. Evaluation
Compare your results against the gold standard:
```bash
uv run python eval_llm_inference.py \
    --pred_files llm_outputs/merged/google--gemini-2.5-flash-lite/*.tsv \
    --write_json_report llm_outputs/merged/google--gemini-2.5-flash-lite/evaluation_report.json
# generates file
# llm_outputs/merged/google--gemini-2.5-flash-lite/evaluation_report.json
```

NOTE: Evaluations compare the column `pred` with the column `gold` (copied in step 3).
