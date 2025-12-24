from pathlib import Path

import more_itertools
import pandas as pd
import torch
import re
import os
import argparse

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from datasets import Dataset
from tqdm import tqdm
from functools import partial


def split_into_segments(df, max_segment_length=None):
    section_id = df['is_section_start'].cumsum()
    for _, group in df.groupby(section_id):
        idx = group.index
        if max_segment_length is None or len(idx) <= max_segment_length:
            yield group
        else:
            for segment_idx in more_itertools.chunked(idx, max_segment_length):
                yield group.loc[segment_idx]



def read_input_tsv(file_path, max_segment_length=None):
    try:
        df = pd.read_csv(file_path, sep='\t', keep_default_na=False, index_col='i')
        if 'is_section_start' not in df.columns:
            raise ValueError("Header likely missing")
    except ValueError:
        print("Header likely missing. Trying to read without header, assuming columns 'token' and 'is_section_start'.")
        df = pd.read_csv(file_path, sep='\t', header=None, keep_default_na=False)
        df.columns.names = ['token', 'is_section_start'][:len(df.columns)]

    if 'token' not in df.columns:
        raise ValueError("No 'token' column found in input file.")

    if 'is_section_start' in df.columns:
        df['is_section_start'] = df['is_section_start'].astype(str).str.lower().isin(['true', '1', 't', 'yes'])
    else:
        df['is_section_start'] = False


    df['mention'] = None
    segments = [segment.copy().reset_index() for segment in split_into_segments(df, max_segment_length=max_segment_length)]
    return segments


def tokenize(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True,
        max_length=1024,
        padding=True,
        return_overflowing_tokens=True
    )

    sample_map = tokenized_inputs["overflow_to_sample_mapping"]
    sample_id = []
    sample_words = []

    for i, sample_idx in enumerate(sample_map):
        sample_id.append(examples['section_id'][sample_idx])
        sample_words.append(tokenized_inputs.word_ids(i))


    tokenized_inputs["section_id"] = sample_id
    tokenized_inputs.pop("overflow_to_sample_mapping")
    tokenized_inputs["word_ids"] = sample_words
    return tokenized_inputs



def extract_spans_from_predictions(word_ids, predictions):
    current_start = None

    processed_words = set()
    cleaned_labels = []

    # Map tokens to Word Labels ("First" strategy)
    for w_id, label in zip(word_ids, predictions):
        if w_id is None:
            continue

        if w_id not in processed_words:
            cleaned_labels.append((w_id, label))
            processed_words.add(w_id)

    # 2. Extract spans using standard BIO logic
    for word_idx, label in cleaned_labels:
        if label == 'B':
            if current_start is not None:
                # Close previous span
                yield (current_start, word_idx)
            current_start = word_idx
        elif label == 'I':
            if current_start is None:
                # recovered from error or partial context split: treat as B
                current_start = word_idx
        else: # O
            if current_start is not None:
                yield (current_start, word_idx)
                current_start = None

    if current_start is not None and cleaned_labels:
        last_word_idx = cleaned_labels[-1][0] + 1
        yield (current_start, last_word_idx)


def predict(examples, model, device):
    section_ids = examples.pop('section_id')
    word_ids = examples.pop('word_ids')

    with torch.no_grad():
        out = model(**{k: torch.tensor(v).to(device) for k, v in examples.items()})
        predictions = []

    for i in range(out.logits.shape[0]):
        pred = out.logits[i].argmax(dim=-1).cpu()
        predictions.append([model.config.id2label[j] for j in pred.tolist()])


    return {'predictions': predictions, 'section_id': section_ids, 'word_ids': word_ids}


def process_sections(sections, basename, model_id, output_dir, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id).to(device)
    model.eval()

    def gen_dataset():
        for sec_id, section_df in enumerate(sections):
            tokens = section_df['token'].astype(str).tolist()
            yield {'tokens': tokens, 'section_id': sec_id}

    ds = Dataset.from_generator(gen_dataset)
    tokenized_ds = ds.map(tokenize, fn_kwargs=dict(tokenizer=tokenizer), batched=True, batch_size=1000, remove_columns=['tokens', 'section_id'])
    predicted_ds = tokenized_ds.map(predict, fn_kwargs=dict(model=model, device=device), batched=True, batch_size=4)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    global_mention_counter = 0
    for sample in predicted_ds:
        predicted_spans = extract_spans_from_predictions(sample['word_ids'], sample['predictions'])
        section_df = sections[sample['section_id']]

        for start, end in predicted_spans:
            global_mention_counter += 1
            section_df.loc[start:end - 1, 'mention'] = global_mention_counter


    for sec_id, section_df in enumerate(sections):
        output_df = section_df[['i', 'token', 'mention']].copy()
        if 'is_section_start' in section_df.columns:
            output_df['is_section_start'] = section_df['is_section_start'].copy().astype(int)
        if 'gold' in section_df.columns:
            output_df['gold'] = section_df['gold'].copy()

        output_df.set_index('i', inplace=True)

        # Write to TSV
        out_path = output_dir / f"{basename}_segment_{sec_id:04d}.tsv"
        print(f"Writing {out_path}...")
        output_df.to_csv(out_path, sep='\t')

    print(f"Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mention detection on TSV files.")
    parser.add_argument("--input_file", type=Path, required=True, help="Path to input TSV file.")
    parser.add_argument("--model_id", type=str, required=True, help="Path to HuggingFace model or Hub ID.")
    parser.add_argument("--output_dir", type=Path, default="outputs/mention_detection", help="Directory to save output TSVs.")
    parser.add_argument("--max_segment_length", type=int, default=8192, help="Maximum segment length. If section is longer than this quantity, section will be split into multiple segments of this length.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu).")

    args = parser.parse_args()

    # Read and split data
    print("Reading input file...")
    sections = read_input_tsv(args.input_file, max_segment_length=args.max_segment_length)

    # Process
    basename = args.input_file.stem
    process_sections(sections, basename, args.model_id, args.output_dir, args.device)
