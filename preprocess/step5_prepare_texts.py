"""Step 5: build text annotations, split files, and prompt templates.

Reads the official dataset CSVs and writes the HumanML3D-style layout
expected by the training code under the dataset root:

    <out_root>/
    ├── TEXT_new_tackle/<clip>.txt     # "caption#word/POS ...#0.0#0.0"
    ├── <split>.txt                    # one clip name per line
    ├── template_pretrain.json         # stage-2 pretraining prompts
    └── template_instructions.json     # stage-3 instruction prompts

Only clips whose final features exist (step3 output) are kept, so the
text and split files always match the motion data.

Caption cleaning follows the original preprocessing: lowercase, keep
alphabetic characters, whitespace and periods (dropping punctuation such
as apostrophes), then attach spaCy POS tags for every token.

Usage (How2Sign, ASL):
    python -m preprocess.step5_prepare_texts --dataset how2sign \
        --splits train=/path/how2sign_realigned_train.csv \
                 val=/path/how2sign_realigned_val.csv \
                 test=/path/how2sign_realigned_test.csv \
        --features_dir /path/to/all_73j_new_joint_vecs_onlylocal_cat_expression \
        --out_root /path/to/data/how2sign

Usage (PHOENIX-2014T, DGS):
    python -m preprocess.step5_prepare_texts --dataset phoenix \
        --splits train=/path/PHOENIX-2014-T.train.corpus.csv \
                 val=/path/PHOENIX-2014-T.dev.corpus.csv \
                 test=/path/PHOENIX-2014-T.test.corpus.csv \
        --features_dir /path/to/all_73j_new_joint_vecs_onlylocal_cat_expression \
        --out_root /path/to/data/phoenix2014t
"""

import argparse
import csv
import os
import shutil
import sys

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

DATASET_DEFAULTS = {
    'how2sign': {'language': 'en', 'spacy_model': 'en_core_web_sm'},
    'phoenix': {'language': 'de', 'spacy_model': 'de_core_news_sm'},
}


def clean_caption(text):
    """Lowercase and keep only letters, whitespace and periods."""
    return ''.join(c for c in text.lower() if c.isalpha() or c.isspace() or c == '.')


def pos_tags(text, nlp):
    """Return 'word/POS word/POS ...' for the caption."""
    doc = nlp(text)
    return ' '.join(f'{token.text}/{token.pos_}' for token in doc)


def read_annotations(csv_path, dataset):
    """Yield (clip_name, sentence) from an official dataset CSV."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        if dataset == 'how2sign':
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                rows.append((row['SENTENCE_NAME'].strip(), row['SENTENCE'].strip()))
        elif dataset == 'phoenix':
            reader = csv.DictReader(f, delimiter='|')
            for row in reader:
                rows.append((row['name'].strip(), row['translation'].strip()))
        else:
            raise ValueError(f'Unknown dataset {dataset}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument('--splits', nargs='+', required=True,
                        help='Split definitions as name=csv_path, e.g. '
                             'train=/path/to/train.csv (repeat for val/test)')
    parser.add_argument('--features_dir', required=True,
                        help='Step3 output directory; only clips with features are kept')
    parser.add_argument('--out_root', required=True,
                        help='Dataset root the training config points at')
    parser.add_argument('--spacy_model', default=None,
                        help='spaCy model for POS tags (default: per-dataset choice)')
    args = parser.parse_args()

    import spacy

    defaults = DATASET_DEFAULTS[args.dataset]
    spacy_model = args.spacy_model or defaults['spacy_model']
    try:
        nlp = spacy.load(spacy_model, disable=['parser', 'ner'])
    except OSError:
        print(f'spaCy model {spacy_model!r} is required. Install it with:\n'
              f'    python -m spacy download {spacy_model}', file=sys.stderr)
        sys.exit(1)

    available = {f[:-4] for f in os.listdir(args.features_dir) if f.endswith('.npy')}
    text_dir = os.path.join(args.out_root, 'TEXT_new_tackle')
    os.makedirs(text_dir, exist_ok=True)
    all_kept = {}

    for split_def in args.splits:
        split_name, _, csv_path = split_def.partition('=')
        if not split_name or not csv_path:
            print(f'Invalid --splits entry: {split_def!r} (expected name=path)',
                  file=sys.stderr)
            sys.exit(1)

        rows = read_annotations(csv_path, args.dataset)
        kept = []
        missing_text = 0
        for clip_name, sentence in rows:
            if clip_name not in available:
                continue
            caption = clean_caption(sentence)
            if not caption.strip(' .'):
                missing_text += 1
                continue
            tags = pos_tags(caption, nlp)
            with open(os.path.join(text_dir, f'{clip_name}.txt'), 'w', encoding='utf-8') as f:
                f.write(f'{caption}#{tags}#0.0#0.0')
            kept.append(clip_name)

        split_file = os.path.join(args.out_root, f'{split_name}.txt')
        with open(split_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(kept)) + '\n')
        print(f'Split {split_name!r}: {len(kept)} clips written to {split_file} '
              f'({missing_text} clips skipped for empty captions)')
        all_kept.setdefault(split_name, kept)

    # Stage 1 (VQ tokenizer) trains on every clip: write the union of all
    # requested splits as train_all.txt unless it was given explicitly.
    if 'train_all' not in all_kept:
        union = sorted({name for kept in all_kept.values() for name in kept})
        with open(os.path.join(args.out_root, 'train_all.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(union) + '\n')
        print(f'Split train_all: {len(union)} clips (union of all splits)')

    for template in ('template_pretrain.json', 'template_instructions.json'):
        src = os.path.join(TEMPLATES_DIR, template)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out_root, template))

    print(f'Done -> {args.out_root}')


if __name__ == '__main__':
    main()
