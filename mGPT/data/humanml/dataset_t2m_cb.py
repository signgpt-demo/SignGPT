import rich
import random
import pickle
import os
import numpy as np
import codecs as cs
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json
import spacy
from scipy.interpolate import interp1d

# Optional data augmentation switches.
AUG_temporal_infer = False
Text_aug = False

def load_spacy_model(language):
    """Loads a SpaCy model based on the language string."""
    model_map = {
        'en': 'en_core_web_sm',
        'de': 'de_core_news_sm'
    }
    if language not in model_map:
        raise ValueError(f"Unsupported language '{language}'. Supported languages are: {list(model_map.keys())}")

    model_name = model_map[language]
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Spacy model '{model_name}' not found. Please install it by running:")
        print(f"python -m spacy download {model_name}")
        raise

class Text2MotionDatasetCB(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        code_path='VQVAE',
        task_path=None,
        std_text=False,
        language: str = 'de',
        **kwargs,
    ):
        self.tiny = tiny
        self.unit_length = unit_length

        # Data mean and std
        self.mean = mean
        self.std = std

        # Data path
        split = 'train'
        split_file = pjoin(data_root, split + '.txt')
        motion_dir = pjoin(data_root, code_path)
        text_dir = pjoin(data_root, 'TEXT_new_tackle')

        if task_path:
            instructions = task_path
        elif stage == 'lm_pretrain':
            instructions = pjoin(data_root, 'template_pretrain.json')
        elif stage in ['lm_instruct', "lm_rl"]:
            instructions = pjoin(data_root, 'template_instructions.json')
        else:
            raise NotImplementedError(f"stage {stage} not implemented")

        # Data id list
        self.id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        # Debug mode
        if tiny or debug:
            enumerator = enumerate(self.id_list)
            maxdata = 100
            subset = '_tiny'
        else:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading HumanML3D {split}",
                ))
            maxdata = 1e10
            subset = ''

        new_name_list = []
        data_dict = {}

        # Fast loading
        for i, name in enumerator:
            if len(new_name_list) > maxdata:
                break
            try:
                # Load motion tokens
                m_token_list = np.load(pjoin(motion_dir, f'{name}.npy'))
                # Read text
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            if len(caption.split()) > 50:
                                continue

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                        except:
                            pass

                if flag:
                    data_dict[name] = {
                        'm_token_list': m_token_list,
                        'text': text_data
                    }
                    new_name_list.append(name)
            except:
                pass

        if tmpFile:
            os.makedirs(pjoin(data_root, 'tmp'), exist_ok=True)
            with open(
                    pjoin(data_root,
                          f'tmp/{split}{subset}_tokens_data.pkl'),
                    'wb') as file:
                pickle.dump(data_dict, file)
            with open(
                    pjoin(data_root,
                          f'tmp/{split}{subset}_tokens_index.pkl'),
                    'wb') as file:
                pickle.dump(new_name_list, file)

        self.data_dict = data_dict
        self.name_list = new_name_list
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])


        self.std_text = std_text
        self.language = language
        if self.std_text:
            self.nlp = load_spacy_model(self.language)
            # Preserve directional words during lemmatization.
            if self.language == 'en':
                self.directional_words = ['left', 'right']
            elif self.language == 'de':
                self.directional_words = ['links', 'rechts']
            else:
                self.directional_words = []

    def __len__(self):
        return len(self.name_list) * len(self.tasks)

    def __getitem__(self, item):
        data_idx = item % len(self.name_list)
        task_idx = item // len(self.name_list)

        data = self.data_dict[self.name_list[data_idx]]
        m_token_list, text_list = data['m_token_list'], data['text']

        m_tokens = random.choice(m_token_list).copy()
        text_data = random.choice(text_list)
        caption = text_data['caption']

        # Normalize text before optional augmentation.
        word_list = caption.split()
        if self.std_text:
            doc = self.nlp(caption)
            processed_words = []
            for token in doc:
                word = token.text.lower()
                if not word.isalpha():
                    continue

                if (token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV', 'AUX'}) and (word not in self.directional_words):
                    processed_words.append(token.lemma_.lower())
                else:
                    processed_words.append(word)
            word_list = processed_words

        # Randomly drop a small fraction of words when text augmentation is enabled.
        if np.random.rand() < 0.50 and Text_aug:
            if len(word_list) > 0:
                drop_pct = np.random.random() * 10
                num_to_drop = int(len(word_list) * drop_pct / 100)

                if num_to_drop > 0:
                    indices_to_drop = set(random.sample(range(len(word_list)), num_to_drop))
                    word_list = [word for i, word in enumerate(word_list) if i not in indices_to_drop]

        caption = ' '.join(word_list)

        # Apply token-level temporal perturbation when enabled.
        if np.random.rand() < 0.20 and AUG_temporal_infer:
            current_m_tokens_len = m_tokens.shape[0]
            codebook_num = 3
            num_units = current_m_tokens_len // codebook_num

            if np.random.choice([True, False]):
                duplicate_pct = np.random.random() * 10
                num_units_to_duplicate = int(num_units * duplicate_pct / 100)

                if num_units_to_duplicate > 0:
                    unit_indices_to_duplicate = np.random.choice(num_units, num_units_to_duplicate, replace=False)
                    unit_indices_to_duplicate = sorted(unit_indices_to_duplicate, reverse=True)

                    for unit_idx in unit_indices_to_duplicate:
                        unit_start = unit_idx * codebook_num
                        unit_end = unit_start + codebook_num
                        unit_to_duplicate = m_tokens[unit_start:unit_end]
                        m_tokens = np.concatenate([m_tokens[:unit_end], unit_to_duplicate, m_tokens[unit_end:]], axis=0)
            else:

                drop_pct = np.random.random() * 10
                num_units_to_drop = int(num_units * drop_pct / 100)

                if num_units_to_drop > 0:
                    unit_indices_to_drop = set(np.random.choice(num_units, num_units_to_drop, replace=False))

                    kept_units = []
                    for unit_idx in range(num_units):
                        if unit_idx not in unit_indices_to_drop:
                            unit_start = unit_idx * codebook_num
                            unit_end = unit_start + codebook_num
                            kept_units.append(m_tokens[unit_start:unit_end])

                    if len(kept_units) > 0:
                        m_tokens = np.concatenate(kept_units, axis=0)
                    remainder_start = num_units * codebook_num
                    if remainder_start < current_m_tokens_len:
                        remainder = m_tokens[remainder_start:]
                        m_tokens = np.concatenate([m_tokens, remainder], axis=0)


        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        coin = np.random.choice([False, False, True])

        if coin and len(m_tokens) > 3:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-3]
            else:
                m_tokens = m_tokens[3:]

        m_tokens_len = m_tokens.shape[0]

        tasks = self.tasks[task_idx]

        return caption, m_tokens, m_tokens_len, None, None, None, None, all_captions, tasks
