import random
import numpy as np
from .dataset_t2m import Text2MotionDataset
import torch
from scipy.interpolate import interp1d

class Text2MotionDatasetEval(Text2MotionDataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        w_vectorizer,
        max_motion_length=196,
        min_motion_length=24,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        std_text=False,
        language: str = 'de',
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        self.w_vectorizer = w_vectorizer


    def __getitem__(self, item):
        # Get text data
        idx = self.pointer + item
        sample_name = self.name_list[idx]

        data = self.data_dict[sample_name]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        motion_token = None

        gloss_embedding = None
        gloss_attention_mask = None

        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        # Text
        max_text_len = 40
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Random crop
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        # "Z Normalization"
        motion = (motion - self.mean) / self.std

        return caption, motion, m_length, word_embeddings, pos_one_hots, sent_len, motion_token, gloss_embedding, sample_name, all_captions, gloss_attention_mask


    def pad_motion(self, motion, target_length):
        current_length = motion.shape[0]
        if current_length >= target_length:
            return motion

        pad_length = target_length - current_length

        if self.padding_strategy == "mirror":
            # Mirror padding builds a cycle from reversed tail and forward head,
            # then tiles and slices to the required pad_length.
            if current_length > 1:
                # backward_chunk: reverse motion except the last frame
                # e.g., [a, b, c, d] -> [c, b, a]
                backward_chunk = motion[:-1][::-1]
                # forward_chunk: forward motion except the first frame
                # e.g., [a, b, c, d] -> [b, c, d]
                forward_chunk = motion[1:]

                # One full mirror cycle
                cycle = np.concatenate([backward_chunk, forward_chunk], axis=0)

                if len(cycle) > 0:
                    # Number of repeats to cover pad_length
                    repeat_times = (pad_length + len(cycle) - 1) // len(cycle)
                    # Tile over the feature dimension unchanged
                    if motion.ndim == 1:
                        full_padding = np.tile(cycle, repeat_times)
                    else:
                        full_padding = np.tile(cycle, (repeat_times, 1))

                    padding = full_padding[:pad_length]
                else:
                    # Fallback: repeat the last frame if cycle is empty (shouldn't happen if current_length > 1)
                    last = motion[-1:]
                    if motion.ndim == 1:
                        padding = np.tile(last, pad_length)
                    else:
                        padding = np.tile(last, (pad_length, 1))
            else:
                # If only one frame, repeat it
                last = motion[-1:]
                if motion.ndim == 1:
                    padding = np.tile(last, pad_length)
                else:
                    padding = np.tile(last, (pad_length, 1))

            return np.concatenate([motion, padding], axis=0)

        elif self.padding_strategy == "interpolate":
            x_original = np.arange(current_length)
            x_new = np.linspace(0, current_length - 1, target_length)
            interp_func = interp1d(
                x_original,
                motion,
                axis=0,
                kind="nearest", # nearest linear
                fill_value="extrapolate",
                assume_sorted=True,
            )
            padded_motion = interp_func(x_new)
            return padded_motion

    def truncate_motion(self, motion, target_length):
        """
        Truncate motion sequence to target_length using the selected strategy.

        Args:
            motion (np.ndarray): [T, D] or [T]
            target_length (int): desired sequence length

        Returns:
            np.ndarray: Truncated motion of shape [target_length, D] (or [target_length]).
        """
        current_length = motion.shape[0]
        if current_length <= target_length:
            return motion

        if self.truncate_strategy == "random":
            start_idx = random.randint(0, current_length - target_length)
            return motion[start_idx : start_idx + target_length]
        elif self.truncate_strategy == "subsample":
            # Uniformly sample target_length indices from [0, current_length-1]
            indices = np.linspace(0, current_length - 1, target_length)
            indices = np.round(indices).astype(np.int64)
            return motion[indices]

        # Fallback to start truncation
        return motion[:target_length]
