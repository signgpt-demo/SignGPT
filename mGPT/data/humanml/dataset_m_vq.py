import os
import random
import numpy as np
import torch
from torch.utils import data
from os.path import join as pjoin
from scipy.interpolate import interp1d

from .dataset_t2m import Text2MotionDataset

class MotionDatasetVQ(Text2MotionDataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        use_gloss=False,
        **kwargs,
    ):
        super().__init__(
            data_root,
            split,
            mean,
            std,
            max_motion_length,
            min_motion_length,
            unit_length,
            fps,
            tmpFile,
            tiny,
            debug,
            **kwargs,
        )

        self.window_size = win_size
        self.use_gloss = use_gloss
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length

        name_list = list(self.name_list)
        self.name_list = name_list

        # Pre-load and pre-pool text embeddings into memory at init time.
        # Stores masked-mean-pooled [D] vectors instead of full [T, D] tensors
        # to minimize memory (~120 MB vs ~7.5 GB for 15k samples).
        self._text_embed_cache = {}
        if self.use_gloss:
            text_embed_dir = pjoin(data_root, "text_embeddings")
            text_mask_dir = pjoin(data_root, "text_attention_masks")
            loaded, skipped = 0, 0
            for name in name_list:
                emb_path = pjoin(text_embed_dir, name + ".npy")
                mask_path = pjoin(text_mask_dir, name + ".npy")
                if os.path.exists(emb_path) and os.path.exists(mask_path):
                    emb = np.load(emb_path).astype(np.float32)   # [T_t, D]
                    mask = np.load(mask_path).astype(np.float32)  # [T_t]
                    mask_sum = mask.sum()
                    if mask_sum > 0:
                        global_emb = (emb * mask[:, None]).sum(axis=0) / mask_sum
                    else:
                        global_emb = emb.mean(axis=0)
                    self._text_embed_cache[name] = torch.from_numpy(global_emb)  # [D]
                    loaded += 1
                else:
                    skipped += 1
            print(f"Text embeddings cached: {loaded} loaded, {skipped} missing")

        self.padding_strategy = "interpolate"     # options: 'mirror', 'interpolate'
        self.truncate_strategy = "random_subsample"      # options: 'random', 'subsample', 'random_subsample'

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):

        idx = self.pointer + item
        name = self.name_list[idx]
        data_i = self.data_dict[name]

        motion, m_length = data_i["motion"], data_i["length"]

        # Retrieve pre-pooled text embedding from memory cache (no disk I/O).
        # All win_size sub-windows from the same motion share the same embedding.
        text_embedding = self._text_embed_cache.get(name, None)  # [D] or None

        # Sample a fixed-size VAE training window.
        if motion.shape[0] < self.window_size:
            motion = self.pad_motion(motion, self.window_size)
        idx = random.randint(0, motion.shape[0] - self.window_size)
        motion = motion[idx:idx + self.window_size]

        m_length = motion.shape[0]

        motion = (motion - self.mean) / self.std

        try:
            caption = data_i["text"][0]["caption"]
        except Exception:
            caption = None

        return (
            None,
            motion,
            m_length,
            None,
            None,
            None,
            None,
            text_embedding,
            name,
            caption,
            None,
        )

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
        """Truncate a motion sequence with the configured sampling strategy."""
        current_length = motion.shape[0]

        if current_length <= target_length:
            return motion

        if self.truncate_strategy == "random":
            start_idx = random.randint(0, current_length - target_length)
            return motion[start_idx : start_idx + target_length]

        elif self.truncate_strategy == "subsample":
            indices = np.linspace(0, current_length - 1, target_length)
            indices = np.round(indices).astype(np.int64)
            return motion[indices]

        elif self.truncate_strategy == "random_subsample":
            all_indices = np.arange(current_length)
            chosen_indices = np.random.choice(all_indices, size=target_length, replace=False)
            sorted_indices = np.sort(chosen_indices)
            return motion[sorted_indices]

        return motion[:target_length]
