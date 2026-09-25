import random
import numpy as np
from torch.utils import data
from .dataset_t2m import Text2MotionDataset
import codecs as cs
from os.path import join as pjoin
from scipy.interpolate import interp1d
class Text2MotionDatasetToken(data.Dataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=24,
        unit_length=4,
        fps=20,
        tmpFile=False,
        tiny=False,
        debug=False,
        **kwargs,
    ):

        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        print(self.max_motion_length,self.min_motion_length)
        self.unit_length = unit_length

        # Data mean and std
        self.mean = mean
        self.std = std

        split = 'train_all'

        # Data path
        split_file = pjoin(data_root, split + '.txt')
        motion_dir = pjoin(data_root, 'all_73j_new_joint_vecs_onlylocal_cat_expression')
        text_dir = pjoin(data_root, 'TEXT_tackle_with_punctuation_uppercase')

        # Data id list
        self.id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        new_name_list = []
        length_list = []
        data_dict = {}
        for name in self.id_list:
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                if (len(motion)) <  self.min_motion_length or (len(motion) >= self.max_motion_length):
                    continue

                data_dict[name] = {'motion': motion,
                                'length': len(motion),
                                'name': name}
                new_name_list.append(name)

                length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list
        self.nfeats = motion.shape[-1]
        self.padding_strategy = "interpolate"     # options: 'mirror', 'interpolate'
        self.truncate_strategy = "subsample"      # options: 'random', 'start', 'end', 'subsample'

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        # Process motion to fixed target length
        if m_length > self.max_motion_length:
            motion = self.truncate_motion(motion, self.max_motion_length)
        elif m_length < self.min_motion_length:
            motion = self.pad_motion(motion, self.min_motion_length)
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]
        m_length = motion.shape[0]
        motion = (motion - self.mean) / self.std

        return name, motion, m_length, True, True, True, True, True, True


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
