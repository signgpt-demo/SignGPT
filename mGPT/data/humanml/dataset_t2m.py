import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from scipy.interpolate import interp1d

class Text2MotionDataset(data.Dataset):

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
        tmpFile=True,
        tiny=False,
        debug=False,
        use_gloss=False,
        **kwargs,
    ):

        self.max_length = 1
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.use_gloss = use_gloss

        # Data mean and std
        self.mean = mean
        self.std = std

        # Data path
        split_file = pjoin(data_root, split + '.txt')
        motion_dir = pjoin(data_root, 'all_73j_new_joint_vecs_onlylocal_cat_expression')
        text_dir = pjoin(data_root, 'TEXT_new_tackle')

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
        length_list = []
        data_dict = {}

        # Fast loading
        if os.path.exists(pjoin(data_root, f'tmp/{split}{subset}_data.pkl')):
            if tiny or debug:
                with open(pjoin(data_root, f'tmp/{split}{subset}_data.pkl'),
                          'rb') as file:
                    data_dict = pickle.load(file)
            else:
                with rich.progress.open(
                        pjoin(data_root, f'tmp/{split}{subset}_data.pkl'),
                        'rb',
                        description=f"Loading HumanML3D {split}") as file:
                    data_dict = pickle.load(file)
            with open(pjoin(data_root, f'tmp/{split}{subset}_index.pkl'),
                      'rb') as file:
                name_list = pickle.load(file)
            new_name_list = list(name_list)
            for name in new_name_list:
                length_list.append(data_dict[name]['length'])

        else:
            for idx, name in enumerator:
                if len(new_name_list) > maxdata:
                    break
                try:
                    motion = np.load(pjoin(motion_dir, name + ".npy"))

                    if (len(motion)) < self.min_motion_length or (len(motion) >= self.max_motion_length):
                        continue

                    text_data = []
                    flag = False
                    with cs.open(pjoin(text_dir, name + '.txt')) as f:
                        lines = f.readlines()

                        for line in lines:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            flag = True
                            text_data.append(text_dict)

                    if flag:
                        data_dict[name] = {
                            'motion': motion,
                            "length": len(motion),
                            'text': text_data,


                        }
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e:
                    if idx < 3:
                        print(f"[WARNING] Failed to load sample '{name}': {e}")
                    pass

            name_list, length_list = zip(
                *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

            if tmpFile:
                os.makedirs(pjoin(data_root, 'tmp'), exist_ok=True)
                with open(pjoin(data_root, f'tmp/{split}{subset}_data.pkl'),
                          'wb') as file:
                    pickle.dump(data_dict, file)
                with open(pjoin(data_root, f'tmp/{split}{subset}_index.pkl'),
                          'wb') as file:
                    pickle.dump(name_list, file)

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.nfeats = data_dict[name_list[0]]['motion'].shape[1]
        self.reset_max_len(self.max_length)
        self.padding_strategy = "interpolate"     # options: 'mirror', 'interpolate'
        self.truncate_strategy = "subsample"      # options: 'random', 'start', 'end', 'subsample'

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]

        text_data = random.choice(text_list)
        caption = text_data["caption"]

        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        # Process motion to fixed target length
        if m_length > self.max_motion_length:
            motion = self.truncate_motion(motion, self.max_motion_length)
        elif m_length < self.min_motion_length:
            motion = self.pad_motion(motion, self.min_motion_length)
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            idx = 0
            motion = motion[idx:idx + m_length]
        m_length = motion.shape[0]

        motion = (motion - self.mean)  / self.std

        return caption, motion, m_length, None, None, None, None, None, all_captions


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
