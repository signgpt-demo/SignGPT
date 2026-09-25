import numpy as np
import torch
import os
from os.path import join as pjoin
from .humanml.utils.word_vectorizer import WordVectorizer
from .humanml.scripts.motion_process import (process_file, recover_from_ric)
from . import BASEDataModule
from .humanml import Text2MotionDatasetEval, Text2MotionDataset, Text2MotionDatasetCB, MotionDatasetVQ, Text2MotionDatasetToken
from .utils import humanml3d_collate

import os
import numpy as np
import torch
from os.path  import join as pjoin
from torch.utils.data  import DataLoader


class HumanML3DDataModule(BASEDataModule):
    """Data module for HumanML3D: supports multi-stage training and evaluation."""

    def __init__(self, cfg, **kwargs):
        """
        Initialize the data module.
        Args:
            cfg: config object with dataset and training parameters.
            **kwargs: optional args.
        """
        # Set collate function
        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)

        # Dataset metadata.
        cfg.DATASET.JOINT_TYPE = 'humanml3d'
        self.name = "humanml3d"
        self.njoints = 73

        # Dataset paths.
        data_root = cfg.DATASET.HUMANML3D.ROOT
        self.hparams.data_root = data_root
        self.hparams.text_dir = pjoin(data_root, "TEXT_tackle_with_punctuation_uppercase")
        self.hparams.motion_dir = pjoin(data_root, 'all_73j_new_joint_vecs_onlylocal_cat_expression')

        # Normalization statistics.
        # Training stats
        dis_data_root = pjoin(
            cfg.DATASET.HUMANML3D.MEAN_STD_PATH,
            't2m',
            "VQVAEV3_CB1024_CMT_H1024_NRES3",
            "meta"
        )
        self.hparams.mean = np.load(pjoin(dis_data_root, "mean_seperate_catexp_new_new.npy"))
        self.hparams.std = np.load(pjoin(dis_data_root, "std_seperate_catexp_new_new.npy"))

        # Eval stats (for fair comparison)
        dis_data_root_eval = pjoin(
            cfg.DATASET.HUMANML3D.MEAN_STD_PATH,
            't2m',
            "VQVAEV3_CB1024_CMT_H1024_NRES3",
            "meta"
        )
        self.hparams.mean_eval = np.load(pjoin(dis_data_root_eval, "mean_seperate_catexp_new_new.npy"))
        self.hparams.std_eval = np.load(pjoin(dis_data_root_eval, "std_seperate_catexp_new_new.npy"))

        # Sequence-length limits.
        self.hparams.max_motion_length = cfg.DATASET.HUMANML3D.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.HUMANML3D.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.HUMANML3D.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.HUMANML3D.UNIT_LEN

        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.w_vectorizer = WordVectorizer(
            cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab"
        )

        # Dataset implementation for the selected stage.
        self.DatasetEval = Text2MotionDatasetEval

        if cfg.TRAIN.STAGE == "vae":
            self.hparams.win_size = cfg.model.params.motion_vae.params.win_size
            self.hparams.use_gloss = getattr(cfg.model.params.motion_vae.params, 'use_gloss', False)
            self.Dataset = MotionDatasetVQ
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.code_path = cfg.DATASET.CODE_PATH
            self.hparams.task_path = cfg.DATASET.TASK_PATH
            self.hparams.std_text = cfg.DATASET.HUMANML3D.STD_TEXT
            self.Dataset = Text2MotionDatasetCB
        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = Text2MotionDatasetToken
            self.DatasetEval = Text2MotionDatasetToken
        else:
            self.Dataset = Text2MotionDataset

        self._sample_set = self.get_sample_set(overrides={"split": "test", "tiny": True})
        self.nfeats = self._sample_set.nfeats
        cfg.DATASET.NFEATS = self.nfeats

    def feats2joints(self, features):
        """Convert feature vectors to joint positions (3D)."""
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean  # de-normalize
        features = features[..., 0:-10]
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        """Convert joint positions to feature vectors."""
        ref = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        ref = ref.reshape(len(ref), -1, 3)
        ref = torch.from_numpy(ref)
        features = process_file(features, self.njoints, ref, 't2m')[0]
        return features

    def normalize(self, features):
        """Normalize with training stats."""
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        return (features - mean) / std

    def denormalize(self, features):
        """De-normalize with training stats."""
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        return features * std + mean

    def renorm4t2m(self, features):
        """Re-normalize to T2M evaluator stats."""
        epsilon = 1e-8
        self.hparams.std[self.hparams.std == 0] = epsilon
        self.hparams.std_eval[self.hparams.std_eval == 0] = epsilon

        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)

        features = features * ori_std + ori_mean
        return (features - eval_mean) / eval_std

    def mm_mode(self, mm_on=True):
        """Toggle multi-modal evaluation."""
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            n_mm = min(self.cfg.METRIC.MM_NUM_SAMPLES, len(self.name_list))
            self.mm_list = np.random.choice(self.name_list, n_mm, replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
