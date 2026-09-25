# Data preprocessing

This directory contains the full preprocessing pipeline that turns raw
SMPL-X pose estimates and official text annotations into the layout the
SignGPT training code expects. All scripts are runnable as modules from
the repository root, e.g. `python -m preprocess.step0_combine_pose_pkls
--help`.

Dependencies: `numpy torch smplx tqdm spacy` plus the spaCy model of the
target language (`en_core_web_sm` for How2Sign, `de_core_news_sm` for
PHOENIX-2014T). SMPL-X model files (`SMPLX_NEUTRAL.pkl`, ...) must be
obtained from https://www.smpl-x.de/ and placed under
`<deps>/smpl_models/smplx/`.

## Pipeline overview

| Step | Input | Output |
|---|---|---|
| `step0_combine_pose_pkls` | `<raw>/<clip>/*.pkl` frame chunks | `<all_pkls_144>/<clip>.pkl` (stacked [T, ...] params) |
| `step1_smplx_to_joints` | merged pkls | `<joints>/<clip>.npy` [T, 73, 3] |
| `step2_joints_to_features` | joint npys | features [T-1, 220] + normalized joints [T-1, 73, 3] |
| `step3_concat_expression` | features + merged pkls | final features [T-1, 230] |
| `step4_cal_mean_std` | final features | `mean/std_seperate_catexp_new_new.npy` |
| `step5_prepare_texts` | official CSVs | `TEXT_new_tackle/`, split txts, prompt templates |

Steps 0 and 1 need a GPU-accelerated or CPU SMPL-X installation; steps
2-4 are CPU-only and fast.

## Dataset-specific notes

* **How2Sign** (ASL). The released pose archive already contains one
  merged pkl per clip, so step 0 is not needed. Step 4 must be run with
  `--xz_velocity_std one` because the fits carry no global translation
  and the root linear velocity is identically zero.
* **PHOENIX-2014T** (DGS). The pose archive stores one directory per
  clip with per-frame pkl chunks, so step 0 merges them first. Step 4
  uses the default `--xz_velocity_std raw`.

## Reference layout produced for training

```text
<dataset_root>/                                  # SIGNGPT_DATA_ROOT
├── train.txt / val.txt / test.txt               # also train_all.txt for stage 1
├── template_pretrain.json
├── template_instructions.json
├── TEXT_new_tackle/<clip>.txt
└── all_73j_new_joint_vecs_onlylocal_cat_expression/<clip>.npy

<stats_root>/                                    # SIGNGPT_STATS_ROOT
└── t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/
    ├── mean_seperate_catexp_new_new.npy
    └── std_seperate_catexp_new_new.npy
```

`all_pkls_144`, `all_npys_smlph_73j`,
`all_73j_new_joints_onlylocal` and
`all_73j_new_joint_vecs_onlylocal` are intermediate outputs and can live
outside the training dataset root.
