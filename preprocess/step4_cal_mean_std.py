"""Step 4: compute per-dimension mean/std normalization statistics.

Aggregates every frame of the [T-1, 230] features and produces
``mean_seperate_catexp_new_new.npy`` / ``std_seperate_catexp_new_new.npy``
inside the target meta directory. Point the training config's
``SIGNGPT_STATS_ROOT`` at the directory that contains
``t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/``.

Normalization details inherited from the original preprocessing:

* The three root-velocity dims get their std replaced by the group mean
  (dims 1:3 additionally share one value).
* The local-joint std is averaged separately over body joints
  (dims 4:67 and 157:190) and hand joints (dims 67:157 and 190:220),
  matching the body/hand split of the VQ-VAE.
* The 10 expression dims share one std.
* How2Sign's SMPL-X fits carry no global translation, so its root
  linear velocity is exactly zero; ``--xz_velocity_std one`` replaces
  the (near-zero) std of dims 1:3 with 1.0 to avoid noise amplification.
  The original How2Sign release stats use this flag; PHOENIX keeps the
  raw values.

Usage:
    python -m preprocess.step4_cal_mean_std \
        --feat_dir /path/to/all_73j_new_joint_vecs_onlylocal_cat_expression \
        --out_dir /path/to/deps/ASL/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta \
        --xz_velocity_std one
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm


def compute_stats(feat_dir, joints_num, xz_velocity_std):
    file_list = sorted(f for f in os.listdir(feat_dir) if f.endswith('.npy'))
    data_list = []
    for file_name in tqdm(file_list, desc='Aggregating features', unit='file'):
        data = np.load(os.path.join(feat_dir, file_name))
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    # Root-velocity dims (0: angular, 1:3 linear XZ, 3: root height)
    std[0:1] = std[0:1].mean()
    std[1:3] = std[1:3].mean()
    std[3:4] = std[3:4].mean()

    if xz_velocity_std == 'one':
        # Constant root position (no global translation in the fits):
        # avoid amplifying numerical noise by a near-zero std.
        std[1:3] = 1.0

    # Separate body/hand std groups, matching the VQ-VAE body/hand split.
    # local layout: body joints 1..21 (63), hand joints (90), face+feet (33),
    # fingertips (30) — face/feet group with the body, fingertips with the hand.
    local_start = 4
    body_idx = list(range(local_start, local_start + 63)) + \
               list(range(local_start + 63 + 90, local_start + 63 + 90 + 33))
    hand_idx = list(range(local_start + 63, local_start + 63 + 90)) + \
               list(range(local_start + 63 + 90 + 33, local_start + (joints_num - 1) * 3))
    std[body_idx] = std[body_idx].mean()
    std[hand_idx] = std[hand_idx].mean()

    # Expression dims share one std
    std[local_start + (joints_num - 1) * 3:] = std[local_start + (joints_num - 1) * 3:].mean()

    expected = local_start + (joints_num - 1) * 3 + 10
    assert std.shape[-1] == expected, \
        f'Expected {expected} dims, got {std.shape[-1]}'

    return mean, std


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--feat_dir', required=True,
                        help='Directory with [T-1, 230] feature npy files (step3 output)')
    parser.add_argument('--out_dir', required=True,
                        help='Meta directory the stats are written to '
                             '(.../t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta)')
    parser.add_argument('--joints_num', type=int, default=73)
    parser.add_argument('--xz_velocity_std', choices=['raw', 'one'], default='raw',
                        help="'one' for How2Sign-style zero-velocity handling, "
                             "'raw' otherwise")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mean, std = compute_stats(args.feat_dir, args.joints_num, args.xz_velocity_std)

    np.save(os.path.join(args.out_dir, 'mean_seperate_catexp_new_new.npy'), mean)
    np.save(os.path.join(args.out_dir, 'std_seperate_catexp_new_new.npy'), std)

    print(f'mean: {mean[:4]}')
    print(f'std:  {std[:4]}')
    print(f'Zero stds remaining: {(std == 0).sum()}')
    print(f'Saved stats to {args.out_dir}')


if __name__ == '__main__':
    main()
