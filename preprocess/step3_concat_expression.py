"""Step 3: concatenate SMPL-X expression parameters onto the motion features.

Reads the [T-1, 220] feature files of step2 and the merged pose pkls,
then appends the 10 expression parameters per frame (skipping the first
frame, which step2 drops), producing the final [T-1, 230] features named
``all_73j_new_joint_vecs_onlylocal_cat_expression`` in the original code.

Usage:
    python -m preprocess.step3_concat_expression \
        --vecs_dir /path/to/all_73j_new_joint_vecs_onlylocal \
        --pkl_dir /path/to/all_pkls_144 \
        --out_dir /path/to/all_73j_new_joint_vecs_onlylocal_cat_expression
"""

import argparse
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm


def extract_expression(data):
    """Return the [T, 10] expression parameters from either pkl layout."""
    if 'smplx_expr' in data:
        expr = data['smplx_expr']
    elif 'smplx' in data:
        # How2Sign concatenated layout: expression occupies columns 169:179
        expr = data['smplx'][:, 169:179]
    else:
        raise KeyError('No expression parameters found in pkl '
                       '(expected "smplx_expr" or "smplx")')
    if not isinstance(expr, np.ndarray):
        expr = expr.numpy()
    return np.asarray(expr, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vecs_dir', required=True,
                        help='Directory with [T-1, 220] feature npy files (step2 output)')
    parser.add_argument('--pkl_dir', required=True,
                        help='Directory with merged per-clip pkls (step0 output)')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for the [T-1, 230] features')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    n_done, n_failed = 0, 0
    for npy_name in tqdm(sorted(f for f in os.listdir(args.vecs_dir) if f.endswith('.npy')),
                         desc='Concatenating expression'):
        try:
            feats = np.load(os.path.join(args.vecs_dir, npy_name))
            with open(os.path.join(args.pkl_dir, npy_name.replace('.npy', '.pkl')), 'rb') as f:
                data = pickle.load(f)
            expr = extract_expression(data)

            # step2 drops the first frame, so the expression stream is offset by one
            out = np.concatenate([feats, expr[1:len(feats) + 1]], axis=-1)
            np.save(os.path.join(args.out_dir, npy_name), out)
            n_done += 1
        except Exception as e:
            print(f'[ERROR] {npy_name}: {e}', file=sys.stderr)
            n_failed += 1

    print(f'Done: {n_done} clips concatenated, {n_failed} failed -> {args.out_dir}')


if __name__ == '__main__':
    main()
