"""Step 0: merge per-frame pose pkl chunks into one pkl per clip.

Some pose-estimation pipelines emit one pkl per frame (or per small chunk)
under ``<raw_dir>/<clip_name>/images*.pkl``. The SignGPT training pipeline
expects one merged pkl per clip with stacked [T, ...] arrays.

Two clip layouts are supported downstream (steps 1 and 3 auto-detect):

* How2Sign style: a single ``smplx`` key holding a concatenated
  [T, 182] parameter vector.
* PHOENIX style: separate keys (``smplx_root_pose``, ``smplx_body_pose``,
  ``smplx_lhand_pose``, ``smplx_rhand_pose``, ``smplx_jaw_pose``,
  ``smplx_shape``, ``smplx_expr``, ``cam_trans``).

Usage:
    python -m preprocess.step0_combine_pose_pkls \
        --raw_dir /path/to/phoenix_poses/test \
        --out_dir /path/to/all_pkls_144
"""

import argparse
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

# Keys expected in every per-frame chunk (PHOENIX style) with per-frame shapes.
EXPECTED_KEYS = {
    'smplx_root_pose': (3,),
    'smplx_body_pose': (63,),
    'smplx_lhand_pose': (45,),
    'smplx_rhand_pose': (45,),
    'smplx_jaw_pose': (3,),
    'smplx_shape': (10,),
    'smplx_expr': (10,),
    'cam_trans': (3,),
}


def merge_pkls_in_dir(dir_path):
    """Concatenate all chunk pkls of one clip into stacked [T, ...] arrays."""
    pkl_files = sorted(
        f for f in os.listdir(dir_path)
        if f.endswith('.pkl') and not os.path.basename(f).startswith('._')
    )
    if not pkl_files:
        return None

    merged_flat = {key: [] for key in EXPECTED_KEYS}
    for pkl_file in pkl_files:
        with open(os.path.join(dir_path, pkl_file), 'rb') as f:
            data = pickle.load(f)

        for key, expected_shape in EXPECTED_KEYS.items():
            if key not in data:
                raise ValueError(f'Missing key {key} in {pkl_file}')
            arr = np.asarray(data[key])
            # Accept per-frame (D,) chunks and multi-frame (N, D) chunks
            if arr.ndim == 1:
                if tuple(arr.shape) != expected_shape:
                    raise ValueError(
                        f'Shape mismatch for {key} in {pkl_file}: expected '
                        f'{expected_shape}, got {arr.shape}'
                    )
            elif arr.ndim == 2 and tuple(arr.shape[1:]) == expected_shape:
                pass
            else:
                raise ValueError(
                    f'Shape mismatch for {key} in {pkl_file}: expected '
                    f'({expected_shape},) or (N, {expected_shape}), got {arr.shape}'
                )
            merged_flat[key].append(arr)

    # Per-frame chunks stack into [T, D]; multi-frame chunks concatenate
    merged = {}
    for key, parts in merged_flat.items():
        merged[key] = np.stack(parts, axis=0) if parts[0].ndim == 1 \
            else np.concatenate(parts, axis=0)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw_dir', required=True,
                        help='Directory containing one sub-directory of pkl chunks per clip')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for merged per-clip pkls')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    clips = sorted(
        d for d in os.listdir(args.raw_dir)
        if os.path.isdir(os.path.join(args.raw_dir, d)) and not d.startswith(('.', '__'))
    )
    if not clips:
        print(f'No clip sub-directories found in {args.raw_dir}', file=sys.stderr)
        sys.exit(1)

    n_done, n_failed = 0, 0
    for clip in tqdm(clips, desc='Merging clips'):
        try:
            merged = merge_pkls_in_dir(os.path.join(args.raw_dir, clip))
        except Exception as e:
            print(f'[ERROR] {clip}: {e}', file=sys.stderr)
            n_failed += 1
            continue
        if merged is None:
            continue
        with open(os.path.join(args.out_dir, f'{clip}.pkl'), 'wb') as f:
            pickle.dump(merged, f)
        n_done += 1

    print(f'Done: {n_done} clips merged, {n_failed} failed -> {args.out_dir}')


if __name__ == '__main__':
    main()
