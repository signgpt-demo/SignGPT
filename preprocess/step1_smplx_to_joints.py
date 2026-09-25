"""Step 1: run SMPL-X forward kinematics and extract 73 joints.

Reads the merged per-clip pkls (see step0) and outputs one
``<clip>.npy`` of shape [T, 73, 3] per clip, containing camera-space
joint positions from the neutral-body SMPL-X model. Global translation
(``cam_trans``) is intentionally NOT applied, matching the original
SignGPT preprocessing.

Both supported pkl layouts are handled:

* How2Sign style: ``smplx`` key with a concatenated [T, 182] vector laid
  out as root(3) body(63) lhand(45) rhand(45) jaw(3) betas(10)
  expression(10) transl(3).
* PHOENIX style: separate ``smplx_root_pose`` / ``smplx_body_pose`` /
  ``smplx_lhand_pose`` / ``smplx_rhand_pose`` / ``smplx_jaw_pose`` /
  ``smplx_shape`` / ``smplx_expr`` keys.

Usage:
    python -m preprocess.step1_smplx_to_joints \
        --pkl_dir /path/to/all_pkls_144 \
        --out_dir /path/to/all_npys_smlph_73j \
        --smplx_model_path /path/to/deps/smpl_models
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.skeleton_73j import JOINT_IDX_73  # noqa: E402

# Parameter slice boundaries of the concatenated How2Sign layout.
SMPLX_LAYOUT_182 = {
    'root_pose': (0, 3),
    'body_pose': (3, 66),
    'left_hand_pose': (66, 111),
    'right_hand_pose': (111, 156),
    'jaw_pose': (156, 159),
    'betas': (159, 169),
    'expression': (169, 179),
    'transl': (179, 182),
}


def load_smplx_params(data):
    """Normalize either pkl layout into a dict of float tensors [T, D]."""
    if isinstance(data, dict) and 'smplx' in data:
        vec = torch.as_tensor(np.asarray(data['smplx'], dtype=np.float32))
        return {name: vec[:, lo:hi] for name, (lo, hi) in SMPLX_LAYOUT_182.items()}

    mapping = {
        'root_pose': 'smplx_root_pose',
        'body_pose': 'smplx_body_pose',
        'left_hand_pose': 'smplx_lhand_pose',
        'right_hand_pose': 'smplx_rhand_pose',
        'jaw_pose': 'smplx_jaw_pose',
        'betas': 'smplx_shape',
        'expression': 'smplx_expr',
        'transl': 'cam_trans',
    }
    params = {}
    for name, key in mapping.items():
        if key not in data:
            raise KeyError(f'Missing key {key!r} in pkl')
        arr = data[key]
        if torch.is_tensor(arr):
            arr = arr.numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if name == 'transl' and arr.ndim == 1 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)  # defensive: flat [T*3] vector
        if arr.ndim == 1 and arr.size % 3 == 0 and name.endswith('pose'):
            arr = arr.reshape(-1, 3)  # defensive: flat [T*3] pose vector
        params[name] = torch.as_tensor(arr)
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pkl_dir', required=True,
                        help='Directory with merged per-clip pkls (step0 output)')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for [T, 73, 3] joint npy files')
    parser.add_argument('--smplx_model_path', required=True,
                        help='Directory that contains the smplx/ model folder '
                             '(e.g. deps/smpl_models)')
    parser.add_argument('--frame_batch', type=int, default=64,
                        help='Number of frames per SMPL-X forward pass')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    import smplx  # imported lazily so --help works without the dependency

    os.makedirs(args.out_dir, exist_ok=True)

    model = smplx.create(args.smplx_model_path, model_type='smplx', gender='NEUTRAL',
                         use_pca=False, use_face_contour=True,
                         create_global_orient=False, create_body_pose=False,
                         create_left_hand_pose=False, create_right_hand_pose=False,
                         create_jaw_pose=False, create_leye_pose=False,
                         create_reye_pose=False, create_betas=False,
                         create_expression=False, create_transl=False)
    model = model.to(args.device)
    model.eval()

    pkl_files = sorted(f for f in os.listdir(args.pkl_dir) if f.endswith('.pkl'))
    n_done, n_failed = 0, 0
    with torch.no_grad():
        for pkl_name in tqdm(pkl_files, desc='SMPL-X FK'):
            try:
                with open(os.path.join(args.pkl_dir, pkl_name), 'rb') as f:
                    data = pickle.load(f)
                params = load_smplx_params(data)
                params = {k: v.to(args.device) for k, v in params.items()}

                num_frames = params['root_pose'].shape[0]
                zero_pose = torch.zeros(num_frames, 3, device=args.device)
                joints = []
                for start in range(0, num_frames, args.frame_batch):
                    sl = slice(start, min(start + args.frame_batch, num_frames))
                    output = model(
                        betas=params['betas'][sl],
                        body_pose=params['body_pose'][sl],
                        global_orient=params['root_pose'][sl],
                        left_hand_pose=params['left_hand_pose'][sl],
                        right_hand_pose=params['right_hand_pose'][sl],
                        jaw_pose=params['jaw_pose'][sl],
                        leye_pose=zero_pose[sl],
                        reye_pose=zero_pose[sl],
                        expression=params['expression'][sl],
                    )
                    # [B, 144, 3] -> [B, 73, 3]
                    joints.append(output.joints[:, JOINT_IDX_73, :].cpu().numpy())
                joints = np.concatenate(joints, axis=0)

                if np.any(np.isnan(joints)):
                    raise ValueError('NaN joints produced by forward kinematics')

                out_path = os.path.join(args.out_dir, pkl_name.replace('.pkl', '.npy'))
                np.save(out_path, joints.astype(np.float32))
                n_done += 1
            except Exception as e:
                print(f'[ERROR] {pkl_name}: {e}', file=sys.stderr)
                n_failed += 1

    print(f'Done: {n_done} clips converted, {n_failed} failed -> {args.out_dir}')


if __name__ == '__main__':
    main()
