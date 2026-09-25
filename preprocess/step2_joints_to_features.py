"""Step 2: skeleton normalization and HumanML3D-style feature extraction.

Converts the [T, 73, 3] joint sequences of step1 into

* ``<out_vecs_dir>/<clip>.npy``  — [T-1, 220] motion features:
  root angular velocity (1) + root linear velocity XZ (2) + root height (1)
  + root-relative local joint positions of joints 1..72 (216)
* ``<out_joints_dir>/<clip>.npy`` — [T-1, 73, 3] normalized joint
  positions recovered from the features

Each clip is skeleton-normalized (uniform rescale to a reference
skeleton, floor alignment, canonical initial orientation facing +Z)
before feature extraction, following HumanML3D.

Usage:
    python -m preprocess.step2_joints_to_features \
        --joints_dir /path/to/all_npys_smlph_73j \
        --out_vecs_dir /path/to/all_73j_new_joint_vecs_onlylocal \
        --out_joints_dir /path/to/all_73j_new_joints_onlylocal \
        [--reference /path/to/first_clip.npy]
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.common.quaternion import (  # noqa: E402
    qbetween_np, qrot_np, qmul_np, qinv_np, qrot, qinv, quaternion_to_cont6d_np,
)
from preprocess.common.skeleton import Skeleton  # noqa: E402
from preprocess.skeleton_73j import (  # noqa: E402
    NEW_T2M_RAW_OFFSETS, NEW_T2M_KINEMATIC_CHAIN, FACE_JOINT_INDX, N_JOINTS,
)


def uniform_skeleton(positions, target_offset):
    """Retarget a motion to a reference skeleton via IK + FK."""
    src_skel = Skeleton(torch.from_numpy(NEW_T2M_RAW_OFFSETS),
                        NEW_T2M_KINEMATIC_CHAIN, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    # Rescale so the leg length matches the reference skeleton
    l_idx1, l_idx2 = 5, 8  # right_knee, right_ankle
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
    scale_rt = tgt_leg_len / src_leg_len

    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    # Inverse kinematics on the source positions, then forward kinematics
    # with the target bone lengths
    quat_params = src_skel.inverse_kinematics_np(positions, FACE_JOINT_INDX)
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)

    return new_joints


def recover_from_ric(data, joints_num):
    """Recover global joint positions from the 220-dim feature vector."""
    # Root rotation and translation
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]

    # Local joint positions back to global
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def process_file(positions, target_offset):
    """Normalize one clip and extract the 220-dim feature sequence."""
    # 1. Retarget to the reference skeleton
    positions = uniform_skeleton(positions, target_offset)

    # 2. Floor alignment
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # 3. Move the initial root position to the XZ origin
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # 4. Rotate so the clip starts facing +Z
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions = qrot_np(root_quat_init, positions)

    global_positions = positions.copy()

    # 5. Per-frame joint rotations (IK) and root motion
    def get_cont6d_params(positions):
        skel = Skeleton(torch.from_numpy(NEW_T2M_RAW_OFFSETS),
                        NEW_T2M_KINEMATIC_CHAIN, 'cpu')
        quat_params = skel.inverse_kinematics_np(positions, FACE_JOINT_INDX,
                                                 smooth_forward=True)
        cont_6d_params = np.nan_to_num(quaternion_to_cont6d_np(quat_params), nan=0.0)

        r_rot = quat_params[:, 0].copy()
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot

    def get_rifke(positions, r_rot):
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions, r_rot)

    # 6. Pack the root stream (4 dims)
    root_y = positions[:, 0, 1:2]
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    # 7. Pack the local joint positions (72 * 3 dims, drop last frame)
    ric_data = positions[:, 1:].reshape(len(positions), -1)
    data = np.concatenate([root_data, ric_data[:-1]], axis=-1)

    return data, global_positions, positions, l_velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--joints_dir', required=True,
                        help='Directory with [T, 73, 3] joint npy files (step1 output)')
    parser.add_argument('--out_vecs_dir', required=True,
                        help='Output directory for [T-1, 220] feature npy files')
    parser.add_argument('--out_joints_dir', required=True,
                        help='Output directory for recovered [T-1, 73, 3] joint npy files')
    parser.add_argument('--reference', default=None,
                        help='Joint npy used as the target skeleton (defaults to the '
                             'lexicographically first file in --joints_dir). All clips in '
                             'one dataset must use the same reference for consistent stats.')
    args = parser.parse_args()

    os.makedirs(args.out_vecs_dir, exist_ok=True)
    os.makedirs(args.out_joints_dir, exist_ok=True)

    if args.reference is None:
        args.reference = os.path.join(
            args.joints_dir, sorted(f for f in os.listdir(args.joints_dir) if f.endswith('.npy'))[0])
        print(f'Using reference skeleton from: {args.reference}')

    example_data = np.load(args.reference)
    example_data = example_data.reshape(len(example_data), -1, 3)[:N_JOINTS]
    tgt_skel = Skeleton(torch.from_numpy(NEW_T2M_RAW_OFFSETS), NEW_T2M_KINEMATIC_CHAIN, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(example_data[0]))

    frame_num, n_done, n_failed = 0, 0, 0
    for file_name in tqdm(sorted(f for f in os.listdir(args.joints_dir) if f.endswith('.npy')),
                          desc='Feature extraction'):
        source_data = np.load(os.path.join(args.joints_dir, file_name))[:, :N_JOINTS]
        if np.any(np.isnan(source_data)):
            print(f'[WARN] NaN detected in {file_name}, skipping', file=sys.stderr)
            n_failed += 1
            continue
        try:
            data, _, _, _ = process_file(source_data, tgt_offsets)
            rec_ric_data = recover_from_ric(
                torch.from_numpy(data).unsqueeze(0).float(), N_JOINTS)
            np.save(os.path.join(args.out_joints_dir, file_name), rec_ric_data.squeeze().numpy())
            np.save(os.path.join(args.out_vecs_dir, file_name), data)
            frame_num += data.shape[0]
            n_done += 1
        except Exception as e:
            print(f'[ERROR] {file_name}: {e}', file=sys.stderr)
            n_failed += 1

    print(f'Done: {n_done} clips ({frame_num} frames, {frame_num / 20 / 60:.2f} min at 20 fps), '
          f'{n_failed} failed')


if __name__ == '__main__':
    main()
