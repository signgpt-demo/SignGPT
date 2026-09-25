from typing import List
import torch
from torch import Tensor
from torchmetrics import Metric
import numpy as np
from .utils import *

class MRMetrics(Metric):
    def __init__(self,
                 njoints,
                 jointstype: str = "mmm",
                 force_in_meter: bool = True,
                 align_root: bool = True,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = 'Motion Reconstructions'
        self.jointstype = jointstype
        self.align_root = align_root
        self.force_in_meter = force_in_meter

        # Distributed metric states.
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("MPJPE", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("PAMPJPE", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("ACCEL", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("Hand_DTW", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("Avg_DTW", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("Body_DTW", default=torch.tensor([0.0]), dist_reduce_fx="sum")

        # Metrics reported by this evaluator.
        self.MR_metrics = ["MPJPE", "PAMPJPE", "ACCEL", "DTW"]
        self.metrics = self.MR_metrics

    def compute(self, sanity_flag):
        if self.force_in_meter:
            factor = 1000.0  # Convert meters to millimeters.
        else:
            factor = 1.0

        count = self.count
        count_seq = self.count_seq

        # Aggregate all reconstruction metrics.
        mr_metrics = {
            "MPJPE": self.MPJPE / count * factor,
            "PAMPJPE": self.PAMPJPE / count * factor,
            "ACCEL": self.ACCEL / (count - 2 * count_seq) * factor,
            "Hand_DTW": self.Hand_DTW / count_seq,
            "Body_DTW": self.Body_DTW / count_seq,
            "AvgDTW": self.Avg_DTW / count_seq
        }

        self.reset()
        return mr_metrics


    def update(self, joints_rst: Tensor, joints_ref: Tensor, lengths: List[int]):
        assert joints_rst.shape == joints_ref.shape
        assert joints_rst.dim() == 4 # (bs, seq, njoint=22, 3)

        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # Compute DTW on CPU to avoid repeated CUDA synchronization.
        rst = joints_rst.detach().cpu().numpy()
        ref = joints_ref.detach().cpu().numpy()

        # Root-alignment indices.
        align_inds = [0] if (self.align_root and self.jointstype in ['mmm', 'humanml3d']) else None

        for i in range(len(lengths)):
            seq_len = lengths[i]

            rst_seq = rst[i, :seq_len]  # (seq_len, njoint, 3)
            ref_seq = ref[i, :seq_len]  # (seq_len, njoint, 3)

            # Compute reconstruction metrics.
            self.MPJPE += torch.sum(calc_mpjpe(
                torch.from_numpy(rst_seq),
                torch.from_numpy(ref_seq),
                align_inds=align_inds
            ))
            try:
                self.PAMPJPE += torch.sum(calc_pampjpe(
                    torch.from_numpy(rst_seq),
                    torch.from_numpy(ref_seq)
                ))
            except:
                pass
            try:
                self.ACCEL += torch.sum(calc_accel(
                    torch.from_numpy(rst_seq),
                    torch.from_numpy(ref_seq)
                ))
            except:
                pass

            try:
                body_rst = np.concatenate((rst_seq[:, 0:22], rst_seq[:, 52:-10]), axis=1)
                body_ref = np.concatenate((ref_seq[:, 0:22], ref_seq[:, 52:-10]), axis=1)
                self.Body_DTW += evaluate_motion(body_rst, body_ref)

                hand_rst = np.concatenate((rst_seq[:, 22:52], rst_seq[:, -10:]), axis=1)
                hand_ref = np.concatenate((ref_seq[:, 22:52], ref_seq[:, -10:]), axis=1)
                self.Hand_DTW += evaluate_motion(hand_rst, hand_ref)

                self.Avg_DTW += evaluate_motion(rst_seq, ref_seq)
            except Exception:
                pass



def evaluate_motion(pred_motions, ref_motions, joint_weights=None):
    """
    DTW-based motion evaluation.
    Args:
        pred_motions: [T_pred, num_joints, 3] numpy array
        ref_motions:  [T_ref,  num_joints, 3] numpy array
    Returns:
        normalized DTW-JPE distance (scalar)
    """
    if isinstance(pred_motions, torch.Tensor):
        pred_motions = pred_motions.numpy()
    if isinstance(ref_motions, torch.Tensor):
        ref_motions = ref_motions.numpy()

    dtw_dist, path = dtw_jpe(pred_motions, ref_motions)
    return dtw_dist


def _compute_frame_jpe_matrix(pred, ref):
    """
    Compute pairwise per-frame mean JPE cost matrix.
    pred: [T1, J, 3], ref: [T2, J, 3]
    Returns: [T1, T2] cost matrix
    """
    diff = pred[:, np.newaxis, :, :] - ref[np.newaxis, :, :, :]  # [T1, T2, J, 3]
    per_joint_dist = np.linalg.norm(diff, axis=-1)                # [T1, T2, J]
    return np.mean(per_joint_dist, axis=-1)                       # [T1, T2]


def dtw_jpe(motion_pred, motion_ref):
    """
    Full DTW using per-frame mean JPE as the local cost.
    Replaces fastdtw which has known backtracking bugs.
    O(T1*T2) — fine for motion sequences (typically < 300 frames).
    """
    if isinstance(motion_pred, torch.Tensor):
        motion_pred = motion_pred.numpy()
    if isinstance(motion_ref, torch.Tensor):
        motion_ref = motion_ref.numpy()

    T1, T2 = motion_pred.shape[0], motion_ref.shape[0]

    cost_matrix = _compute_frame_jpe_matrix(motion_pred, motion_ref)

    # DP accumulation
    D = np.full((T1 + 1, T2 + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            D[i, j] = cost_matrix[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # Backtrack
    i, j = T1, T2
    path = []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]
            argmin = int(np.argmin(candidates))
            if argmin == 0:
                i, j = i - 1, j - 1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1
    path.reverse()

    normalized_distance = D[T1, T2] / len(path)
    return normalized_distance, path
