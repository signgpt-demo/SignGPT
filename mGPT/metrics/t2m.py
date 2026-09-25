from typing import List

import numpy as np
import torch
from fastdtw import fastdtw
from omegaconf import DictConfig
from torchmetrics import Metric


class TM2TMetrics(Metric):
    full_state_update = True

    def __init__(self,
                 cfg: DictConfig,
                 dataname: str = 'humanml3d',
                 top_k: int = 3,
                 R_size: int = 32,
                 diversity_times: int = 300,
                 dist_sync_on_step: bool = True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname.lower()
        self.name = "matching, fid, diversity and dtw scores"
        self.top_k = top_k
        self.R_size = R_size
        self.text = 'lm' in cfg.TRAIN.STAGE and cfg.model.params.task == 't2m'
        self.diversity_times = diversity_times
        self.add_state("count",  default=torch.tensor(0),  dist_reduce_fx="sum")
        self.add_state("count_seq",  default=torch.tensor(0),  dist_reduce_fx="sum")

        self.back_translation_feats_ref = []
        self.back_translation_feats_rst = []
        self.back_translation_gt_texts = []

        self.metrics = []

        if self.text:
            self.add_state("Matching_score",  default=torch.tensor(0.0),  dist_reduce_fx="sum")
            self.add_state("gt_Matching_score",  default=torch.tensor(0.0),  dist_reduce_fx="sum")

            self.Matching_metrics = ["Matching_score", "gt_Matching_score"]
            for k in range(1, top_k + 1):
                self.add_state(f"R_precision_top_{str(k)}",  default=torch.tensor(0.0),  dist_reduce_fx="sum")
                self.Matching_metrics.append(f"R_precision_top_{str(k)}")
            for k in range(1, top_k + 1):
                self.add_state(f"gt_R_precision_top_{str(k)}",  default=torch.tensor(0.0),  dist_reduce_fx="sum")
                self.Matching_metrics.append(f"gt_R_precision_top_{str(k)}")

            self.metrics.extend(self.Matching_metrics)

        self.add_state("FID",  default=torch.tensor(0.0),  dist_reduce_fx="sum")
        self.metrics.append("FID")

        self.add_state("Diversity",  default=torch.tensor(0.0),  dist_reduce_fx="sum")
        self.add_state("gt_Diversity",  default=torch.tensor(0.0),  dist_reduce_fx="sum")
        self.metrics.extend(["Diversity",  "gt_Diversity"])

        self.add_state("DTW_avg",  default=torch.tensor(0.0),  dist_reduce_fx="sum")
        self.metrics.append("DTW_avg")
        self.add_state("DTW_body",  default=torch.tensor(0.0),  dist_reduce_fx="sum")
        self.metrics.append("DTW_body")
        self.add_state("DTW_hand",  default=torch.tensor(0.0),  dist_reduce_fx="sum")
        self.metrics.append("DTW_hand")

        self.add_state("text_embeddings",  default=[], dist_reduce_fx=None)
        self.add_state("recmotion_embeddings",  default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings",  default=[], dist_reduce_fx=None)

        # Feature caches for FID and diversity.
        self.add_state("feats_rst_list",  default=[], dist_reduce_fx=None)
        self.add_state("feats_ref_list",  default=[], dist_reduce_fx=None)

        # Joint-position caches for DTW-JPE.
        self.add_state("joint_rst",  default=[], dist_reduce_fx=None)
        self.add_state("joint_ref",  default=[], dist_reduce_fx=None)
        self.add_state("motion_lengths",  default=[], dist_reduce_fx=None)

    @torch.no_grad()
    def compute(self, sanity_flag: bool = False):

        count_seq = self.count_seq.item()

        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        if sanity_flag:
            return metrics

        if len(self.feats_rst_list) > 0 and len(self.feats_ref_list) > 0:
            try:
                from mGPT.metrics.utils import calculate_frechet_distance_np, calculate_activation_statistics_np
                feats_rst_np = []
                feats_ref_np = []

                for feat in self.feats_rst_list:
                    if isinstance(feat, torch.Tensor):
                        feats_rst_np.append(feat.cpu().numpy().reshape(1, -1))
                    else:
                        feats_rst_np.append(np.array(feat).reshape(1, -1))

                for feat in self.feats_ref_list:
                    if isinstance(feat, torch.Tensor):
                        feats_ref_np.append(feat.cpu().numpy().reshape(1, -1))
                    else:
                        feats_ref_np.append(np.array(feat).reshape(1, -1))

                feats_rst_all = np.concatenate(feats_rst_np, axis=0)
                feats_ref_all = np.concatenate(feats_ref_np, axis=0)

                mu_rst, sigma_rst = calculate_activation_statistics_np(feats_rst_all)
                mu_ref, sigma_ref = calculate_activation_statistics_np(feats_ref_all)
                fid_score = calculate_frechet_distance_np(mu_rst, sigma_rst, mu_ref, sigma_ref)
                metrics["FID"] = torch.tensor(fid_score)

                from mGPT.metrics.utils import calculate_diversity_np
                div_score = calculate_diversity_np(feats_rst_all, min(self.diversity_times, len(feats_rst_all) - 1))
                metrics["Diversity"] = torch.tensor(div_score)

                gt_div_score = calculate_diversity_np(feats_ref_all, min(self.diversity_times, len(feats_ref_all) - 1))
                metrics["gt_Diversity"] = torch.tensor(gt_div_score)

            except Exception as e:
                print(f"Error computing FID/Diversity: {e}")
                import traceback
                traceback.print_exc()
                metrics["FID"] = torch.tensor(0.0)
                metrics["Diversity"] = torch.tensor(0.0)
                metrics["gt_Diversity"] = torch.tensor(0.0)
        else:
            metrics["FID"] = torch.tensor(0.0)
            metrics["Diversity"] = torch.tensor(0.0)
            metrics["gt_Diversity"] = torch.tensor(0.0)

        try:
            if len(self.joint_rst) > 0 and len(self.joint_ref) > 0:
                dtw_jpe_score = self._calculate_dtw_jpe_scores_avg()
                metrics["DTW_avg"] = torch.tensor(dtw_jpe_score / count_seq if count_seq > 0 else 0)
                dtw_jpe_score = self._calculate_dtw_jpe_scores_body()
                metrics["DTW_body"] = torch.tensor(dtw_jpe_score / count_seq if count_seq > 0 else 0)
                dtw_jpe_score = self._calculate_dtw_jpe_scores_hand()
                metrics["DTW_hand"] = torch.tensor(dtw_jpe_score / count_seq if count_seq > 0 else 0)
            else:
                metrics["DTW_avg"] = torch.tensor(0.0)
                metrics["DTW_body"] = torch.tensor(0.0)
                metrics["DTW_hand"] = torch.tensor(0.0)
        except Exception as e:
            print(f"Error computing DTW scores: {e}")
            import traceback
            traceback.print_exc()
            metrics["DTW_avg"] = torch.tensor(0.0)
            metrics["DTW_body"] = torch.tensor(0.0)
            metrics["DTW_hand"] = torch.tensor(0.0)

        self.reset()
        return {**metrics}

    @torch.no_grad()
    def update(self,
               feats_ref: torch.Tensor,
               feats_rst: torch.Tensor,
               lengths_ref: List[int],
               lengths_rst: List[int],
               word_embs: torch.Tensor = None,
               pos_ohot: torch.Tensor = None,
               text_lengths: torch.Tensor = None,
               gt_texts: List[str] = None,
               joints_rst: torch.Tensor = None,
               joints_ref:  torch.Tensor = None
               ):

        self.count  += sum(lengths_ref)
        self.count_seq  += len(lengths_ref)

        if feats_rst is not None and feats_ref is not None:
            for i in range(len(lengths_ref)):
                feat_rst = feats_rst[i, :lengths_ref[i], :].reshape(-1)
                feat_ref = feats_ref[i, :lengths_ref[i], :].reshape(-1)
                self.feats_rst_list.append(feat_rst)
                self.feats_ref_list.append(feat_ref)

        if joints_rst is not None and joints_ref is not None:
            for i in range(joints_rst.shape[0]):
                self.joint_rst.append(joints_rst[i])  # (seq_len, num_joints, 3)
                self.joint_ref.append(joints_ref[i])  # (seq_len, num_joints, 3)
                self.motion_lengths.append(torch.tensor(lengths_ref[i]))

    def _calculate_dtw_jpe_scores_avg(self):
        """Calculate aggregate DTW-JPE over all joints."""

        total_dtw_jpe = 0.0

        all_rec_joints = self.joint_rst
        all_gt_joints = self.joint_ref
        all_lengths = self.motion_lengths

        num_samples = min(len(all_rec_joints), len(all_gt_joints), len(all_lengths))

        if num_samples == 0:
            print("[DEBUG] No samples for DTW avg calculation")
            return 0.0

        for i in range(num_samples):
            try:
                rec_joints = all_rec_joints[i]  # (seq_len, num_joints, 3)
                gt_joints = all_gt_joints[i]    # (seq_len, num_joints, 3)

                length = min(int(all_lengths[i]), rec_joints.shape[0], gt_joints.shape[0])

                rec_seq = rec_joints[:length].cpu().numpy()  # (length, num_joints, 3)
                gt_seq = gt_joints[:length].cpu().numpy()    # (length, num_joints, 3)

                dtw_jpe_score = evaluate_motion(rec_seq, gt_seq)
                total_dtw_jpe += dtw_jpe_score
            except Exception as e:
                print(f"[DEBUG] Error in DTW avg calculation for sample {i}: {e}")
                continue

        return float(total_dtw_jpe)

    def _calculate_dtw_jpe_scores_body(self):
        """Calculate aggregate DTW-JPE over body and face joints."""

        total_dtw_jpe = 0.0

        all_rec_joints = self.joint_rst
        all_gt_joints = self.joint_ref
        all_lengths = self.motion_lengths

        num_samples = min(len(all_rec_joints), len(all_gt_joints), len(all_lengths))

        if num_samples == 0:
            return 0.0

        for i in range(num_samples):
            try:
                rec_joints = torch.cat((all_rec_joints[i][:,:22,:],all_rec_joints[i][:,52:63,:]),dim=-2)  # (seq_len, num_joints, 3)
                gt_joints = torch.cat((all_gt_joints[i][:,:22,:],all_gt_joints[i][:,52:63,:]),dim=-2)    # (seq_len, num_joints, 3)
                length = min(int(all_lengths[i]), rec_joints.shape[0], gt_joints.shape[0])

                rec_seq = rec_joints[:length].cpu().numpy()  # (length, num_joints, 3)
                gt_seq = gt_joints[:length].cpu().numpy()    # (length, num_joints, 3)

                dtw_jpe_score = evaluate_motion(rec_seq, gt_seq)
                total_dtw_jpe += dtw_jpe_score

            except Exception as e:
                print(f"Error calculating DTW-JPE body for sequence {i}: {e}")
                continue

        return float(total_dtw_jpe)

    def _calculate_dtw_jpe_scores_hand(self):
        """Calculate aggregate DTW-JPE over hand joints."""

        total_dtw_jpe = 0.0

        all_rec_joints = self.joint_rst
        all_gt_joints = self.joint_ref
        all_lengths = self.motion_lengths

        num_samples = min(len(all_rec_joints), len(all_gt_joints), len(all_lengths))

        if num_samples == 0:
            return 0.0

        for i in range(num_samples):
            try:
                rec_joints = torch.cat((all_rec_joints[i][:,22:52],all_rec_joints[i][:,63:73]),dim=-2)
                gt_joints = torch.cat((all_gt_joints[i][:,22:52],all_gt_joints[i][:,63:73]),dim=-2)    # (seq_len, num_joints, 3)
                length = min(int(all_lengths[i]), rec_joints.shape[0], gt_joints.shape[0])

                rec_seq = rec_joints[:length].cpu().numpy()  # (length, num_joints, 3)
                gt_seq = gt_joints[:length].cpu().numpy()    # (length, num_joints, 3)

                dtw_jpe_score = evaluate_motion(rec_seq, gt_seq)
                total_dtw_jpe += dtw_jpe_score

            except Exception as e:
                print(f"Error calculating DTW-JPE hand for sequence {i}: {e}")
                continue

        return float(total_dtw_jpe)

def evaluate_motion(pred_motions, ref_motions, joint_weights=None):
    """Evaluate a predicted motion against a reference with DTW-JPE."""
    pred = pred_motions
    ref = ref_motions
    dtw_dist, paths = dtw_jpe(pred, ref)

    return dtw_dist

def dtw_jpe(motion_pred, motion_ref):
    """Return normalized DTW-JPE and its alignment path."""
    T_pred, num_joints, _ = motion_pred.shape
    T_ref, num_joints_ref, _ = motion_ref.shape

    assert num_joints == num_joints_ref, "Joint counts do not match"

    def frame_jpe_distance(frame1, frame2):
        """Calculate mean joint-position error between two frames."""
        return np.mean(np.linalg.norm(frame1 - frame2, axis=1))

    distance, path = fastdtw(motion_pred, motion_ref, dist=frame_jpe_distance)

    normalized_distance = distance / len(path)

    return normalized_distance, path
