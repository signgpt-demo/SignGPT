import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLosses


class CommitLoss(nn.Module):
    """
    A simple wrapper class for commitment loss.
    This loss is often used in VQ-VAE architectures.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, commit, commit2, **kwargs):
        return commit

def create_mask(length, device):
    b_size = len(length)
    mask = torch.zeros(b_size, max(length), 1).to(device)
    for i, l in enumerate(length):
        mask[i, :l, 0] = 1
    return mask  # [B, maxT, 1]

class SmoothL1LossWithMask(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred, target, length=None):
        if length is not None:
            # mask paddings
            mask = create_mask(length, pred.device)
            while mask.dim() < pred.dim():
                mask = mask.unsqueeze(-1)
            pred = pred * mask
            target = target * mask
        return F.smooth_l1_loss(pred, target)

class GPTLosses(BaseLosses):

    def __init__(self, cfg, stage, num_joints, **kwargs):
        # Save parameters
        self.stage = stage
        recons_loss_type = cfg.LOSS.ABLATION.RECONS_LOSS
        self.nfeats = cfg.DATASET.NFEATS
        print("nfeats in loss for only local", self.nfeats) if self.nfeats==230 else print("nfeats in loss for withvel", self.nfeats)

        # Define losses and their weights
        losses = []
        params = {}
        if stage == "vae":
            losses.extend([
                "recons_feature", "recons_feature_hand", "recons_feature_body",
                "recons_location_hand", "recons_location_body",
                "recons_expression",
                "recons_position", "vq_commit",
                "recons_acceleration",  "recons_hand_angle", "semantic_commit",
            ])
            params.update({
                "recons_feature": cfg.LOSS.LAMBDA_FEATURE,
                'recons_feature_hand': cfg.LOSS.LAMBDA_FEATURE_HAND,
                'recons_feature_body': cfg.LOSS.LAMBDA_FEATURE_BODY,
                'recons_expression': cfg.LOSS.LAMBDA_EXPRESSION,
                'recons_location_hand': cfg.LOSS.LAMBDA_LOCATION_HAND,
                'recons_location_body': cfg.LOSS.LAMBDA_LOCATION_BODY,
                'recons_position': cfg.LOSS.LAMBDA_POS,
                'vq_commit': cfg.LOSS.LAMBDA_COMMIT,
                'recons_acceleration': cfg.LOSS.LAMBDA_ACC,
                'recons_hand_angle': cfg.LOSS.LAMBDA_ANGLE,
                'semantic_commit': cfg.LOSS.LAMBDA_SEMANTIC,
            })

        elif stage in ["lm_pretrain", "lm_instruct"]:
            losses.append("gpt_loss")
            params['gpt_loss'] = cfg.LOSS.LAMBDA_CLS
            losses.append("motion_loss")
            params['motion_loss'] = cfg.LOSS.LAMBDA_MOTION
            losses.append("text_loss")
            params['text_loss'] = cfg.LOSS.LAMBDA_TEXT

        # Define loss functions
        losses_func = {}
        for loss in losses:
            if loss.startswith('recons'):
                if recons_loss_type == "l1":
                    losses_func[loss] = nn.L1Loss
                elif recons_loss_type == "l2":
                    losses_func[loss] = nn.MSELoss
                elif recons_loss_type == "l1_smooth":
                    losses_func[loss] = nn.SmoothL1Loss
            elif loss.split('_')[1] in ['commit', 'loss', 'gpt']:
                losses_func[loss] = CommitLoss
            else:
                raise NotImplementedError(f"Loss {loss} not implemented.")

        super().__init__(cfg, losses, params, losses_func, num_joints, **kwargs)

    def update(self, results_set):
        '''Update and log the losses for the current step.'''
        total_loss: float = 0.0
        vel_start_idx = 4  # Index where local joint features begin
        face_begain = self.nfeats - 10

        if self.stage == "vae":
            if self.nfeats==230:
                pred_features = results_set['m_rst']
                gt_features = results_set['m_ref']

                pred_body_hand = pred_features[...,:face_begain]
                gt_body_hand = gt_features[...,:face_begain]

                # Expression reconstruction loss
                pred_face = pred_features[..., face_begain:]
                gt_face = gt_features[..., face_begain:]
                total_loss += self._update_loss("recons_expression", pred_face, gt_face, results_set['length'])

                # Body and Hand reconstruction loss
                local_features_pred = pred_body_hand[..., vel_start_idx:(self.num_joints - 1) * 3 + vel_start_idx]
                pred_body_features = torch.cat((pred_body_hand[..., :vel_start_idx], local_features_pred[..., :63], local_features_pred[..., 63+90:63+90+33]), dim=-1)
                pred_hand_features = torch.cat((local_features_pred[..., 63:63+90], local_features_pred[..., 63+90+33:]), dim=-1)

                local_features_gt = gt_body_hand[..., vel_start_idx:(self.num_joints - 1) * 3 + vel_start_idx]
                gt_body_features = torch.cat((gt_body_hand[..., :vel_start_idx], local_features_gt[..., :63], local_features_gt[..., 63+90:63+90+33]), dim=-1)
                gt_hand_features = torch.cat((local_features_gt[..., 63:63+90], local_features_gt[..., 63+90+33:]), dim=-1)

                total_loss += self._update_loss("recons_feature_body", pred_body_features, gt_body_features, results_set['length'])
                total_loss += self._update_loss("recons_feature_hand", pred_hand_features, gt_hand_features, results_set['length'])

                # VQ-VAE commitment loss
                total_loss += self._update_loss("vq_commit", results_set['loss_commit'], results_set['loss_commit'])

                # Joint position reconstruction loss for hands
                pred_hand_joints = torch.cat((results_set['joints_rst'][:,:,22:52], results_set['joints_rst'][:,:,63:73]), dim=2)
                gt_hand_joints = torch.cat((results_set['joints_ref'][:,:,22:52], results_set['joints_ref'][:,:,63:73]), dim=2)
                total_loss += self._update_loss("recons_position", pred_hand_joints, gt_hand_joints, results_set['length'])

                # Finger angle reconstruction loss
                total_loss += self.compute_finger_angle_loss(
                    results_set['joints_rst'], results_set['joints_ref'], results_set['length']
                )

                # Joint acceleration loss: penalizes jittery motion via second-order finite difference
                pred_acc = results_set['joints_rst'][:, 2:] - 2 * results_set['joints_rst'][:, 1:-1] + results_set['joints_rst'][:, :-2]
                gt_acc = results_set['joints_ref'][:, 2:] - 2 * results_set['joints_ref'][:, 1:-1] + results_set['joints_ref'][:, :-2]
                total_loss += self._update_loss("recons_acceleration", pred_acc, gt_acc)

                # Semantic loss
                total_loss += self._update_loss("semantic_commit", results_set['loss_semantic'], results_set['loss_semantic'])

                self.total += total_loss.detach()

            elif self.nfeats==449:
                pred_features = results_set['m_rst']
                gt_features = results_set['m_ref']

                # Full feature reconstruction loss
                total_loss += self._update_loss("recons_feature", pred_features, gt_features)

                pred_features_face = pred_features[..., face_begain:]
                gt_features_face = gt_features[..., face_begain:]

                # Expression reconstruction loss
                total_loss += self._update_loss("recons_expression", pred_features_face, gt_features_face, results_set['length'])

                pred_features = pred_features[...,:face_begain]
                gt_features = gt_features[...,:face_begain]

                root_pred = pred_features[..., :vel_start_idx]
                local_features_pred = pred_features[..., vel_start_idx:(self.num_joints - 1) * 3 + vel_start_idx]
                vel_features_pred = pred_features[..., (self.num_joints - 1) * 3 + vel_start_idx:]

                pred_body_features = torch.cat((root_pred, local_features_pred[..., :63], local_features_pred[..., 63+90:63+90+33], vel_features_pred[..., :66], vel_features_pred[..., 66+90:66+90+33]), dim=-1)
                pred_hand_features = torch.cat((local_features_pred[..., 63:63+90], local_features_pred[..., 63+90+33:], vel_features_pred[..., 66:66+90], vel_features_pred[..., 66+90+33:]), dim=-1)

                root_gt = gt_features[..., :vel_start_idx]
                local_features_gt = gt_features[..., vel_start_idx:(self.num_joints - 1) * 3 + vel_start_idx]
                vel_features_gt = gt_features[..., (self.num_joints - 1) * 3 + vel_start_idx:]

                gt_body_features = torch.cat((root_gt, local_features_gt[..., :63], local_features_gt[..., 63+90:63+90+33], vel_features_gt[..., :66], vel_features_gt[..., 66+90:66+90+33]), dim=-1)
                gt_hand_features = torch.cat((local_features_gt[..., 63:63+90], local_features_gt[..., 63+90+33:], vel_features_gt[..., 66:66+90], vel_features_gt[..., 66+90+33:]), dim=-1)

                # Body and Hand feature loss
                total_loss += self._update_loss("recons_feature_body", pred_body_features, gt_body_features, results_set['length'])
                total_loss += self._update_loss("recons_feature_hand", pred_hand_features, gt_hand_features, results_set['length'])

                # Local joint location reconstruction loss
                local_pred = pred_features[..., vel_start_idx:(self.num_joints - 1) * 3 + vel_start_idx]
                local_gt = gt_features[..., vel_start_idx:(self.num_joints - 1) * 3 + vel_start_idx]

                total_loss += self._update_loss("recons_location_hand",
                    torch.cat((local_pred[..., 63:63+90], local_pred[..., 63+90+33:]), dim=-1),
                    torch.cat((local_gt[..., 63:63+90], local_gt[..., 63+90+33:]), dim=-1), results_set['length'])
                total_loss += self._update_loss("recons_location_body",
                    torch.cat((local_pred[..., :63], local_pred[..., 63+90:63+90+33]), dim=-1),
                    torch.cat((local_gt[..., :63], local_gt[..., 63+90:63+90+33]), dim=-1), results_set['length'])

                # VQ-VAE commitment loss
                total_loss += self._update_loss("vq_commit", results_set['loss_commit'], results_set['loss_commit'])

                # Joint position reconstruction loss for hands
                pred_hand_joints = torch.cat((results_set['joints_rst'][:,:,22:52], results_set['joints_rst'][:,:,63:73]), dim=2)
                gt_hand_joints = torch.cat((results_set['joints_ref'][:,:,22:52], results_set['joints_ref'][:,:,63:73]), dim=2)
                total_loss += self._update_loss("recons_position", pred_hand_joints, gt_hand_joints, results_set['length'])

                # Finger angle reconstruction loss
                total_loss += self.compute_finger_angle_loss(
                    results_set['joints_rst'], results_set['joints_ref'], results_set['length']
                )

                # Semantic loss
                total_loss += self._update_loss("semantic_commit", results_set['loss_semantic'], results_set['loss_semantic'])

                self.total += total_loss.detach()

        elif self.stage in ["lm_pretrain", "lm_instruct"]:
            total_loss += self._update_loss("gpt_loss", results_set['outputs'].loss, results_set['outputs'].loss)
            try:
                total_loss += self._update_loss("motion_loss", results_set['outputs'].motion_loss, results_set['outputs'].motion_loss)
                total_loss += self._update_loss("text_loss", results_set['outputs'].text_loss, results_set['outputs'].text_loss)
            except:
                pass
            self.total += total_loss.detach()

        self.count += 1
        return total_loss

    # SMPLH finger kinematic chains: wrist → knuckle1 → knuckle2 → knuckle3 → fingertip
    # Each triplet (parent, joint, child) defines the angle computed at `joint`.
    FINGER_TRIPLETS = [
        # Left hand (wrist=20)
        # Index:  20 → 22 → 23 → 24 → 64(tip)
        (20, 22, 23), (22, 23, 24), (23, 24, 64),
        # Middle: 20 → 25 → 26 → 27 → 65(tip)
        (20, 25, 26), (25, 26, 27), (26, 27, 65),
        # Pinky:  20 → 28 → 29 → 30 → 67(tip)
        (20, 28, 29), (28, 29, 30), (29, 30, 67),
        # Ring:   20 → 31 → 32 → 33 → 66(tip)
        (20, 31, 32), (31, 32, 33), (32, 33, 66),
        # Thumb:  20 → 34 → 35 → 36 → 63(tip)
        (20, 34, 35), (34, 35, 36), (35, 36, 63),
        # Right hand (wrist=21)
        # Index:  21 → 37 → 38 → 39 → 69(tip)
        (21, 37, 38), (37, 38, 39), (38, 39, 69),
        # Middle: 21 → 40 → 41 → 42 → 70(tip)
        (21, 40, 41), (40, 41, 42), (41, 42, 70),
        # Pinky:  21 → 43 → 44 → 45 → 72(tip)
        (21, 43, 44), (43, 44, 45), (44, 45, 72),
        # Ring:   21 → 46 → 47 → 48 → 71(tip)
        (21, 46, 47), (46, 47, 48), (47, 48, 71),
        # Thumb:  21 → 49 → 50 → 51 → 68(tip)
        (21, 49, 50), (49, 50, 51), (50, 51, 68),
    ]

    def compute_finger_angle_loss(self, pred_joints, gt_joints, length=None):
        """
        Computes angle loss at each finger knuckle using the full joint tensor.

        Args:
            pred_joints: [B, T, J, 3] predicted joint positions (full skeleton).
            gt_joints:   [B, T, J, 3] ground truth joint positions.
            length:      list/tensor of valid sequence lengths [B].

        Returns:
            Weighted angle loss scalar.
        """
        pred_angles = self._compute_joint_angles(pred_joints)  # [B, T, 30]
        gt_angles = self._compute_joint_angles(gt_joints)

        if length is not None:
            mask = create_mask(length, pred_angles.device)  # [B, T, 1]
            pred_angles = pred_angles * mask
            gt_angles = gt_angles * mask

        return self._update_loss("recons_hand_angle", pred_angles, gt_angles)

    def _compute_joint_angles(self, joints):
        """
        Compute the angle at each finger knuckle using atan2 for numerical stability.

        Args:
            joints: [B, T, J, 3] joint positions.

        Returns:
            [B, T, 30] angles in radians (3 angles × 5 fingers × 2 hands).
        """
        eps = 1e-7
        angles = []

        for parent_idx, joint_idx, child_idx in self.FINGER_TRIPLETS:
            vec1 = joints[:, :, parent_idx] - joints[:, :, joint_idx]  # [B, T, 3]
            vec2 = joints[:, :, child_idx] - joints[:, :, joint_idx]

            vec1_norm = torch.norm(vec1, p=2, dim=-1, keepdim=True).clamp(min=eps)
            vec2_norm = torch.norm(vec2, p=2, dim=-1, keepdim=True).clamp(min=eps)

            vec1_n = vec1 / vec1_norm
            vec2_n = vec2 / vec2_norm

            cos_val = torch.sum(vec1_n * vec2_n, dim=-1)           # [B, T]
            cross = torch.cross(vec1_n, vec2_n, dim=-1)            # [B, T, 3]
            sin_val = torch.norm(cross, p=2, dim=-1)               # [B, T]

            angle = torch.atan2(sin_val, cos_val)                  # [B, T]
            angles.append(angle)

        return torch.stack(angles, dim=-1)  # [B, T, 30]
