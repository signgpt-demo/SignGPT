import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import math
import numpy as np

def safe_normalize(tensor, eps=1e-8, dim=-1):
    """
    Safely normalizes a tensor along a given dimension, avoiding NaN for zero-vectors.

    Args:
        tensor (torch.Tensor): The input tensor to normalize.
        eps (float): A small epsilon value to prevent division by zero.
        dim (int): The dimension along which to normalize.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return tensor / norm


def _compute_dtw_path_from_cost_matrix(cost_matrix):
    """
    Computes the optimal DTW path from a cost matrix using dynamic programming.
    CPU-based implementation using NumPy.

    Args:
        cost_matrix (np.ndarray): 2D numpy array where cost_matrix[i, j] is the
                                  cost of aligning element i (seq1) with element j (seq2).

    Returns:
        list[tuple[int, int]]: A list of (row, col) indices representing the optimal path.
    """
    # Numerical stability: replace non-finite
    if np.isnan(cost_matrix).any() or np.isinf(cost_matrix).any():
        print("Warning: cost_matrix contains non-finite values")
        cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=0.0)

    l1, l2 = cost_matrix.shape

    # Handle empty sequences
    if l1 == 0 or l2 == 0:
        return []

    # 1) Accumulated cost DP
    acc_cost = np.zeros((l1, l2), dtype=cost_matrix.dtype)
    acc_cost[0, 0] = cost_matrix[0, 0]

    for i in range(1, l1):
        acc_cost[i, 0] = cost_matrix[i, 0] + acc_cost[i - 1, 0]
    for j in range(1, l2):
        acc_cost[0, j] = cost_matrix[0, j] + acc_cost[0, j - 1]

    for i in range(1, l1):
        for j in range(1, l2):
            up = acc_cost[i - 1, j]
            left = acc_cost[i, j - 1]
            diag = acc_cost[i - 1, j - 1]
            acc_cost[i, j] = cost_matrix[i, j] + min(up, left, diag)

    # 2) Backtrack with argmin to avoid float equality issues
    path = []
    i, j = l1 - 1, l2 - 1
    path.append((i, j))

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            candidates = (acc_cost[i - 1, j], acc_cost[i, j - 1], acc_cost[i - 1, j - 1])  # up, left, diag
            idx = int(np.argmin(candidates))
            if idx == 2:      # diag
                i, j = i - 1, j - 1
            elif idx == 0:    # up
                i = i - 1
            else:             # left
                j = j - 1
        elif i > 0:  # j == 0
            i = i - 1
        else:        # i == 0
            j = j - 1
        path.append((i, j))

    return path[::-1]


def create_length_mask(max_len, lengths, device):
    """
    Creates a float mask tensor for variable-length sequences.

    Args:
        max_len (int): The maximum length of the sequences (e.g., tensor.size(1)).
        lengths (torch.Tensor): A 1D tensor of integers with the true length of each sequence.
        device: The device to create the tensor on.

    Returns:
        torch.Tensor: A float mask of shape [batch_size, max_len], with 1=valid, 0=padding.
    """
    batch_size = len(lengths)
    ar = torch.arange(max_len, device=device).expand(batch_size, max_len)
    mask = ar < lengths.unsqueeze(1)
    return mask.float()


def masked_mean(tensor, mask):
    """
    Safely computes the mean of a tensor over a dimension, ignoring masked values.

    Args:
        tensor (torch.Tensor): The input tensor, e.g., of shape [B, T, D].
        mask (torch.Tensor): A float mask tensor, e.g., of shape [B, T], with 1=valid, 0=padding.

    Returns:
        torch.Tensor: The masked mean tensor of shape [B, D].
    """
    masked_tensor = tensor * mask.unsqueeze(-1)
    sum_values = masked_tensor.sum(dim=1)
    count_values = mask.sum(dim=1, keepdim=True)
    count_values = torch.clamp(count_values, min=1e-8)
    result = sum_values / count_values
    return result


def compute_local_alignment_loss(motion_emb, text_emb, motion_lengths, text_mask):
    """
    Computes a local alignment loss between motion and text sequences.
    Uses DTW to get optimal alignment path, then applies local InfoNCE on aligned pairs.

    Args:
        motion_emb (torch.Tensor): [B, T_m, D]
        text_emb (torch.Tensor):   [B, T_t, D]
        motion_lengths (torch.Tensor): [B] true length of motion sequences (int)
        text_mask (torch.Tensor):      [B, T_t] float/bool mask with 1=valid, 0=padding

    Returns:
        torch.Tensor: scalar loss tensor (device and dtype follow motion_emb).
    """
    device = motion_emb.device
    dtype = motion_emb.dtype

    # Handle empty sequences dimension-wise quickly
    if motion_emb.size(1) == 0 or text_emb.size(1) == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    assert text_mask is not None, "text_mask must be provided and follow 1=valid, 0=padding semantics."

    # 1) Normalize features
    motion_emb_norm = safe_normalize(motion_emb, eps=1e-8, dim=-1)
    text_emb_norm = safe_normalize(text_emb, eps=1e-8, dim=-1)

    # 2) Similarity and cost
    similarity_matrix = torch.matmul(motion_emb_norm, text_emb_norm.transpose(1, 2))  # [B, T_m, T_t]
    cost_matrix = 1 - similarity_matrix

    # 3) DTW paths (CPU, numpy)
    with torch.no_grad():
        alignment_paths = []
        motion_lengths_cpu = motion_lengths.detach().cpu().to(torch.long)
        # Ensure text_mask is float with 1=valid
        text_mask_float = text_mask.detach().float()
        for i in range(cost_matrix.shape[0]):
            valid_motion_len = int(max(1, motion_lengths_cpu[i].item()))
            valid_motion_len = min(valid_motion_len, motion_emb.size(1))

            valid_text_len = int(max(1, text_mask_float[i].sum().item()))
            valid_text_len = min(valid_text_len, text_emb.size(1))

            if valid_motion_len == 0 or valid_text_len == 0:
                alignment_paths.append(torch.empty(0, 2, dtype=torch.long, device=device))
                continue

            valid_cost_matrix_np = cost_matrix[i, :valid_motion_len, :valid_text_len].detach().cpu().numpy()
            path_list = _compute_dtw_path_from_cost_matrix(valid_cost_matrix_np)

            if not path_list:
                alignment_paths.append(torch.empty(0, 2, dtype=torch.long, device=device))
            else:
                alignment_paths.append(torch.tensor(path_list, dtype=torch.long, device=device))

    # 4) Local InfoNCE with padding-masked negatives
    temperature = 0.1

    valid_paths_with_indices = [(i, path) for i, path in enumerate(alignment_paths) if path.numel() > 0]
    if not valid_paths_with_indices:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Flatten paths
    batch_map = torch.cat([torch.full_like(path[:, 0], fill_value=i) for i, path in valid_paths_with_indices])  # [N]
    m_indices = torch.cat([path[:, 0] for _, path in valid_paths_with_indices])  # [N]
    t_indices = torch.cat([path[:, 1] for _, path in valid_paths_with_indices])  # [N]

    # Bound checks
    max_motion_len = motion_emb.size(1)
    max_text_len = text_emb.size(1)
    batch_size = similarity_matrix.size(0)

    m_indices = torch.clamp(m_indices, 0, max_motion_len - 1)
    t_indices = torch.clamp(t_indices, 0, max_text_len - 1)
    batch_map = torch.clamp(batch_map, 0, batch_size - 1)

    # Build valid lengths per sample for masking logits
    motion_valid_len = torch.clamp(motion_lengths.to(device=device, dtype=torch.long), min=1, max=max_motion_len)
    text_valid_len = torch.clamp(text_mask_float.sum(dim=1).to(device=device, dtype=torch.long), min=1, max=max_text_len)

    # Create -inf masks for negatives at padding positions (per pair)
    # For m2t: logits shape [N, T_t]
    m2t_masks = []
    for bi in batch_map.tolist():
        L = int(text_valid_len[bi].item())
        # 0 for valid, -inf for padding
        mask = torch.zeros(max_text_len, device=device, dtype=dtype)
        if L < max_text_len:
            mask[L:] = float("-inf")
        m2t_masks.append(mask.unsqueeze(0))
    m2t_mask_logits = torch.cat(m2t_masks, dim=0)  # [N, T_t]

    # For t2m: logits shape [N, T_m]
    t2m_masks = []
    for bi in batch_map.tolist():
        L = int(motion_valid_len[bi].item())
        mask = torch.zeros(max_motion_len, device=device, dtype=dtype)
        if L < max_motion_len:
            mask[L:] = float("-inf")
        t2m_masks.append(mask.unsqueeze(0))
    t2m_mask_logits = torch.cat(t2m_masks, dim=0)  # [N, T_m]

    # Motion-to-Text logits over all text tokens of the same sample
    sim_m2t = similarity_matrix[batch_map, m_indices] / temperature  # [N, T_t]
    sim_m2t = sim_m2t + m2t_mask_logits
    loss_m2t = F.cross_entropy(sim_m2t, t_indices)

    # Text-to-Motion logits over all motion frames of the same sample
    sim_t2m = similarity_matrix.permute(0, 2, 1)[batch_map, t_indices] / temperature  # [N, T_m]
    sim_t2m = sim_t2m + t2m_mask_logits
    loss_t2m = F.cross_entropy(sim_t2m, m_indices)

    final_loss = (loss_m2t + loss_t2m) / 2
    return final_loss


def compute_semantic_loss_length(motion_emb, text_emb, motion_lengths=None, text_mask=None):
    """
    Computes a global semantic alignment loss (CLIP-style contrastive loss)
    between motion and text sequences, supporting variable lengths.

    Args:
        motion_emb (torch.Tensor): [B, T_m, D]
        text_emb (torch.Tensor):   [B, T_t, D]
        motion_lengths (torch.Tensor, optional): [B] true lengths of motion sequences.
        text_mask (torch.Tensor, optional): [B, T_t] 1=valid, 0=padding.

    Returns:
        torch.Tensor: scalar loss tensor.
    """
    device = motion_emb.device
    dtype = motion_emb.dtype

    # Global embeddings (masked mean if lengths/mask provided)
    if motion_lengths is not None:
        motion_mask = create_length_mask(motion_emb.size(1), motion_lengths.to(torch.long), device=motion_emb.device)
        motion_global = masked_mean(motion_emb, motion_mask)
    else:
        motion_global = motion_emb.mean(dim=1)

    if text_mask is not None:
        text_global = masked_mean(text_emb, text_mask.float())
    else:
        text_global = text_emb.mean(dim=1)

    # Normalize globals
    motion_global = safe_normalize(motion_global, eps=1e-8, dim=-1)
    text_global = safe_normalize(text_global, eps=1e-8, dim=-1)

    # Contrastive logits
    temperature = 0.07
    logits = motion_global @ text_global.t() / temperature  # [B, B]
    labels = torch.arange(logits.shape[0], device=device)

    loss_motion = F.cross_entropy(logits, labels)
    loss_text = F.cross_entropy(logits.t(), labels)

    final_loss = (loss_motion + loss_text) / 2
    return final_loss


def compute_cosine_semantic_loss(motion_emb, text_global):
    """
    Per-sample cosine similarity loss between motion and text embeddings.
    O(BD) complexity, no negative sampling needed.

    Args:
        motion_emb:  [B, T_m, D] motion latent features (will be mean-pooled).
        text_global: [B, D] pre-pooled text embeddings (masked mean already applied).

    Returns:
        Scalar loss: mean(1 - cosine_similarity).
    """
    motion_global = F.normalize(motion_emb.mean(dim=1), dim=-1)
    text_global = F.normalize(text_global, dim=-1)
    cos_sim = (motion_global * text_global).sum(dim=-1)  # [B]
    return (1.0 - cos_sim).mean()