import torch
import rich
import pickle
import numpy as np
from scipy.interpolate import interp1d
import random

def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# padding to max length in one batch
def collate_tensors(batch):
    if isinstance(batch[0], np.ndarray):
        batch = [torch.tensor(b).float() for b in batch]

    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def humanml3d_collate(batch, target_lengths=[196]): # [64, 128, 196]
    notnone_batches = [b for b in batch if b is not None]
    EvalFlag = False if notnone_batches[0][5] is None else True

    # Select one target length for the entire batch.
    batch_target_length = random.choice(target_lengths)

    # Sort by text length
    if EvalFlag:
        notnone_batches.sort(key=lambda x: x[5], reverse=True)

    processed_motions = []
    processed_lengths = []
    processed_motions = [torch.tensor(b[1]).float() for b in notnone_batches]
    processed_lengths = [b[2] for b in notnone_batches]

    # Motion only
    try:
        adapted_batch = {
            "motion": collate_tensors(processed_motions),
            "length": processed_lengths,
            "file_name": [b[8] for b in notnone_batches],
            "GLOSS": [b[9] for b in notnone_batches],
            "GLOSS_emb": [b[7] for b in notnone_batches],
            "GLOSS_mask": [None for b in notnone_batches],
            "target_length": batch_target_length,
        }
    except:
        adapted_batch = {
            "motion": collate_tensors(processed_motions),
            "length": processed_lengths,
            "file_name": [b[8] for b in notnone_batches],
            "target_length": batch_target_length,
        }

    # Text and motion
    if notnone_batches[0][0] is not None:
        try:
            adapted_batch.update({
                "text": [b[0] for b in notnone_batches],
                "file_name": [b[8] for b in notnone_batches],
                "all_captions": [b[7] for b in notnone_batches],
            })
        except:
            adapted_batch.update({
                "text": [b[0] for b in notnone_batches],
                "all_captions": [b[7] for b in notnone_batches],
            })

    # Evaluation related
    if EvalFlag:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "word_embs": collate_tensors([torch.tensor(b[3]).float() for b in notnone_batches]),
            "pos_ohot": collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
            "text_len": collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
            "tokens": [b[6] for b in notnone_batches],
        })

    # Tasks
    if len(notnone_batches[0]) == 9:
        adapted_batch.update({"tasks": [b[8] for b in notnone_batches]})
    if len(notnone_batches[0]) == 10:
        adapted_batch.update({"tasks": [b[9] for b in notnone_batches]})

    return adapted_batch
