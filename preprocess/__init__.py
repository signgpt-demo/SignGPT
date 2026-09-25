"""Data preprocessing pipeline for SignGPT.

Converts raw SMPL-X pose estimates and text annotations into the
HumanML3D-style 73-joint representation expected by the training code:

    raw frame-level pose pkls
        -> step0: one merged pkl per clip
        -> step1: SMPL-X forward kinematics -> [T, 73, 3] joints
        -> step2: skeleton normalization + HumanML3D features -> [T-1, 220]
        -> step3: concatenate expression parameters       -> [T-1, 230]
        -> step4: per-dimension mean/std statistics
        -> step5: text files, split files, and prompt templates

Each step is a standalone CLI script; see ``preprocess/README.md``.
"""
