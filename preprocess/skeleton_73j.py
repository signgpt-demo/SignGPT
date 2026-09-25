"""Topology constants for the 73-joint sign-language skeleton.

The skeleton is derived from the 144-joint SMPL-X output: the 68 detailed
face landmarks are dropped, together with the jaw and the two eyeball
joints, leaving 73 joints (22 body, 30 finger, 5 face keypoints, 6 feet,
10 fingertips).

Joint order (73 joints, SMPL-X indices with jaw/eyes removed):

    0-21   body (pelvis, spine, head, limbs)
    22-51  left/right finger joints (index/middle/pinky/ring/thumb x 1-3)
    52-56  face keypoints (nose, right/left eye, right/left ear)
    57-62  feet (big/small toe, heel x left/right)
    63-72  fingertips

Index map from the 76-joint SMPL-H layout: keep 0..75, drop 22 (jaw),
23 (left eye), 24 (right eye), i.e. ``JOINT_IDX_73`` below.
"""

import numpy as np

# SMPL-X output joint indices selected for the 73-joint skeleton.
JOINT_IDX_73 = tuple(sorted(set(range(76)) - {22, 23, 24}))

# Face joints used to estimate the body forward direction:
# (right_hip, left_hip, right_shoulder, left_shoulder) in 73-joint indexing.
FACE_JOINT_INDX = [2, 1, 17, 16]

N_JOINTS = 73

# Canonical rest-pose offsets (unit directions; magnitudes are fitted per
# subject from the first frame by Skeleton.get_offsets_joints).
NEW_T2M_RAW_OFFSETS = np.array([
    # 0-21: body
    [0.0, 0.0, 0.0],     # 0  pelvis (root)
    [1.0, 0.0, 0.0],     # 1  left_hip
    [-1.0, 0.0, 0.0],    # 2  right_hip
    [0.0, 1.0, 0.0],     # 3  spine1
    [0.0, -1.0, 0.0],    # 4  left_knee
    [0.0, -1.0, 0.0],    # 5  right_knee
    [0.0, 1.0, 0.0],     # 6  spine2
    [0.0, -1.0, 0.0],    # 7  left_ankle
    [0.0, -1.0, 0.0],    # 8  right_ankle
    [0.0, 1.0, 0.0],     # 9  spine3
    [0.0, 0.0, 1.0],     # 10 left_foot
    [0.0, 0.0, 1.0],     # 11 right_foot
    [0.0, 1.0, 0.0],     # 12 neck
    [1.0, 0.0, 0.0],     # 13 left_collar
    [-1.0, 0.0, 0.0],    # 14 right_collar
    [0.0, 1.0, 0.0],     # 15 head
    [0.0, -1.0, 0.0],    # 16 left_shoulder
    [0.0, -1.0, 0.0],    # 17 right_shoulder
    [0.0, -1.0, 0.0],    # 18 left_elbow
    [0.0, -1.0, 0.0],    # 19 right_elbow
    [0.0, -1.0, 0.0],    # 20 left_wrist
    [0.0, -1.0, 0.0],    # 21 right_wrist

    # 22-36: left hand (index/middle/pinky/ring/thumb x 1-3)
    [0.0, 0.0, -1.0],    # 22 left_index1
    [0.0, 0.0, -1.0],    # 23 left_index2
    [0.0, 0.0, -1.0],    # 24 left_index3
    [0.0, 0.0, -1.0],    # 25 left_middle1
    [0.0, 0.0, -1.0],    # 26 left_middle2
    [0.0, 0.0, -1.0],    # 27 left_middle3
    [0.0, 0.0, -1.0],    # 28 left_pinky1
    [0.0, 0.0, -1.0],    # 29 left_pinky2
    [0.0, 0.0, -1.0],    # 30 left_pinky3
    [0.0, 0.0, -1.0],    # 31 left_ring1
    [0.0, 0.0, -1.0],    # 32 left_ring2
    [0.0, 0.0, -1.0],    # 33 left_ring3
    [0.0, -1.0, 0.0],    # 34 left_thumb1
    [0.0, -1.0, 0.0],    # 35 left_thumb2
    [0.0, -1.0, 0.0],    # 36 left_thumb3

    # 37-51: right hand
    [0.0, 0.0, -1.0],    # 37 right_index1
    [0.0, 0.0, -1.0],    # 38 right_index2
    [0.0, 0.0, -1.0],    # 39 right_index3
    [0.0, 0.0, -1.0],    # 40 right_middle1
    [0.0, 0.0, -1.0],    # 41 right_middle2
    [0.0, 0.0, -1.0],    # 42 right_middle3
    [0.0, 0.0, -1.0],    # 43 right_pinky1
    [0.0, 0.0, -1.0],    # 44 right_pinky2
    [0.0, 0.0, -1.0],    # 45 right_pinky3
    [0.0, 0.0, -1.0],    # 46 right_ring1
    [0.0, 0.0, -1.0],    # 47 right_ring2
    [0.0, 0.0, -1.0],    # 48 right_ring3
    [0.0, -1.0, 0.0],    # 49 right_thumb1
    [0.0, -1.0, 0.0],    # 50 right_thumb2
    [0.0, -1.0, 0.0],    # 51 right_thumb3

    # 52-56: face keypoints
    [0.0, 0.0, 1.0],     # 52 nose (forward)
    [1.0, 0.0, 0.0],     # 53 right_eye
    [-1.0, 0.0, 0.0],    # 54 left_eye
    [1.0, 0.0, 0.0],     # 55 right_ear
    [-1.0, 0.0, 0.0],    # 56 left_ear

    # 57-62: feet
    [0.0, 0.0, 1.0],     # 57 left_big_toe
    [0.0, 0.0, 0.5],     # 58 left_small_toe
    [0.0, -1.0, 0.0],    # 59 left_heel
    [0.0, 0.0, 1.0],     # 60 right_big_toe
    [0.0, 0.0, 0.5],     # 61 right_small_toe
    [0.0, -1.0, 0.0],    # 62 right_heel

    # 63-72: fingertips
    [0.0, -1.0, 0.0],    # 63 left_thumb
    [0.0, 0.0, -1.0],    # 64 left_index
    [0.0, 0.0, -1.0],    # 65 left_middle
    [0.0, 0.0, -1.0],    # 66 left_ring
    [0.0, 0.0, -1.0],    # 67 left_pinky
    [0.0, -1.0, 0.0],    # 68 right_thumb
    [0.0, 0.0, -1.0],    # 69 right_index
    [0.0, 0.0, -1.0],    # 70 right_middle
    [0.0, 0.0, -1.0],    # 71 right_ring
    [0.0, 0.0, -1.0],    # 72 right_pinky
], dtype=np.float32)

NEW_T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 62, 11],       # right leg: pelvis->r_hip->r_knee->r_ankle->r_heel->r_foot
    [0, 1, 4, 7, 59, 10],       # left leg
    [0, 3, 6, 9, 12, 15],       # spine: pelvis->spine1->spine2->spine3->neck->head
    [9, 14, 17, 19, 21],        # right arm: spine3->r_collar->r_shoulder->r_elbow->r_wrist
    [9, 13, 16, 18, 20],        # left arm

    # right fingers
    [21, 37, 38, 39, 69],       # right index
    [21, 40, 41, 42, 70],       # right middle
    [21, 43, 44, 45, 72],       # right pinky
    [21, 46, 47, 48, 71],       # right ring
    [21, 49, 50, 51, 68],       # right thumb

    # left fingers
    [20, 22, 23, 24, 64],       # left index
    [20, 25, 26, 27, 65],       # left middle
    [20, 28, 29, 30, 67],       # left pinky
    [20, 31, 32, 33, 66],       # left ring
    [20, 34, 35, 36, 63],       # left thumb

    # face keypoints
    [15, 52],                   # head->nose
    [52, 53],                   # nose->right_eye
    [52, 54],                   # nose->left_eye
    [53, 55],                   # right_eye->right_ear
    [54, 56],                   # left_eye->left_ear

    # feet details
    [62, 60],                   # right heel->right big toe
    [62, 61],                   # right heel->right small toe
    [59, 57],                   # left heel->left big toe
    [59, 58],                   # left heel->left small toe
]

SMPLX_JOINT_NAMES_73 = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_index1', 'left_index2', 'left_index3',
    'left_middle1', 'left_middle2', 'left_middle3',
    'left_pinky1', 'left_pinky2', 'left_pinky3',
    'left_ring1', 'left_ring2', 'left_ring3',
    'left_thumb1', 'left_thumb2', 'left_thumb3',
    'right_index1', 'right_index2', 'right_index3',
    'right_middle1', 'right_middle2', 'right_middle3',
    'right_pinky1', 'right_pinky2', 'right_pinky3',
    'right_ring1', 'right_ring2', 'right_ring3',
    'right_thumb1', 'right_thumb2', 'right_thumb3',
    'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear',
    'left_big_toe', 'left_small_toe', 'left_heel',
    'right_big_toe', 'right_small_toe', 'right_heel',
    'left_thumb_tip', 'left_index_tip', 'left_middle_tip', 'left_ring_tip',
    'left_pinky_tip', 'right_thumb_tip', 'right_index_tip', 'right_middle_tip',
    'right_ring_tip', 'right_pinky_tip',
]
