# configs/_base_/datasets/monkey17.py
# Monkey-17 (your order) dataset_info for MMPose 0.x

dataset_info = dict(
    dataset_name='monkey17',
    paper_info=dict(
        author='Your Team',
        title='Monkey-17 keypoint definition',
        container='Internal dataset',
        year='2025',
        homepage=''
    ),

    # -------- Keypoints (IDs follow YOUR order) --------
    keypoint_info={
        0:  dict(name='right_eye', id=0,  color=[255, 128,   0], type='upper', swap='left_eye'),
        1:  dict(name='left_eye',  id=1,  color=[  0, 255,   0], type='upper', swap='right_eye'),
        2:  dict(name='nose',      id=2,  color=[ 51, 153, 255], type='upper', swap=''),
        3:  dict(name='head',      id=3,  color=[ 51, 153, 255], type='upper', swap=''),
        4:  dict(name='neck',      id=4,  color=[ 51, 153, 255], type='upper', swap=''),
        5:  dict(name='right_shoulder', id=5, color=[255,128,0], type='upper', swap='left_shoulder'),
        6:  dict(name='right_elbow',    id=6, color=[255,128,0], type='upper', swap='left_elbow'),
        7:  dict(name='right_hand',     id=7, color=[255,128,0], type='upper', swap='left_hand'),
        8:  dict(name='left_shoulder',  id=8, color=[0,255,0],   type='upper', swap='right_shoulder'),
        9:  dict(name='left_elbow',     id=9, color=[0,255,0],   type='upper', swap='right_elbow'),
        10: dict(name='left_hand',      id=10,color=[0,255,0],   type='upper', swap='right_hand'),
        11: dict(name='hip',            id=11,color=[51,153,255],type='lower', swap=''),
        12: dict(name='right_knee',     id=12,color=[255,128,0], type='lower', swap='left_knee'),
        13: dict(name='right_foot',     id=13,color=[255,128,0], type='lower', swap='left_foot'),
        14: dict(name='left_knee',      id=14,color=[0,255,0],   type='lower', swap='right_knee'),
        15: dict(name='left_foot',      id=15,color=[0,255,0],   type='lower', swap='right_foot'),
        16: dict(name='tail',           id=16,color=[51,153,255],type='lower', swap=''),
    },

    # -------- Skeleton (lines for visualization/eval) --------
    skeleton_info={
        0:  dict(link=('right_eye', 'head'),          id=0,  color=[255,128,0]),
        1:  dict(link=('left_eye',  'head'),          id=1,  color=[0,255,0]),
        2:  dict(link=('nose',      'head'),          id=2,  color=[51,153,255]),
        3:  dict(link=('head',      'neck'),          id=3,  color=[51,153,255]),

        4:  dict(link=('neck', 'right_shoulder'),     id=4,  color=[255,128,0]),
        5:  dict(link=('right_shoulder','right_elbow'), id=5, color=[255,128,0]),
        6:  dict(link=('right_elbow','right_hand'),   id=6,  color=[255,128,0]),

        7:  dict(link=('neck', 'left_shoulder'),      id=7,  color=[0,255,0]),
        8:  dict(link=('left_shoulder','left_elbow'), id=8,  color=[0,255,0]),
        9:  dict(link=('left_elbow','left_hand'),     id=9,  color=[0,255,0]),

        10: dict(link=('neck','hip'),                 id=10, color=[51,153,255]),
        11: dict(link=('hip','right_knee'),           id=11, color=[255,128,0]),
        12: dict(link=('right_knee','right_foot'),    id=12, color=[255,128,0]),
        13: dict(link=('hip','left_knee'),            id=13, color=[0,255,0]),
        14: dict(link=('left_knee','left_foot'),      id=14, color=[0,255,0]),

        15: dict(link=('hip','tail'),                 id=15, color=[255,255,0]),
        16: dict(link=('left_eye','right_eye'),       id=16, color=[51,153,255]),  # 可选
    },

    # 关键点权重（可按需微调；手/脚略高）
    joint_weights=[
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.2, 1.5, 1.0, 1.2, 1.5,
        1.0, 1.2, 1.5, 1.2, 1.5,
        1.0,  # tail
    ],

    # 要你说的：直接沿用 COCO 的 sigmas（17 个）
    sigmas=[0.026, 0.025, 0.025, 0.035, 0.035,
            0.079, 0.079, 0.072, 0.072, 0.062,
            0.062, 0.107, 0.107, 0.087, 0.087,
            0.089, 0.089],
    flip_pairs=[
        (0, 1),   # right_eye  ↔ left_eye
        (5, 8),   # right_shoulder ↔ left_shoulder
        (6, 9),   # right_elbow    ↔ left_elbow
        (7, 10),  # right_hand     ↔ left_hand
        (12, 14), # right_knee     ↔ left_knee
        (13, 15), # right_foot     ↔ left_foot
        # 中线点不写：nose(2), head(3), neck(4), hip(11), tail(16)
    ],
)
