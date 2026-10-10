_base_ = ['../../../_base_/default_runtime.py']

default_scope = 'mmpose'

custom_imports = dict(
    imports=[
        'mmpose.datasets',
        'mmpose.models',
        'mmpose.visualization',
    ],
    allow_failed_imports=False
)

# ===== codec (SimCC) =====
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False
)

# ===== model =====
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='YOLOv11SPPFDWBackbone',
        deepen_factor=0.50,
        widen_factor=0.25,
        max_channels=1024,
        out_indices=(4, 6, 10),
        norm_eval=False,
    ),
    neck=dict(
        type='MSDF',
        C3=128,
        C4=128,
        C5=256,
        d=128,
        eca_k=3
    ),
    head=dict(
        type='RTMCC_CoordAttnHead',
        in_channels=128,
        out_channels=17,
        input_size=(256, 256),
        in_featuremap_size=(256 // 8, 256 // 8),
        simcc_split_ratio=2.0,
        axial_d=1,
        final_layer_cfg=dict(
            axial_k=7,
            ca_reduction=8
        ),
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False
        ),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True
        ),
        decoder=dict(
            type='SimCCLabel',
            input_size=(256, 256),
            sigma=(5.66, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False
        )
    ),
    test_cfg=dict(flip_test=True),
)

# ===== dataset / inference pipeline =====
dataset_type = 'AP10KDataset'
data_mode = 'topdown'
data_root = 'data/ap10k/'
backend_args = dict(backend='local')

test_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/ap10k-test-split1.json',
        data_prefix=dict(img='data/'),
        test_mode=True,
        pipeline=test_pipeline,
    )
)

val_dataloader = test_dataloader