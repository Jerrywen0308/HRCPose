# def mhsa(num_heads, key_dim, value_dim, px):
#     """Helper to build MHSA spec for Hybrid variants."""
#     if px == 24:
#         kv_strides = 2
#     elif px == 12:
#         kv_strides = 1
#     else:
#         kv_strides = 1
#     query_h_strides = 1
#     query_w_strides = 1
#     use_layer_scale = True
#     use_multi_query = True
#     use_residual = True
#     return [
#         num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides,
#         use_layer_scale, use_multi_query, use_residual
#     ]
def mhsa(num_heads=2, key_dim=32, value_dim=16, px=12):
    """轻量 MQA：Small/OS=16 场景建议 kv_strides=1。"""
    # 你的 mhsa() 若已固定 kv_strides=1，就保持不变即可；
    # 若还按 px 切换，这里传 px=12 会让 kv_strides=1。
    kv_strides = 1
    query_h_strides = 1
    query_w_strides = 1
    use_layer_scale = True
    use_multi_query = True
    use_residual = True
    return [
        num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides,
        use_layer_scale, use_multi_query, use_residual
    ]


# ------------------------------ Conv Small ------------------------------
MNV4ConvSmall_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [32, 32, 3, 2],
            [32, 32, 1, 1]
        ]
    },
    "layer2": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [32, 96, 3, 2],
            [96, 64, 1, 1]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            # [inp, oup, start_k, middle_k, middle_down, stride, expand, use_layer_scale]
            # 修正：stride=2 且 start/middle 同时存在，middle_down -> False
            [64,  96, 5, 5, False, 2, 3, False],
            [96,  96, 0, 3, True,  1, 2, False],
            [96,  96, 0, 3, True,  1, 2, False],
            [96,  96, 0, 3, True,  1, 2, False],
            [96,  96, 0, 3, True,  1, 2, False],
            [96,  96, 3, 0, True,  1, 4, False],
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            # 修正：同理 middle_down -> False
            [96,  128, 3, 3, False, 2, 6, False],
            [128, 128, 5, 5, True,  1, 4, False],
            [128, 128, 0, 5, True,  1, 4, False],
            [128, 128, 0, 5, True,  1, 3, False],
            [128, 128, 0, 3, True,  1, 4, False],
            [128, 128, 0, 3, True,  1, 4, False],
        ]
    }
}


# ------------------------------ Conv Medium ------------------------------
MNV4ConvMedium_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            # [inp, oup, stride, expand_ratio, act]
            [32, 48, 2, 4.0, False]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            # 修正：middle_down -> False
            [48, 80, 3, 5, False, 2, 4, False],
            [80, 80, 3, 3, True,   1, 2, False]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 8,
        "block_specs": [
            # 修正：middle_down -> False
            [80,  160, 3, 5, False, 2, 6, False],
            [160, 160, 3, 3, True,   1, 4, False],
            [160, 160, 3, 3, True,   1, 4, False],
            [160, 160, 3, 5, True,   1, 4, False],
            [160, 160, 3, 3, True,   1, 4, False],
            [160, 160, 3, 0, True,   1, 4, False],
            [160, 160, 0, 0, True,   1, 2, False],
            [160, 160, 3, 0, True,   1, 4, False]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            # 修正：middle_down -> False
            [160, 256, 5, 5, False, 2, 6, False],
            [256, 256, 5, 5, True,   1, 4, False],
            [256, 256, 3, 5, True,   1, 4, False],
            [256, 256, 3, 5, True,   1, 4, False],
            [256, 256, 0, 0, True,   1, 4, False],
            [256, 256, 3, 0, True,   1, 4, False],
            [256, 256, 3, 5, True,   1, 2, False],
            [256, 256, 5, 5, True,   1, 4, False],
            [256, 256, 0, 0, True,   1, 4, False],
            [256, 256, 0, 0, True,   1, 4, False],
            [256, 256, 5, 0, True,   1, 2, False]
        ]
    }
}


# ------------------------------ Conv Large ------------------------------
MNV4ConvLarge_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 24, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [24, 48, 2, 4.0, False]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            # 修复：补上 use_layer_scale=False；并把 middle_down -> False
            [48,  96, 3, 5, False, 2, 4, False],
            [96,  96, 3, 3, True,  1, 4, False]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            # 修正：middle_down -> False
            [96,  192, 3, 5, False, 2, 4, False],
            [192, 192, 3, 3, True,  1, 4, False],
            [192, 192, 3, 3, True,  1, 4, False],
            [192, 192, 3, 3, True,  1, 4, False],
            [192, 192, 3, 5, True,  1, 4, False],
            [192, 192, 5, 3, True,  1, 4, False],
            [192, 192, 5, 3, True,  1, 4, False],
            [192, 192, 5, 3, True,  1, 4, False],
            [192, 192, 5, 3, True,  1, 4, False],
            [192, 192, 5, 3, True,  1, 4, False],
            [192, 192, 3, 0, True,  1, 4, False]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 13,
        "block_specs": [
            # 修正：middle_down -> False
            [192, 512, 5, 5, False, 2, 4, False],
            [512, 512, 5, 5, True,  1, 4, False],
            [512, 512, 5, 5, True,  1, 4, False],
            [512, 512, 5, 5, True,  1, 4, False],
            [512, 512, 5, 0, True,  1, 4, False],
            [512, 512, 5, 3, True,  1, 4, False],
            [512, 512, 5, 0, True,  1, 4, False],
            [512, 512, 5, 0, True,  1, 4, False],
            [512, 512, 5, 3, True,  1, 4, False],
            [512, 512, 5, 5, True,  1, 4, False],
            [512, 512, 5, 0, True,  1, 4, False],
            [512, 512, 5, 0, True,  1, 4, False],
            [512, 512, 5, 0, True,  1, 4, False]
        ]
    }
}


# ------------------------------ Hybrid Conv Small PLUS ------------------------------
MNV4ConvSmallPlus_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]                 # /2
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            # [inp, oup, stride, expand_ratio, act]
            [32, 48, 2, 4.0, False]       # /4
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            # [inp, oup, start_k, middle_k, middle_down, stride, expand, use_layer_scale]
            # 下采样到 /8；start & middle 同时存在 → middle_down=False
            [48,  80, 3, 5, False, 2, 4, True],   # /8
            [80,  80, 3, 3, True,   1, 2, True]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            # 下采样到 /16；保持卷积为主，不放注意力
            [80,  128, 5, 5, False, 2, 4, True],  # /16
            [128, 128, 0, 3, True,   1, 4, True],
            [128, 128, 0, 3, True,   1, 4, True],
            [128, 128, 3, 5, True,   1, 4, True],
            [128, 128, 3, 3, True,   1, 4, True],
            [128, 128, 3, 0, True,   1, 4, True],
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            # 保持 /16 输出，通道升到 256；去掉注意力
            [128, 256, 3, 3, True,   1, 6, True],  # /16 保持
            [256, 256, 5, 5, True,   1, 4, True],
            [256, 256, 3, 5, True,   1, 4, True],
            [256, 256, 0, 3, True,   1, 4, True],
            [256, 256, 0, 3, True,   1, 4, True],  # ← 原来这里有 mhsa(...)，已移除
            [256, 256, 0, 3, True,   1, 4, True],  # ← 原来这里有 mhsa(...)，已移除
        ]
    }
}


MNV4HybridConvSmallPlus_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]                 # /2
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            # [inp, oup, stride, expand_ratio, act]
            [32, 48, 2, 4.0, False]       # /4
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            # [inp, oup, start_k, middle_k, middle_down, stride, expand, use_layer_scale, (opt)mhsa]
            # 下采样到 /8；start & middle 同时存在 → middle_down=False
            [48,  80, 3, 5, False, 2, 4, True],   # /8
            [80,  80, 3, 3, True,   1, 2, True]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            # 下采样到 /16；保持卷积为主，不放注意力
            [80,  128, 5, 5, False, 2, 4, True],  # /16
            [128, 128, 0, 3, True,   1, 4, True],
            [128, 128, 0, 3, True,   1, 4, True],
            [128, 128, 3, 5, True,   1, 4, True],
            [128, 128, 3, 3, True,   1, 4, True],
            [128, 128, 3, 0, True,   1, 4, True],
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            # 关键：首块恢复下采样（从 /16 -> /32），middle_down=False 保证只在 start_dw 下采样
            [128, 256, 3, 3, False, 2, 6, True],           # /32 now
            [256, 256, 5, 5, True,   1, 4, True],
            [256, 256, 3, 5, True,   1, 4, True],
            [256, 256, 0, 3, True,   1, 4, True],
            # 只在最后两块挂轻量 MQA（heads=2, key_dim=32, value_dim=16, kv_strides=1）
            [256, 256, 0, 3, True,   1, 4, True, mhsa(2, 32, 16, 12)],
            [256, 256, 0, 3, True,   1, 4, True, mhsa(2, 32, 16, 12)],
        ]
    }
}

# ------------------------------ Hybrid Conv Medium ------------------------------
MNV4HybridConvMedium_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [32, 48, 2, 4.0, False]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            # 修正：middle_down -> False
            [48, 80, 3, 5, False, 2, 4, True],
            [80, 80, 3, 3, True,   1, 2, True]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 8,
        "block_specs": [
            # 修正：middle_down -> False
            [80,  160, 3, 5, False, 2, 6, True],
            [160, 160, 0, 0, True,   1, 2, True],
            [160, 160, 3, 3, True,   1, 4, True],
            [160, 160, 3, 5, True,   1, 4, True, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 3, True,   1, 4, True, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 0, True,   1, 4, True, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 3, True,   1, 4, True, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 0, True,   1, 4, True]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 12,
        "block_specs": [
            [160, 256, 5, 5, False, 2, 6, True],  # 修正
            [256, 256, 5, 5, True,   1, 4, True],
            [256, 256, 3, 5, True,   1, 4, True],
            [256, 256, 3, 5, True,   1, 4, True],
            [256, 256, 0, 0, True,   1, 2, True],
            [256, 256, 3, 5, True,   1, 2, True],
            [256, 256, 0, 0, True,   1, 2, True],
            [256, 256, 0, 0, True,   1, 4, True, mhsa(4, 64, 64, 12)],
            [256, 256, 3, 0, True,   1, 4, True, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 5, True,   1, 4, True, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 0, True,   1, 4, True, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 0, True,   1, 4, True]
        ]
    }
}


# ------------------------------ Hybrid Conv Large ------------------------------
MNV4HybridConvLarge_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 24, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            # 修复：去掉多余参数
            [24, 48, 2, 4.0, False]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            # 修正：middle_down -> False
            [48, 96, 3, 5, False, 2, 4, True],
            [96, 96, 3, 3, True,   1, 4, True]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            # 修正：middle_down -> False
            [96,  192, 3, 5, False, 2, 4, True],
            [192, 192, 3, 3, True,   1, 4, True],
            [192, 192, 3, 3, True,   1, 4, True],
            [192, 192, 3, 3, True,   1, 4, True],
            [192, 192, 3, 5, True,   1, 4, True],
            [192, 192, 5, 3, True,   1, 4, True],
            [192, 192, 5, 3, True,   1, 4, True, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True,   1, 4, True, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True,   1, 4, True, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True,   1, 4, True, mhsa(8, 48, 48, 24)],
            [192, 192, 3, 0, True,   1, 4, True]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 14,
        "block_specs": [
            # 修正：middle_down -> False
            [192, 512, 5, 5, False, 2, 4, True],
            [512, 512, 5, 5, True,   1, 4, True],
            [512, 512, 5, 5, True,   1, 4, True],
            [512, 512, 5, 5, True,   1, 4, True],
            [512, 512, 5, 0, True,   1, 4, True],
            [512, 512, 5, 3, True,   1, 4, True],
            [512, 512, 5, 0, True,   1, 4, True],
            [512, 512, 5, 0, True,   1, 4, True],
            [512, 512, 5, 3, True,   1, 4, True],
            [512, 512, 5, 5, True,   1, 4, True, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True,   1, 4, True, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True,   1, 4, True, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True,   1, 4, True, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True,   1, 4, True]
        ]
    }
}


MODEL_SPECS = {
    "MobileNetV4ConvSmall": MNV4ConvSmall_BLOCK_SPECS,
    "MobileNetV4ConvMedium": MNV4ConvMedium_BLOCK_SPECS,
    "MobileNetV4ConvLarge": MNV4ConvLarge_BLOCK_SPECS,
    "MobileNetV4ConvSmallPlus": MNV4ConvSmallPlus_BLOCK_SPECS, 
    "MobileNetV4HybridMedium": MNV4HybridConvMedium_BLOCK_SPECS,
    "MobileNetV4HybridLarge": MNV4HybridConvLarge_BLOCK_SPECS,
    "MobileNetV4HybridSmallPlus": MNV4HybridConvSmallPlus_BLOCK_SPECS, 
}
