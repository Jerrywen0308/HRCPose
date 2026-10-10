# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import io
import numpy as np
import torch
from mmengine.config import DictAction
from mmengine.logging import MMLogger
import time
from mmpose.apis.inference import init_model
from contextlib import nullcontext
try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')
def autocast_ctx(device: str, enabled: bool, amp_dtype=None):
    if not enabled:
        return nullcontext()


    if device.startswith('cuda') and torch.cuda.is_available():

        if hasattr(torch, 'autocast'):
            if amp_dtype is not None:
                try:
                    return torch.autocast('cuda', dtype=amp_dtype)
                except TypeError:
                    return torch.autocast('cuda')
            else:
                return torch.autocast('cuda')

        if amp_dtype is not None:
            try:
                return torch.cuda.amp.autocast(dtype=amp_dtype)
            except TypeError:
                return torch.cuda.amp.autocast()
        else:
            return torch.cuda.amp.autocast()


    if device == 'cpu' and hasattr(torch, 'autocast'):
        if amp_dtype is not None:
            try:
                return torch.autocast('cpu', dtype=amp_dtype)
            except TypeError:
                return torch.autocast('cpu')
        else:
            return torch.autocast('cpu')


    return nullcontext()



# ==== NEW: size utils ====
import io

def _human_bytes(num_bytes: int) -> str:
    units = ['B','KiB','MiB','GiB','TiB']
    size = float(num_bytes)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f'{size:.2f} {u}'
        size /= 1024.0




def parse_args():
    parser = argparse.ArgumentParser(
        description='Get complexity information from a model config')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--device', default='cpu', help='Device used for model initialization')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs='+',
        default=[256, 192],
        help='input image size')
    parser.add_argument(
        '--batch-size',
        '-b',
        type=int,
        default=1,
        help='Input batch size. If specified and greater than 1, it takes a '
        'callable method that generates a batch input. Otherwise, it will '
        'generate a random tensor with input shape to calculate FLOPs.')
    parser.add_argument(
        '--show-arch-info',
        '-s',
        action='store_true',
        help='Whether to show model arch information')

    
    
    
    args = parser.parse_args()
    return args


def batch_constructor(flops_model, batch_size, input_shape):
    """Generate a batch of tensors to the model."""
    batch = {}

    inputs = torch.randn(batch_size, *input_shape).new_empty(
        (batch_size, *input_shape),
        dtype=next(flops_model.parameters()).dtype,
        device=next(flops_model.parameters()).device)

    batch['inputs'] = inputs
    return batch




def make_inputs_like_model(model, batch_size, input_shape, device, dtype_str, channels_last=False):

    if dtype_str == 'fp16':
        dtype = torch.float16
        if device.startswith('cpu'):

            dtype = torch.float32
    elif dtype_str == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32


    ishape = tuple(input_shape)
    if len(ishape) == 4:

        ishape = ishape[1:]
    if len(ishape) == 1:         
        ishape = (3, ishape[0], ishape[0])
    elif len(ishape) == 2:       
        ishape = (3, ishape[0], ishape[1])
    elif len(ishape) == 3:
        pass
    else:
        raise ValueError(f'invalid input_shape={input_shape}, expect (H,W) or (3,H,W)')

    C, H, W = ishape
    assert C in (1, 3), f'Channel C should be 1 or 3, got {C} from {ishape}'

    x = torch.randn((max(1, int(batch_size)), C, H, W), device=device, dtype=dtype)

    if channels_last and device.startswith('cuda') and torch.cuda.is_available() and x.ndim == 4:

        x = x.to(memory_format=torch.channels_last)

    return x

def inference(args, input_shape, logger):
    model = init_model(
        args.config, checkpoint=None, device=args.device, cfg_options=args.cfg_options
    )

    if hasattr(model, '_forward'):
        model.forward = model._forward
    else:
        raise NotImplementedError(
            f'FLOPs counter is not supported with {model.__class__.__name__}'
        )

    # FLOPs/Params（与 batch 无关）
    outputs = get_model_complexity_info(
        model,
        input_shape,   # (3, H, W)
        inputs=None,
        show_table=True,
        show_arch=args.show_arch_info
    )


    return outputs




def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    if len(args.input_shape) == 1:
        input_shape = (3, args.input_shape[0], args.input_shape[0])
    elif len(args.input_shape) == 2:
        input_shape = (3, ) + tuple(args.input_shape)
    else:
        raise ValueError('invalid input shape')

    if args.device == 'cuda:0':
        assert torch.cuda.is_available(
        ), 'No valid cuda device detected, please double check...'

    outputs = inference(args, input_shape, logger)
    flops = outputs['flops_str']
    params = outputs['params_str']
    split_line = '=' * 30
    input_shape = (args.batch_size, ) + input_shape
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print(outputs['out_table'])
    if args.show_arch_info:
        print(outputs['out_arch'])
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')





if __name__ == '__main__':
    main()
#python get_flops.py configs/animal_2d_keypoint/HRCPose/ap10k/HRCPose_ap10k-256x256.py --input-shape 256 256 --device cuda:0
