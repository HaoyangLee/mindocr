"""
The online ckpt files are downloaded from https://download.mindspore.cn/toolkits/mindocr/
Usage:
    To export all trained models from online ckpt to mindir as listed in configs/, run
       $ python tools/export.py

    To export a specific model by downloading online ckpt, taking dbnet_resnet50 for example, run
       $ python tools/export.py --model_name dbnet_resnet50

    To export a specific model by loading local ckpt, taking dbnet_resnet50 for example, run
       $ python tools/export.py --model_name dbnet_resnet50 --local_ckpt_path /path/to/local_ckpt
"""
import sys
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import argparse
import mindspore as ms
from mindocr import list_models, build_model
import numpy as np


# def export(name, task='rec', local_ckpt_path="", save_dir=""):
def export(name, data_shape, local_ckpt_path="", save_dir=""):
    ms.set_context(mode=ms.GRAPH_MODE) #, device_target='Ascend')
    if local_ckpt_path:
        net = build_model(name, pretrained=False, ckpt_load_path=local_ckpt_path)
    else:
        net = build_model(name, pretrained=True)
    net.set_train(False)

    # TODO: extend input shapes for more models
    # if task == 'rec':
    #     c, h, w = 3, 32, 100
    # else:
    #     c, h, w = 3, 736, 1280
    h, w = data_shape
    bs, c = 1, 3
    x = ms.Tensor(np.ones([bs, c, h, w]), dtype=ms.float32)

    output_path = os.path.join(save_dir, name) + '.mindir'
    ms.export(net, x, file_name=output_path, file_format='MINDIR')

    print(f'=> Finish exporting {name} to {os.path.realpath(output_path)}. The data shape [H, W] is {data_shape}')


# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ("yes", "true", "1"):
#         return True
#     elif v.lower() in ("no", "false", "0"):
#         return False
#     else:
#         raise argparse.ArgumentTypeError("Boolean value expected.")


def check_args(args):
    if args.model_name not in list_models():
        raise ValueError(f"Invalid arg 'model_name': {args.model_name}. 'model_name' must be empty or one of names in {list_models()}.")
    if args.local_ckpt_path and not os.path.isfile(args.local_ckpt_path):
        raise ValueError(f"Local ckpt file {args.local_ckpt_path} doesn't not exist. Please check arg 'local_ckpt_path'.")
    if args.save_dir and not os.path.isdir(args.save_dir):
        raise ValueError(f"Directory {args.save_dir} doesn't not exist. Please check arg 'save_dir'.")

    # if args.save_dir:
    #     if os.path.isdir(args.save_dir):
    #         os.makedirs(args.save_dir, exist_ok=True)
    #     else:
    #         raise ValueError(f"Invalid arg 'save_dir': {args.save_dir}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert model checkpoint to mindir format.")
    parser.add_argument(
        '--model_name',
        type=str,
        default="",
        required=True,
        help=f'Name of the model to convert. Available choices: {list_models()}. '
             'If empty, all supported models will be converted.')
    parser.add_argument(
        '--data_shape',
        type=int,
        nargs=2,
        default="",
        required=True,
        help=f'The data shape [H, W] for exporting mindir files. It is recommended to be the same as the training data shape in order to get the best inference performance.')
    parser.add_argument(
        '--local_ckpt_path',
        type=str,
        default="",
        help='Path to a local checkpoint. If set, export a specific model by loading local ckpt. Otherwise, export all models or a specific model by downloading online ckpt.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default="",
        help='Dir to save the exported model')

    args = parser.parse_args()
    check_args(args)

    export(args.model_name, args.data_shape, args.local_ckpt_path, args.save_dir)

    # if args.model_name == "":
    #     model_names = list_models()
    # else:
    #     model_names = [args.model_name]

    # for n in model_names:
    #     task = 'rec'
    #     if 'db' in n or 'east' in n:
    #         task = 'det'
    #     export(n, task, args.local_ckpt_path, args.save_dir)
