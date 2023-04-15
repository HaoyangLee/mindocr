'''
Detection model inference
'''
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../../')))

import os
import yaml
import argparse
from addict import Dict
from tqdm import tqdm

import cv2
import numpy as np
import mindspore as ms

from mindocr.data import build_dataset
from mindocr.models import build_model
from mindocr.postprocess import build_postprocess
# from tools.utils.visualize import VisMode, Visualization
from mindocr.utils.visualize import show_img, draw_bboxes, show_imgs, recover_image
from tools.utils.visualize import Visualization

class BaseInfer(object):
    def __init__(self, infer_cfg):
        # env init
        ms.set_context(mode=infer_cfg.system.mode)
        if infer_cfg.system.distribute:
            print("WARNING: Distribut mode blocked. Evaluation only runs in standalone mode.")

        self.loader_infer, self.dataset_column_names = build_dataset(
            infer_cfg.eval.dataset,
            infer_cfg.eval.loader,
            num_shards=None,
            shard_id=None,
            is_train=False)
        self.img_path_column_idx = self.dataset_column_names.index('img_path')
        self.image_column_idx = self.dataset_column_names.index('image')
        num_batches = self.loader_infer.get_dataset_size()
        print('---:', num_batches)

        # model
        assert 'ckpt_load_path' in infer_cfg.eval, f'Please provide \n`eval:\n\tckpt_load_path`\n in the yaml config file '
        self.network = build_model(infer_cfg.model, ckpt_load_path=infer_cfg.eval.ckpt_load_path)
        self.network.set_train(False)

        if infer_cfg.system.amp_level != 'O0':
            print('INFO: Evaluation will run in full-precision(fp32)')

        # TODO: check float type conversion in official Model.eval
        #ms.amp.auto_mixed_precision(network, amp_level='O0')

        self.postprocessor = build_postprocess(infer_cfg.postprocess)

    def __call__(self, num_image):
        img_path_list = []
        infer_image = []
        pred_result = []
        # TODO: it's better for loader should to return dict with keys rather than list of tensors
        for idx, data in tqdm(enumerate(self.loader_infer)):
            if len(pred_result) > num_image - 1:
                break
            img_path = data[self.img_path_column_idx]
            image = data[self.image_column_idx]
            pred = self.network(image)
            pred = self.postprocessor(pred)  # [bboxes, scores], shape=[(N, K, 4, 2), (N, K)]

            img_path_list.append(img_path)
            pred_result.append(pred)
            infer_image.append(image)
        return infer_image, pred_result, img_path_list

    # def vis(self, vis_mode):
    #     vis_tool = Visualization(vis_mode)




    # print('='*40)
    # print(
    #     f'Num batches: {num_batches}\n'
    #     )
    # if 'name' in cfg.model:
    #     print(f'Model: {cfg.model.name}')
    # else:
    #     print(f'Model: {cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}')
    # print('='*40)
    #
    # # column_names = cfg.eval.dataset.output_columns  # all column names=['img_path', 'label', 'image', 'polys', 'texts', 'ignore_tags']
    # # print('column_names:', column_names)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation Config', add_help=False)
    parser.add_argument('-c', '--config', type=str, default='../../../configs/rec/crnn/crnn_resnet34.yaml',
                        help='YAML config file specifying default arguments (default='')')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # argpaser
    args = parse_args()
    yaml_fp = args.config
    with open(yaml_fp) as fp:
        config = yaml.safe_load(fp)
    config = Dict(config)
    det_infer = BaseInfer(config)
    pred_result = det_infer()
    for res in pred_result:
        print('-------->>>>>>')
        import pdb
        pdb.set_trace()
        print(res)


