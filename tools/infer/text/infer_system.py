import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../../')))

import yaml
import shutil
import argparse
from addict import Dict
import numpy as np
import cv2

from base_infer import BaseInfer
from tools.utils.visualize import Visualization, VisMode
from mindocr.utils.visualize import recover_image

def infer_det(args):
    num_image = 1 # TODO
    det_cfg = load_yaml(args.det_config_path)
    det_infer = BaseInfer(det_cfg)
    # image_list = [image1, image2, ...]
    # box_list = [[[(boxes1, scores1)]], [[(boxes2, scores2)]], ...]
    image_list, box_list, img_path_list = det_infer(num_image)
    vis_tool = Visualization(VisMode.crop)
    cropped_image_list = []
    for idx, image in enumerate(image_list):
        image = np.squeeze(image.asnumpy(), axis=0)  # TODO: only works when batch size = 1
        image = recover_image(image)
        cropped_images = vis_tool(image, box_list[idx][0][0])
        cropped_image_list.append(cropped_images)

        # TODO: VisMode.bbox
        # save cropped images
        # TODO: check the text direction of all saved images
        if args.pipeline_crop_save_dir:
            original_img_path = img_path_list[idx].asnumpy()[0]
            original_img_filename = os.path.splitext(os.path.basename(original_img_path))[0]
            for i, crop in enumerate(cropped_images):
                crop_save_filename = original_img_filename + '_crop_' + str(i) + '.jpg'
                cv2.imwrite(os.path.join(args.pipeline_crop_save_dir, crop_save_filename), crop)

    return image_list, box_list, img_path_list, cropped_image_list

def infer_rec(args, image_list, box_list, img_path_list, cropped_image_list):
    num_image = len(cropped_image_list[0])# TODO
    rec_cfg = load_yaml(args.rec_config_path)
    rec_cfg.eval.dataset.data_dir = args.pipeline_crop_save_dir # TODO
    rec_cfg.eval.loader.batch_size = 1 # TODO
    rec_infer = BaseInfer(rec_cfg)
    _, rec_result, _ = rec_infer(num_image)
    text_list = [r['texts'][0] for r in rec_result]

    vis_tool = Visualization(VisMode.bbox_text)
    for idx, image in enumerate(image_list):
        original_img_path = img_path_list[idx].asnumpy()[0]
        original_img_filename = os.path.splitext(os.path.basename(original_img_path))[0]
        pl_vis_filename = original_img_filename + '_pl_vis' + '.jpg'
        image = np.squeeze(image.asnumpy(), axis=0)  # TODO: only works when batch size = 1
        image = recover_image(image)
        box_text = vis_tool(recover_image(image), box_list[0][0][0], text_list, font_path=args.vis_font_path)
        cv2.imwrite(os.path.join(args.vis_pipeline_save_dir, pl_vis_filename), box_text)

    return text_list

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation Config', add_help=False)
    parser.add_argument('--det_config_path', type=str, default='../../../configs/det/db_r50_icdar15.yaml',
                        help='YAML config file specifying default arguments (default='')')
    parser.add_argument('--rec_config_path', type=str, default='../../../configs/rec/crnn/crnn_resnet34.yaml',
                        help='YAML config file specifying default arguments (default='')')
    parser.add_argument('--pipeline_crop_save_dir', type=str, default='/home/mindspore/lhy/mindocr_0317/mindocr/tools/infer/text/crop', required=False,
                        help='Saving dir for images cropped during pipeline.')
    parser.add_argument('--vis_font_path', type=str, default='/home/mindspore/lhy/mindocr_0317/mindocr/tools/utils/simfang.ttf' , required=False,
                        help='Font file path for recognition model.')
    parser.add_argument('--vis_det_save_dir', type=str, required=False,
                        help='Saving dir for visualization of detection results.')
    parser.add_argument('--vis_pipeline_save_dir', type=str, default='/home/mindspore/lhy/mindocr_0317/mindocr/tools/infer/text/pipeline_save', required=False,
                        help='Saving dir for visualization of pipeline inference results.')
    args = parser.parse_args()

    # TODO
    if os.path.exists(args.pipeline_crop_save_dir):
        shutil.rmtree(args.pipeline_crop_save_dir)
    os.makedirs(args.pipeline_crop_save_dir)

    if os.path.exists(args.vis_pipeline_save_dir):
        shutil.rmtree(args.vis_pipeline_save_dir)
    os.makedirs(args.vis_pipeline_save_dir)
    return args

def load_yaml(yaml_fp):
    with open(yaml_fp) as fp:
        config = yaml.safe_load(fp)
    config = Dict(config)
    return config


if __name__ == '__main__':
    args = parse_args()
    image_list, box_list, img_path_list, cropped_image_list = infer_det(args)
    print('infer det finished!!!')
    text_list = infer_rec(args, image_list, box_list, img_path_list, cropped_image_list)
    print('infer system finished!!!')
