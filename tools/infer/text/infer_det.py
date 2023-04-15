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

# from mindocr.metrics import build_metric
# from mindocr.utils.callbacks import Evaluator

def main(cfg):
    # env init
    ms.set_context(mode=cfg.system.mode)
    if cfg.system.distribute:
        print("WARNING: Distribut mode blocked. Evaluation only runs in standalone mode.")

    loader_infer, dataset_column_names = build_dataset(
            cfg.eval.dataset, 
            cfg.eval.loader,
            num_shards=None,
            shard_id=None,
            is_train=False)
    num_batches = loader_infer.get_dataset_size()
    print('---:', num_batches)

    # model
    assert 'ckpt_load_path' in cfg.eval, f'Please provide \n`eval:\n\tckpt_load_path`\n in the yaml config file '
    network = build_model(cfg.model, ckpt_load_path=cfg.eval.ckpt_load_path)
    network.set_train(False)

    if cfg.system.amp_level != 'O0':
        print('INFO: Evaluation will run in full-precision(fp32)')

    # TODO: check float type conversion in official Model.eval 
    #ms.amp.auto_mixed_precision(network, amp_level='O0')  

    # postprocess, metric
    postprocessor = build_postprocess(cfg.postprocess)
    # postprocess network prediction
    # metric = build_metric(cfg.metric)

    # net_evaluator = Evaluator(network, None, postprocessor, [metric])

    # log
    print('='*40)
    print(
        f'Num batches: {num_batches}\n'
        )
    if 'name' in cfg.model:
        print(f'Model: {cfg.model.name}')
    else:
        print(f'Model: {cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}')
    print('='*40)

    column_names = cfg.eval.dataset.output_columns  # all column names=['img_path', 'label', 'image', 'polys', 'texts', 'ignore_tags']
    print('column_names:', column_names)

    image_column_idx = dataset_column_names.index('image')
    pred_result = []
    # TODO: it's better for loader should to return dict with keys rather than list of tensors
    for idx, data in tqdm(enumerate(loader_infer)):
        if len(pred_result) > 3:
            break
        pred = network(data[image_column_idx])
        pred = postprocessor(pred)    # [bboxes, scores], shape=[(N, K, 4, 2), (N, K)]
        pred_result.append(pred)

    # TODO: reshape(-1, 2) if polygon
    # TODO: batch_size issue, need to squeeze axis=0 now
    # TODO: no shuffle in cfg.eval, check and match the order of img name of loader_infer
    visualize = True
    vis_det_save_dir = '/home/group1/lhy/vis/det'
    os.makedirs(vis_det_save_dir, exist_ok=True)
    # if visualize:
    #     for idx, data in tqdm(enumerate(loader_infer)):
    #         if idx >= len(pred_result):
    #             break
    #         box_list = [box for box in pred_result[idx][0][0].reshape(-1, 4, 2)]
    #         image = data[0].asnumpy()
    #         print('\n---imageshape:\n', image.shape)
    #         image = np.squeeze(image, axis=0)
    #         # filename = os.path.join(self.args.vis_det_save_dir, os.path.splitext(image_name)[0])
    #         vis_tool = Visualization(VisMode.bbox)
    #         # box_list = [np.array(x).reshape(-1, 2) for x in image_pipeline_res[image_name]]
    #         box_line = vis_tool(image, box_list)
    #         print("\n---box_line:\n", box_line.shape)
    #         print('\n---type(box_line):\n', type(box_line))
    #         print('\nbox_line.shape:\n', box_line.shape)
    #         box_line = box_line.transpose(1, 2, 0)
    #         print('\nbox_line_trans.shape:\n', box_line.shape)
    #         print(box_line)
    #         assert cv2.imwrite(os.path.join(vis_det_save_dir, f'{idx}.jpg'), box_line), f'fail to save image {idx}.jpg'


    if visualize:
        for idx, data in tqdm(enumerate(loader_infer)):
            if idx >= len(pred_result):
                break
            image = data[image_column_idx].asnumpy()
            image = np.squeeze(image, axis=0)
            # assert ('polys' in preds) or ('polygons' in preds), 'Only support detection'
            # gt_img_polys = draw_bboxes(recover_image(img), gt[0].asnumpy())
            pred_img_polys = draw_bboxes(recover_image(image), pred_result[idx][0][0].reshape(-1, 4, 2))
            # TODO: data[0]=img_path is Tensor, need to find elegant way to convert to str
            show_img(pred_img_polys, show=False, save_path=os.path.join(vis_det_save_dir, f'vis_{os.path.basename(str(data[0])[2:-2])}'))

    return pred_result


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation Config', add_help=False)
    parser.add_argument('-c', '--config', type=str, default='../../../configs/det/dbnet/db_r50_icdar15.yaml',
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

    # print(config)

    pred_result = main(config)
    for res in pred_result:
        print('-------->>>>>>')
        # print(res)
        # print(type(res))
        # print(res[0].shape)

