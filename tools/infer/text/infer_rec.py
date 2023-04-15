'''
Recognition model inference
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
    # ms.amp.auto_mixed_precision(network, amp_level='O0')

    # postprocess, metric
    postprocessor = build_postprocess(cfg.postprocess)
    # postprocess network prediction
    # metric = build_metric(cfg.metric)

    # net_evaluator = Evaluator(network, None, postprocessor, [metric])

    # log
    print('=' * 40)
    print(
        f'Num batches: {num_batches}\n'
    )
    if 'name' in cfg.model:
        print(f'Model: {cfg.model.name}')
    else:
        print(f'Model: {cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}')
    print('=' * 40)

    # measures = net_evaluator.eval(loader_eval)
    # print('Performance: ', measures)

    # model = ms.Model(network)
    # column_names = cfg.eval.dataset.output_columns  # all_columns=['lmdb_idx', 'file_idx', 'img_lmdb', 'label', 'image', 'length', 'text_seq', 'text_length', 'text_padded', 'valid_ratio']
    # print('????column_names:', column_names)
    image_column_idx = dataset_column_names.index('image')
    pred_result = []
    for idx, data in tqdm(enumerate(loader_infer)):
        if idx > 1:
            break
        print('\n#-----check lmdb_idx and file_idx:-----#\n')
        # print('\nlmdb_idx:\n', data[0])
        # print('\nfile_idx:\n', data[1])
        # print('\nlabel:\n', data[3])
        # print('\nimage:\n', data[4])
        print('\nimage shape:\n', data[image_column_idx].shape)
        # for i, d in enumerate(data):
        #     if column_names[i] == 'img_lmdb':
        #         continue
        #     print(f'\n---------{column_names[i]}\n', d)
        pred = network(data[image_column_idx])
        pred = postprocessor(pred)  # [bboxes, scores], shape=[(N, K, 4, 2), (N, K)]
        pred_result.append(pred)

    # TODO: reshape(-1, 2) if polygon
    # TODO: batch_size issue, need to squeeze axis=0 now
    # TODO: no shuffle in cfg.eval, check and match the order of img name of loader_infer
    vis_rec_save_dir = '/home/mindspore/lhy/vis/rec_jpg'
    os.makedirs(vis_rec_save_dir, exist_ok=True)
    visualize = True

    if visualize:
        #### LMDB dataset
        # f = open(os.path.join(f'{vis_rec_save_dir}', 'text_label.txt'), 'w')
        # for idx, data in tqdm(enumerate(loader_infer)):
        #     if idx >= len(pred_result):
        #         break
        #     for i, single_img in enumerate(data[image_column_idx]):    # iterate the images inside one batch
        #         lmdb_i, file_i = data[0][i], data[1][i]
        #         text_predict = pred_result[idx]['texts'][i]
        #         img_name = f'vis_{lmdb_i}_{file_i}.jpg'
        #         single_img = single_img.asnumpy()
        #         show_img(recover_image(single_img), show=False,
        #              save_path=os.path.join(vis_rec_save_dir, img_name))
        #         f.write(f'{img_name}\t{text_predict}\n')
        # f.close()

        #### normal image dataset
        f = open(os.path.join(f'{vis_rec_save_dir}', 'text_predict.txt'), 'w')
        for idx, data in tqdm(enumerate(loader_infer)):
            if idx >= len(pred_result):
                break
            for i, single_img in enumerate(data[image_column_idx]):    # iterate the images inside one batch
                text_predict = pred_result[idx]['texts'][i]
                single_img = single_img.asnumpy()
                img_save_path = os.path.join(vis_rec_save_dir, f'vis_{os.path.basename(data[0].asnumpy()[i])}')
                show_img(recover_image(single_img), show=False, save_path=img_save_path)
                f.write(f'{img_save_path}\t{text_predict}\n')
        f.close()

    return pred_result


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

    # print(config)

    pred_result = main(config)
    for res in pred_result:
        print('!!!main function')
        print('texts:\n', res['texts'], '\nlength:', len(res['texts']))
        print('confs:\n', res['confs'], '\nlength:', len(res['confs']))
        # for k, v in res.items():
        #     print('>>>>>>>>>>>>:\n', k,'\n', v,'\n')
        #     print('-------')

