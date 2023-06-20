import os
import shutil
import random
import argparse
from tqdm import tqdm

from PIL import Image

random.seed(2)

def parse_args():
    parser = argparse.ArgumentParser(description='Text direction classification data generation args', add_help=False)
    # parser.add_argument('--annotation_root', type=str, default='/home/lihaoyang/dataset/cls',
    #                     help='Root directory for annotation files')    
    parser.add_argument('--input_img_root', type=str, default='/home/lihaoyang/dataset/cls/RCTW-17',
                        help='Root directory for input images')
    parser.add_argument('--output_img_root', type=str, default='/home/lihaoyang/dataset/cls/RCTW-17_mindcv',
                        help='Root directory for saving images after processing ')
    parser.add_argument('--angle_num', type=int, default=2,
                        help='angle_num=2: [0, 180], angle_num=4: [0, 90, 180, 270]', choices=[2, 4])
    
    args = parser.parse_args()

    return args

def get_split_dir(train_ratio=0.9, val_ratio=0.05):
    """
    train: [0:train_ratio]
    val: [train_ratio:train_ratio+val_ratio]
    test: [val_ratio:1]
    """
    assert (train_ratio + val_ratio) < 1.0
    i = random.random()
    if i < train_ratio:
        split_dir = 'train'
    elif train_ratio <= i < (train_ratio + val_ratio):
        split_dir = 'val'
    else:
        split_dir = 'test'

    return split_dir


def data_gen(img_root, save_root, img_name, angles):
    img_path = os.path.join(img_root, img_name)
    im = Image.open(img_path)

    count = dict()
    for a in angles:
        split_dir = get_split_dir()
        label_dir = str(a)
        split_label = os.path.join(split_dir, label_dir)
        save_path = os.path.join(save_root, split_dir, label_dir, f'{a}_{img_name}')
        if a != 0:
            im = im.rotate(a, expand=True)  # counterclockwise
        im.save(save_path)

        count[split_label] = count.get(split_label, 0) + 1

    return count

if __name__ == '__main__':
    args = parse_args()
    if args.angle_num == 2:
        angles = [0, 180]
    elif args.angle_num == 4:
        angles = [0, 90, 180, 270]
    else:
        raise ValueError("args.angle_num must be integer 2 or 4.")

    # step 1: create directory structure like ImageNet
    if os.path.exists(args.output_img_root):
        shutil.rmtree(args.output_img_root)
    splits = ['train', 'val', 'test']
    labels = list(map(str, angles))
    for s in splits:
        for l in labels:
            os.makedirs(os.path.join(args.output_img_root, s, l), exist_ok=False)
    print("Finished step 1: create directory structure like ImageNet")

    # step 2: rotate, split and save images to correct directories
    if os.path.isdir(args.input_img_root):
        all_img_names = os.listdir(args.input_img_root)
    else:
        raise ValueError("Invalid image root directory.")

    img_split_distribution = dict()
    for img_name in tqdm(all_img_names):
        count = data_gen(args.input_img_root, args.output_img_root, img_name, angles)
        for k, v in count.items():
            img_split_distribution[k] = img_split_distribution.get(k, 0) + v

    print("Finished step 2: rotate, split and save images to correct directories")

    print(f"\nRotated images are saved in {os.path.realpath(args.output_img_root)}.")
    print(f"\nTotal number of original images: {len(all_img_names)}.\nNumber of images in each directory:")
    for d, n in img_split_distribution.items():
        print(f"{d}:\t{n}")
    # for d, n in img_split_distribution.items():
    #     tmp_count = 0
    #     for s in splits:
    #         if d.starswith(s):
    #             tmp_count += n
    #     print(f"Number of images in dir `{s}`")