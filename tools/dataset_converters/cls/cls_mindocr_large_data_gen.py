import os
from tqdm import tqdm
import shutil
import random
from PIL import Image
import argparse

random.seed(4)

def parse_args():
    parser = argparse.ArgumentParser(description='Text direction classification data generation args', add_help=False)
    parser.add_argument('--data_name', type=str, default='' ,choices=['RCTW17', 'MTWI', 'LSVT'])
    parser.add_argument('--data_root', type=str, default='/home/lihaoyang/dataset/cls')
    # parser.add_argument('--annotation_root', type=str, default='/home/lihaoyang/dataset/cls/anno_LSVT',
    #                     help='Root directory for annotation files')
    # parser.add_argument('--input_img_root', type=str, default='/home/lihaoyang/dataset/cls/LSVT',
    #                     help='Root directory for input images')
    # parser.add_argument('--output_img_root', type=str, default='/home/lihaoyang/dataset/cls/LSVT_ori_and_rot',
    #                     help='Root directory for saving images after processing ')
    parser.add_argument('--angle_num', type=int, default=2,
                        help='angle_num=2: [0, 180], angle_num=4: [0, 90, 180, 270]', choices=[2, 4])
    
    args = parser.parse_args()

    return args

def split_trian_eval(img_names, ratio1=0.9, ratio2=0.05):
    """
    train: [0:ratio1]
    val: [ratio1:ratio1+ratio2]
    eval: [ratio2:1]
    """
    assert (ratio1 + ratio2) < 1.0
    img_names_train, img_names_val, img_names_eval = [], [], []
    for n in img_names:
        i = random.random()
        if i < ratio1:
            img_names_train.append(n)
        elif ratio1 <= i < (ratio1 + ratio2):
            img_names_val.append(n)
        else:
            img_names_eval.append(n)

    return img_names_train, img_names_val, img_names_eval


def rotate_img(input_img_path, output_dir, angles):
    im = Image.open(input_img_path)
    anno_rot = dict()

    for a in angles:
        output_img_path = os.path.join(output_dir, f'{a}_{img_name}')
        if a != 0:
            im = im.rotate(a, expand=True)  # counterclockwise
        im.save(output_img_path)
        anno_rot[f'{a}_{img_name}'] = a
        # rot[f'{a}_{img_name}'] = {'img': im.rotate(a, expand=True), 'angle': a}    # counterclockwise
    return anno_rot

def save_anno(img_names, anno_dict, anno_path):
    with open(anno_path, 'w') as f:
        for n in img_names:
            f.write(f"{n}\t{anno_dict[n]}\n")

if __name__ == '__main__':
    # step 0: args formatting
    args = parse_args()
    if args.angle_num == 2:
        angles = [0, 180]
    elif args.angle_num == 4:
        angles = [0, 90, 180, 270]
    else:
        raise ValueError("args.angle_num must be integer 2 or 4.")

    annotation_root = os.path.join(args.data_root, f'anno_{args.data_name}')
    input_img_root = os.path.join(args.data_root, args.data_name)
    output_img_root = os.path.join(args.data_root, f'{args.data_name}_ori_and_rot')

    os.makedirs(annotation_root, exist_ok=True)

    if os.path.exists(output_img_root):
        shutil.rmtree(output_img_root)
    os.makedirs(output_img_root, exist_ok=False)


    # step 1: load all original image names
    all_input_image_names = os.listdir(input_img_root)
    print("Finished step 1: load all original image names")

    """
    # step 1: load annotation txt
    # annotation = dict()
    # with open(os.path.join(os.path.dirname(annotation_root), f'anno_{args.data_name}.txt'), 'r') as f:
    #     for line in f.readlines():
    #         line = line.strip('\n')
    #         img_name, _ = line.split('\t')
    #         # annotation[img_name] = 0  # orginal images
    # print("Finished step 1: load annotation txt")
    """

    # step 2: rotate and save images
    annotation = dict()
    for img_name in tqdm(all_input_image_names):
        anno_rot_tmp = rotate_img(os.path.join(input_img_root, img_name), output_img_root, angles)
        annotation.update(anno_rot_tmp)
    print("Finished step 2: rotate and save images")


    # step 3: split train/val/eval image names
    img_names_train, img_names_val, img_names_eval = split_trian_eval(annotation.keys(), ratio1=0.9, ratio2=0.05)
    print("Finished step 3: split train/val/eval image names")


    # step 4: save the images name to train/val/eval annotation txt
    save_anno(img_names_train, annotation, os.path.join(annotation_root, f'anno_{args.data_name}_train.txt'))
    save_anno(img_names_val, annotation, os.path.join(annotation_root, f'anno_{args.data_name}_val.txt'))
    save_anno(img_names_eval, annotation, os.path.join(annotation_root, f'anno_{args.data_name}_eval.txt'))
    print("Finished step 4: save the images name to train/val/eval annotation txt")

    print(f"\nRotated images are saved in {os.path.realpath(output_img_root)}.")
    print(f"Train/val/eval label files are saved in  {os.path.realpath(annotation_root)}.")