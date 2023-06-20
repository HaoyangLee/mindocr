import os
import shutil
import random

from PIL import Image

def split_trian_eval(rotated_imgs, ratio=0.8):
    rotated_imgs_train, rotated_imgs_eval = {}, {}
    for img_name, img in rotated_imgs.items():
        i = random.random()
        if i < ratio:
            rotated_imgs_train[img_name] = img
        else:
            rotated_imgs_eval[img_name] = img

    return rotated_imgs_train, rotated_imgs_eval


def data_gen(img_root, img_name, angles):
    img_path = os.path.join(img_root, img_name)
    im = Image.open(img_path)
    rot = dict()
    rot[f'{0}_{img_name}'] = {'img': im, 'angle': 0}
    for a in angles:
        rot[f'{a}_{img_name}'] = {'img': im.rotate(a, expand=True), 'angle': a}    # counterclockwise
    return rot

if __name__ == '__main__':
    # angles = [90, 180, 270]
    angles = [180]
    img_root = '/home/lihaoyang/dataset/crnn_jpg'
    if os.path.isdir(img_root):
        all_img_names = os.listdir(img_root)
    else:
        raise ValueError("Invalid image root directory.")

    rotated_imgs = dict()
    for img_name in all_img_names:
        rot = data_gen(img_root, img_name, angles)
        rotated_imgs.update(rot)

    # save rotated images
    save_img_root = '/home/lihaoyang/dataset/crnn_jpg_rot'
    if os.path.exists(save_img_root):
        shutil.rmtree(save_img_root)
    os.makedirs(os.path.join(save_img_root, 'train/images'), exist_ok=False)
    os.makedirs(os.path.join(save_img_root, 'eval/images'), exist_ok=False)

    rotated_imgs_train, rotated_imgs_eval = split_trian_eval(rotated_imgs)

    with open(os.path.join(save_img_root, 'train/cls_gt.txt'), 'w') as f:
        for img_name, img in rotated_imgs_train.items():
            img['img'].save(os.path.join(save_img_root, 'train/images', img_name))
            f.write(f"{img_name}\t{img['angle']}\n")
    with open(os.path.join(save_img_root, 'eval/cls_gt.txt'), 'w') as f:
        for img_name, img in rotated_imgs_eval.items():
            img['img'].save(os.path.join(save_img_root, 'eval/images', img_name))
            f.write(f"{img_name}\t{img['angle']}\n")

    print(f"Rotated images and angle label file are saved in {os.path.realpath(save_img_root)}")
    # rotated_imgs[0].show()