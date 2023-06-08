import os
import sys
import shutil
from time import time

import cv2
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from infer.text.utils import crop_text_region
# crop_text_region(img, points, box_type="quad", rotate_if_vertical=True) #polygon_type="poly")
# img: [H, W, C]  points: [4, 2], output: [H, W, C]


# def _check(boxes, crops):
#     # Check whether the cropping is correct.
#     num_boxes = 0
#     for k, b in enumerate(boxes):
#         if b.ndim == 2:
#             num_boxes += 1
#         elif b.ndim == 3:
#             num_boxes += b.shape[2]
#         else:
#             raise ValueError(f"Dimension of box {k}is not 2 or 3. Please check the validity of the box.")
#     assert len(crops) == num_boxes, "Number of cropped images is not equal to number of bounding boxes. Please check the validity of input dataset."
#     print(len(crops))
#     print(num_boxes)


if __name__ == "__main__":
    start = time()
    img_root = "/home/group1/rustam/data/SynthText"
    label_path = "/home/group1/rustam/data/SynthText/gt.mat"
    # label_path = "/home/group1/rustam/data/SynthText/gt_orig_sample.mat"
    # label_path = "/home/lihaoyang/code/mindocr_synthtext/mindocr/SynthText/gt_processed.mat"
    output_img_root = "/home/lihaoyang/code/mindocr_synthtext/mindocr/SynthText_crop_all"
    if os.path.exists(output_img_root):
        shutil.rmtree(output_img_root)
    os.makedirs(output_img_root, exist_ok=False)

    print("Loading .mat label file of SynthText dataset. It might take a while...")
    mat = loadmat(label_path)
    imnames, boxes, texts = mat["imnames"][0], mat["wordBB"][0], mat["txt"][0]

    # crop text regions
    # crops = []
    print("Cropping images and convert labels...")
    batch_size = 5000
    st = 0
    while st < len(imnames):
        print(f"Batch: {st} - {st+batch_size}")
        f = open(os.path.join(output_img_root, "gt.txt"), "a")
        for i, imname in tqdm(enumerate(imnames[st:(st+batch_size)])):
            i += st
            imname = imname.item()
            ori_img = cv2.imread(os.path.join(img_root, imname))
            save_crop_dir = os.path.join(output_img_root, os.path.dirname(imname))
            os.makedirs(save_crop_dir, exist_ok=True)

            tmp_text = [t for text in texts[i].tolist() for t in text.split()]    # TODO: check the correctness of texts order

            for j, box in enumerate(boxes[i].transpose().reshape(-1, 4, 2)):  # some boxes have (4, 2) shape (no batch dimension)
                box = box.astype(np.float32)
                cropped_img = crop_text_region(ori_img, box, box_type="poly")
                # crops.append(cropped_img)
                save_crop_path = os.path.join(save_crop_dir, f"{os.path.splitext(os.path.basename(imname))[0]}_crop_{j}.jpg")
                
                # Save images and label file            
                cv2.imwrite(save_crop_path, cropped_img)
                f.write(f"{os.path.splitext(imname)[0]}_crop_{j}.jpg" + "\t" + tmp_text[j] + "\n")
        f.close()
        st += batch_size

    end = time()
    # _check(boxes, crops)

    print("Finished! time cost:", end - start)