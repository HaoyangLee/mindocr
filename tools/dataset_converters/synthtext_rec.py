import os
import sys
import shutil
from tqdm import tqdm
from time import time
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from infer.text.utils import crop_text_region
# crop_text_region(img, points, box_type="quad", rotate_if_vertical=True) #polygon_type="poly")
# img: [H, W, C]  points: [4, 2], output: [H, W, C]


class SYNTHTEXT_REC_Converter:
    """
    Convert the format of original SynthText dataset:
        images (.jpg) + labels (.mat) --> images cropped by bounding box (.jpg) + labels (.txt)
    """
    def __init__(self, *args):
        self.batch_size = None
        self.image_dir = None
        self.output_path = None

    def convert(self, task='rec', image_dir=None, label_path=None, output_path="./SynthText_crop", batch_size=5000):
        start = time()
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.output_path = output_path
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path, exist_ok=False)

        print("Loading .mat label file of SynthText dataset. It might take a while...")
        mat = loadmat(label_path)

        print("Cropping images and convert labels...")
        self._crop_and_save(mat["imnames"][0], mat["wordBB"][0], mat["txt"][0])
        # with ProcessPoolExecutor(max_workers=8) as pool:
        #     tqdm(pool.map(self._crop_and_save, (mat["imnames"][0], mat["wordBB"][0], mat["txt"][0])), \
        #         total=len(mat['imnames'][0]), desc='Processing data', miniters=10000)
        end = time()
        print("Finished! time cost: {:.2f} minutes".format((end - start)/60.))

    def _crop_and_save(self, imnames, boxes, texts):
        st = 0
        while st < len(imnames):
            print(f"Batch: {st} - {st + self.batch_size}")
            f = open(os.path.join(self.output_path, "gt.txt"), "a")
            for i, imname in tqdm(enumerate(imnames[st:(st + self.batch_size)])):
                i += st
                imname = imname.item()
                ori_img = cv2.imread(os.path.join(self.image_dir, imname))
                save_crop_dir = os.path.join(self.output_path, os.path.dirname(imname))
                os.makedirs(save_crop_dir, exist_ok=True)

                tmp_text = [t for text in texts[i].tolist() for t in text.split()]

                for j, box in enumerate(boxes[i].transpose().reshape(-1, 4, 2)):  # some boxes have (4, 2) shape (no batch dimension)
                    box = box.astype(np.float32)
                    cropped_img = crop_text_region(ori_img, box, box_type="poly")
                    save_crop_path = os.path.join(save_crop_dir, f"{os.path.splitext(os.path.basename(imname))[0]}_crop_{j}.jpg")

                    # Save images and label file            
                    cv2.imwrite(save_crop_path, cropped_img)
                    f.write(f"{os.path.splitext(imname)[0]}_crop_{j}.jpg" + "\t" + tmp_text[j] + "\n")
            f.close()
            st += self.batch_size

# image_dir = "/home/group1/rustam/data/SynthText"
# label_path = "/home/group1/rustam/data/SynthText/gt.mat"  # all images
# label_path = "/home/group1/rustam/data/SynthText/gt_orig_sample.mat"  # 100 images
# label_path = "/home/lihaoyang/code/mindocr_synthtext/mindocr/SynthText/gt_processed.mat"  # 100 images
# output_path = "/home/lihaoyang/code/mindocr_synthtext/mindocr/SynthText_crop_all"
