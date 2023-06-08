English | [中文](../../cn/datasets/synthtext_rec_CN.md)

# Data Downloading

SynthText is a synthetically generated dataset, in which word instances are placed in natural scene images, while taking into account the scene layout. This tutorial shows you how to convert the SynthText dataset format from images (.jpg) + labels (.mat) to images cropped by bounding box (.jpg) + labels (.txt).

[Paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Gupta16/) | [Download SynthText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c)


Download the `SynthText.zip` file and unzip in `[path-to-data-dir]` folder:
```
path-to-data-dir/
 ├── SynthText/
 │   ├── 1/
 │   │   ├── ant+hill_1_0.jpg
 │   │   └── ...
 │   ├── 2/
 │   │   ├── ant+hill_4_0.jpg
 │   │   └── ...
 │   ├── ...
 │   └── gt.mat
```

# Data Conversion

Run the following command to convert the SynthText dataset from images (.jpg) + labels (.mat) to images cropped by bounding box (.jpg) + labels (.txt).

```shell
python tools/dataset_converters/convert.py 
        --dataset_name=synthtext_rec \
        --task=rec \
        --image_dir=/path-to-data-dir/SynthText \
        --label_dir=/path-to-data-dir/SynthText/gt.mat \
        --output_path=/path-to-data-dir/SynthText_crop
```

After the data conversion above, you will get the output data organized in the following file structure. Furthermore, you can refer to [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#when-you-need-to-train-on-your-own-dataset-or-non-latin-language-datasets) to convert images cropped by bounding box (.jpg) + labels (.txt) to LMDB format, which can be used for training text recognition models more efficiently.

```
output-path-to-data-dir/
 ├── SynthText/
 │   ├── 1/
 │   │   ├── ant+hill_1_0_crop_0.jpg
 │   │   ├── ant+hill_1_0_crop_1.jpg
 │   │   └── ...
 │   ├── 2/
 │   │   ├── ant+hill_4_0_crop_0.jpg
 │   │   ├── ant+hill_4_0_crop_1.jpg
 │   │   └── ...
 │   ├── ...
 │   └── gt.txt
```

[Back to README](../../../tools/dataset_converters/README.md)