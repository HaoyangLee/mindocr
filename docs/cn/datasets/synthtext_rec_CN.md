[English](../../en/datasets/synthtext_rec.md) | 中文

# 数据下载

SynthText是一个合成生成的数据集，其中单词实例被放置在自然场景图像中，并考虑了场景布局。本数据集转换流程将把SynthText数据集的图片(.jpg)+标注文件(.mat)转换为根据标注bounding box裁剪后的图片(.jpg)+标注文件(.txt)。

[论文](https://www.robots.ox.ac.uk/~vgg/publications/2016/Gupta16/) | [下载SynthText](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c)

下载`SynthText.zip`文件并解压缩到`[path-to-data-dir]`文件夹中：
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
# 数据转换

执行以下命令，把SynthText数据集的图片(.jpg)+标注文件(.mat)转换为根据标注bounding box裁剪后的图片(.jpg)+标注文件(.txt)。

```shell
python tools/dataset_converters/convert.py 
        --dataset_name=synthtext_rec \
        --task=rec \
        --image_dir=/path-to-data-dir/SynthText \
        --label_dir=/path-to-data-dir/SynthText/gt.mat \
        --output_path=/path-to-data-dir/SynthText_crop
```

完成上述数据格式转换后，输出数据的文件夹结构如下。你可以参考[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark#when-you-need-to-train-on-your-own-dataset-or-non-latin-language-datasets)进一步将图片(.jpg)和标注(.txt)数据转换为LMDB格式，用于更高效的文本识别模型训练。

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

[返回](../../../tools/dataset_converters/README_CN.md)