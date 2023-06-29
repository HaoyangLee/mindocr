## 推理 - MindOCR模型支持列表

MindOCR推理支持训练端ckpt导出的模型，本文档展示了已适配的模型列表。

请[自行导出](https://github.com/mindspore-lab/mindocr/blob/main/tools/export.py)或下载已预先导出的MindIR文件，并参考[模型转换教程](convert_tutorial.md)，再进行推理。

### 1. 文本检测

| 模型                                                                             | 骨干网络     | 语言 | 配置文件                                                                                                                          | 下载                                                                                                              |
|:--------------------------------------------------------------------------------|:------------|:----|:--------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|
| [DBNet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)   | MobileNetV3 | en  | [db_mobilenetv3_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_mobilenetv3_icdar15.yaml) | [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_mobilenetv3-62c44539-f14c6a13.mindir)         |
|                                                                                 | ResNet-18   | en  | [db_r18_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_r18_icdar15.yaml)                 | [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet18-0c0c4cfa-cf46eb8b.mindir)            |
|                                                                                 | ResNet-50   | en  | [db_r50_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db_r50_icdar15.yaml)                 | [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnet_resnet50-c3a4aa24-fbf95c82.mindir)            |
| [DBNet++](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet) | ResNet-50   | en  | [db++_r50_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet/db++_r50_icdar15.yaml)             | [mindir](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnetpp_resnet50-068166c2-9934aff0.mindir)          |
| [EAST](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east)     | ResNet-50   | en  | [east_r50_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east/east_r50_icdar15.yaml)              | [mindir](https://download.mindspore.cn/toolkits/mindocr/east/east_resnet50_ic15-7262e359-5f05cd42.mindir)         |
| [PSENet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) | ResNet-152  | en  | [pse_r152_icdar15.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet/pse_r152_icdar15.yaml)            | [mindir](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ic15-6058a798-0d755205.mindir)    |
|                                                                                 | ResNet-152  | ch  | [pse_r152_ctw1500.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet/pse_r152_ctw1500.yaml)            | [mindir](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ctw1500-58b1b1ff-b95c7f85.mindir) |

### 2. 文本识别

| 模型                                                                         | 骨干网络     | 字典文件                                                                                           | 语言 | 配置文件                                                                                                            | 下载                                                                                                     |
|:----------------------------------------------------------------------------|:------------|:-------------------------------------------------------------------------------------------------|:----|:-------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------|
| [CRNN](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn) | VGG7        | Default                                                                                          | en  | [crnn_vgg7.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_vgg7.yaml)               | [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_vgg7-ea7e996c-573dbd61.mindir)        |
|                                                                             | ResNet34_vd | Default                                                                                          | en  | [crnn_resnet34.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_resnet34.yaml)       | [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34-83f37f07-eb10a0c9.mindir)    |
|                                                                             | ResNet34_vd | [ch_dict.txt](https://github.com/mindspore-lab/mindocr/tree/main/mindocr/utils/dict/ch_dict.txt) | ch  | [crnn_resnet34_ch.yaml](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn/crnn_resnet34_ch.yaml) | [mindir](https://download.mindspore.cn/toolkits/mindocr/crnn/crnn_resnet34_ch-7a342e3c-105bccb2.mindir) |