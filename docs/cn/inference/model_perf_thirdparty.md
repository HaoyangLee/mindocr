## 第三方模型推理精度列表

本文档将给出第三方模型（如PaddleOCR、MMOCR等）在进行[模型转换](convert_tutorial.md)后，使用MindIR格式推理时的精度。

### 1. 文本检测

|             名称             |  模型   |   骨干网络    |    来源    | 测试数据 | recall | precision | f-score |
|:---------------------------:|:-------:|:-----------:|:---------:|:------:|:------:|:---------:|:-------:|
|    ch_pp_server_det_v2.0    |  DBNet  | ResNet18_vd | PaddleOCR | MLT17  | 0.3637 |  0.6340   | 0.4622  |
|       ch_pp_det_OCRv3       |  DBNet  | MobileNetV3 | PaddleOCR | MLT17  | 0.2557 |  0.5021   | 0.3389  |
|       ch_pp_det_OCRv2       |  DBNet  | MobileNetV3 | PaddleOCR | MLT17  | 0.3258 |  0.6318   | 0.4299  |
| ch_pp_mobile_det_v2.0_slim  |  DBNet  | MobileNetV3 | PaddleOCR | MLT17  | 0.2346 |  0.4868   | 0.3166  |
|    ch_pp_mobile_det_v2.0    |  DBNet  | MobileNetV3 | PaddleOCR | MLT17  | 0.2403 |  0.4597   | 0.3156  |
|       en_pp_det_OCRv3       |  DBNet  | MobileNetV3 | PaddleOCR |  IC15  | 0.3866 |  0.4630   | 0.4214  |
|       ml_pp_det_OCRv3       |  DBNet  | MobileNetV3 | PaddleOCR | MLT17  | 0.5992 |  0.7348   | 0.6601  |
| en_pp_det_dbnet_resnet50vd  |  DBNet  | ResNet50_vd | PaddleOCR |  IC15  | 0.8281 |  0.7716   | 0.7989  |
|  en_pp_det_sast_resnet50vd  |  SAST   | ResNet50_vd | PaddleOCR |  IC15  | 0.7463 |  0.9043   | 0.8177  |
| en_pp_det_psenet_resnet50vd | PSENet  | ResNet50_vd | PaddleOCR |  IC15  | 0.7664 |  0.8463   | 0.8044  |
| en_mm_det_dbnetpp_resnet50  | DBNet++ |  ResNet50   |   MMOCR   |  IC15  | 0.8387 |  0.7900   | 0.8136  |
|  en_mm_det_fcenet_resnet50  | FCENet  |  ResNet50   |   MMOCR   |  IC15  | 0.8681 |  0.8074   | 0.8367  |

### 2. 文本识别

|                名称                |  模型   |       骨干网络       |    来源    |       测试数据        | accuracy | norm edit distance |
|:---------------------------------:|:-------:|:------------------:|:---------:|:--------------------:|:--------:|:------------------:|
|       ch_pp_server_rec_v2.0       |  CRNN   |      ResNet34      | PaddleOCR | MLT17 (only Chinese) |  0.4991  |       0.7411       |
|          ch_pp_rec_OCRv3          |  SVTR   | MobileNetV1Enhance | PaddleOCR | MLT17 (only Chinese) |  0.4991  |       0.7535       |
|          ch_pp_rec_OCRv2          |  CRNN   | MobileNetV1Enhance | PaddleOCR | MLT17 (only Chinese) |  0.4459  |       0.7036       |
|       ch_pp_mobile_rec_v2.0       |  CRNN   |    MobileNetV3     | PaddleOCR | MLT17 (only Chinese) |  0.2459  |       0.4878       |
|          en_pp_rec_OCRv3          |  SVTR   | MobileNetV1Enhance | PaddleOCR | MLT17 (only English) |  0.7964  |       0.8854       |
| en_pp_mobile_rec_number_v2.0_slim |  CRNN   |    MobileNetV3     | PaddleOCR | MLT17 (only English) |  0.0164  |       0.0657       |
|   en_pp_mobile_rec_number_v2.0    |  CRNN   |    MobileNetV3     | PaddleOCR | MLT17 (only English) |  0.4304  |       0.5944       |
|     en_pp_rec_crnn_resnet34vd     |  CRNN   |    Resnet34_vd     | PaddleOCR |         IC15         |  0.6635  |       0.8392       |
|   en_pp_rec_rosetta_resnet34vd    | Rosetta |    Resnet34_vd     | PaddleOCR |         IC15         |  0.6428  |       0.8321       |
|      en_pp_rec_vitstr_vitstr      | VITSTR  |       vitstr       | PaddleOCR |         IC15         |  0.6842  |       0.8578       |
|      en_mm_rec_nrtr_resnet31      |  NRTR   |      ResNet31      |   MMOCR   |         IC15         |  0.6726  |       0.8574       |
|    en_mm_rec_satrn_shallowcnn     |  SATRN  |     shallowcnn     |   MMOCR   |         IC15         |  0.7352  |       0.8887       |

请注意，上述模型采用了shape分档，因此该性能仅表示在某些shape下的性能。

### 3. 评估方法

请参考[模型推理精度评估](model_evaluation.md)文档。