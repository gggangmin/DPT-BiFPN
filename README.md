# Vision Transformers for Dense Prediction with BiFPN

[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html)

This project reproduces the Dense prediction transformer based on paper and proposes ideas to improve it.

<p align="center"><img src = "DPT-BiFPN.PNG"></p>

DPT is used as a backbone for the dense prediction task to produce an improved global reactive feature. For dense prediction task, FPN mixes the features of several levels produced based on it. This project proposes to effectively mix the features of DPT through BiFPN to perform the dense prediction task.

## DPT followed by [paper](https://arxiv.org/abs/2103.13413)

### Setup

[git link](https://github.com/isl-org/DPT) for setup

### Usage

./DPT/train_monodepth.ipynb attempted to implement the training process of the model. For the loss function of this paper, the scale and shift invariant loss function of [paper](https://ieeexplore.ieee.org/abstract/document/9178977/) was used.

### FOD followed by [git](https://github.com/antocad/FocusOnDepth)

### Setup

[git link](https://github.com/antocad/FocusOnDepth) for setup

### Usage

./FOD/train.py attempted to implement the training process of the model. For the loss function of this paper, the scale and shift invariant loss function of [paper](https://ieeexplore.ieee.org/abstract/document/9178977/) was used and BiFPN of [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html) was used.