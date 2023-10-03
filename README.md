## Introduction

Mask R-CNN is a cutting-edge deep learning model designed for computer vision tasks, developed by Facebook's AI group. It excels in semantic segmentation tasks, which include object recognition and pixel-level labeling. Matterport has provided an implementation of this model on their [GitHub page](github.com/matterport). The provided code is compatible with the MS Coco dataset—a benchmark dataset for semantic segmentation—right out of the gate, and is designed for easy adaptability to other datasets or image segmentation tasks.

![Instance Segmentation Sample](assets/street.png)

This repository is a fork of the [matterport/mask_rcnn repo](github.com/matterport/mask-rcnn), enhanced to integrate with Weights and Biases (wandb). wandb is a cloud-based interface designed for tracking model parameters and performance, facilitating collaborative efforts among machine learning teams. The results of our model runs can be found [here](https://app.wandb.ai/trentwatson1/mask-rcnn). For an in-depth discussion on our process and findings, refer to our [blog post](https://medium.com/@connorandtrent/mask-r-cnn-hyperparameter-experiments-with-weights-and-biases-bd2319faae26).

## Setup

We've streamlined the setup procedure of the original repository to ensure a swift installation on the tensorflow_p36 environment of AWS Deep Learning AMI (Ubuntu) Version 10.0. To get started, initiate an instance with a minimum of 100 GB storage, SSH into it, and execute the following steps:

1. `source activate tensorflow_p36`
2. `git clone https://github.com/connorhough/mask_rcnn`
3. `cd mask_rcnn`
4. `pip install cython`
5. `pip install -r requirements.txt`
6. `pip install tensorflow-gpu==1.7.0`
7. `python setup.py install`
8. Execute `wandb init`, then follow the on-screen instructions
9. Run `wandb run samples/coco/coco.py train --model=imagenet --dataset=../coco --download=True`

For subsequent runs, use the above command without the `--download=True` argument.

To execute a parameter sweep, run `./sweep.sh`.
