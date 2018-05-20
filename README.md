## Intro

Mask R-CNN is a computer vision model developed by the Facebook AI group that achieves state-of-the-art results on semantic segmentation (object recognition and pixel labeling) tasks. An implementation of the model is made available by Matterport on their github page. The code in their repo works with the MS Coco dataset out of the box, but provides for easy extensibility to any kind of dataset or image segmentation task.

This is a fork of the [matterport/mask_rcnn repo](github.com/matterport/mask-rcnn) that we have set up to integrate with Weights and Biases (wandb). wandb is a cloud interface for tracking model parameters and performance, allowing machine learning teams to coordinate work in a way similar to github. [Here](app.wandb.ai/trentwatson1/mask-rcnn) are the results of our runs. For a more detailed overview of our process and results see our [blog post](medium.com).

## Setup

We have also streamlined the setup process of the original repo to get it up and running quickly on the tensorflow_p36 environment of the AWS Deep Learning AMI (Ubuntu) Version 9.0. To do so, start up an instance with at least 125 GB of storage, ssh into it, and do:

1. `source activate tensorflow_p36`
2. `git clone https://github.com/connorhough/mask_rcnn`
3. `cd mask_rcnn`
4. `pip install cython`
5. `pip install -r requirements.txt`
6. `python setup.py install`
7. `wandb init`
8. `wandb run samples/coco/coco.py train --model=imagenet --dataset=../coco --download=True`

After the first run, use the above command without the `--download=True` argument

The parameter sweep can be run with `bash sweep.sh`
