import sys
import time
import numpy as np
import scipy
import imgaug  # https://github.com/aleju/imgaug (pip3 install imageaug)
import wandb
import matplotlib.pyplot as plt
import keras
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")


# sys.path.append(ROOT_DIR)  # To find local version of the library

# from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import load_image_gt
from mrcnn import model as modellib, utils
# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"


run = wandb.init()
wandb.config.hello = 'world'
wandb.config.hi = 'there'
print(wandb.config)


############################################################
#  Configurations
############################################################

'''
class CocoConfig(mrcnn_config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

    # Halve STEPS_PER_EPOCH to speed up training time for the sake of demonstration
    STEPS_PER_EPOCH = 500

    # MODEL TUNING
    if os.environ.get('BACKBONE'):
        BACKBONE = os.environ.get('BACKBONE')
    if os.environ.get('GRADIENT_CLIP_NORM'):
        GRADIENT_CLIP_NORM = float(os.environ.get('GRADIENT_CLIP_NORM'))
    if os.environ.get('LEARNING_RATE'):
        LEARNING_RATE = float(os.environ.get('LEARNING_RATE'))
    if os.environ.get('WEIGHT_DECAY'):
        WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY'))


    def get_config_dict(self):
        """Return Configuration values as a dictionary for the sake of syncing with wandb"""
        d = {}
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                d[a] = getattr(self, a)
        return d
'''


############################################################
# WANDB
############################################################

# _config = CocoConfig()

# config_dict = _config.get_config_dict()
# configs_of_interest = ['BACKBONE', 'GRADIENT_CLIP_NORM', 'LEARNING_MOMENTUM', 'LEARNING_RATE',
#                        'WEIGHT_DECAY', 'STEPS_PER_EPOCH']
# run.history.row.update({k: config_dict[k] for k in configs_of_interest})

'''
def fig_to_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    return buf

class ImageCallback(keras.callbacks.Callback):
    def __init__(self, run, dataset_val, dataset_train):
        self.run = run
        self.dataset_val = dataset_val
        self.dataset_train = dataset_train
        self.image_ids = dataset_val.image_ids[:3]

    def label_image(self, image_id):
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(
            self.dataset_val, _config, image_id, use_mini_mask=False)
        _, ax = plt.subplots(figsize=(16, 16))
        display_instances(
            original_image,
            gt_bbox,
            gt_mask,
            gt_class_id,
            self.dataset_train.class_names,
            figsize=(
                16,
                16),
            ax=ax)
        return fig_to_array(ax.figure)

    def on_epoch_end(self, epoch, logs):
        print("Uploading images to wandb")
        labeled_images = [self.label_image(i) for i in self.image_ids]
        self.run.history.row["img_segmentations"] = [
            wandb.Image(
                scipy.misc.imresize(img, 50),
                caption="Caption",
                mode='RGBA') for img in labeled_images]

class PerformanceCallback(keras.callbacks.Callback):
    def __init__(self, run):
        self.run = run
    def on_epoch_end(self, epoch, logs):
        print("Uploading metrics to wandb")
        self.run.history.row.update(logs)
        self.run.history.add()
'''

