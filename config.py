"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import numpy as np


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = 'ACNE'

    # Using GPUs. For CPU set USE_GPU = False
    USE_GPU = True
    GPU_IDS = [0]

    # Dataset config
    DATA_BASE_DIR = '../autodl-tmp/ACNE_seg'
    # DATA_BASE_DIR = 'F:/dataset/ACNE_seg'
    IMAGE_SHAPE = [1024, 1024, 3]
    BATCH_SIZE = 4
    NUM_WORKERS = 8

    BBOX_MIN_AREA = 50
    BBOX_MIN_WIDTH = 5
    BBOX_MIN_HEIGHT = 5
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    # Image mean/std (RGB)
    RGB_MEAN = np.array([0.559, 0.411, 0.353])
    RGB_STD = np.array([0.306, 0.246, 0.227])
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Number of classification classes (including background)
    NUM_CLASSES = 11
    BACKBONE_ARCH = 'resnet101'
    BACKBONE_INIT_WEIGHT = 'DEFAULT'  # ['DEFAULT', 'IMAGENET1K_V1', 'IMAGENET1K_V2']
    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200
    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.5
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    SAVE_INTERVAL = 2000
    VALID_INTERVAL = 100
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Inferace
    INFER_WINDOW_SIZE = IMAGE_SHAPE[:2]
    INFER_WINDOW_STRIDES = (960, 960)

    def __init__(self):
        """Set values of computed attributes."""
        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])

    def __str__(self):
        """Stringfy Configuration values."""
        lines = ["\nConfigurations:"]
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                lines.append("{:30} {}".format(a, getattr(self, a)))
        lines.append('\n')
        return '\n'.join(lines)
