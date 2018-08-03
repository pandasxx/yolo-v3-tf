from easydict import EasyDict as edict
import numpy as np

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

__C.anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
__C.classes = 20
__C.num = 9
__C.num_anchors_per_layer = 3
__C.batch_size = 6
__C.scratch = False
__C.names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#
# Training options
#
__C.train = edict()

__C.train.ignore_thresh = .5
__C.train.momentum = 0.9
__C.train.decay = 0.0005
__C.train.learning_rate = 0.0001
__C.train.max_batches = 50200
__C.train.lr_steps = [40000, 45000]
__C.train.lr_scales = [1e-4, 1e-5, 1e-6]
__C.train.max_truth = 30
__C.train.mask = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
__C.train.image_resized = 608   # { 320, 352, ... , 608} multiples of 32
__C.train.random = 0

#
# Prediction options
#
__C.predict = edict()

__C.predict.batch_size = 1
__C.predict.scratch = True
__C.predict.iou_thresh = 0.3
__C.predict.score_thresh = 0.3

#
# image process options
#
__C.preprocess = edict()
__C.preprocess.angle = 0
__C.preprocess.saturation = 1.5
__C.preprocess.exposure = 1.5
__C.preprocess.hue = .1
__C.preprocess.jitter = .3
__C.preprocess.random = 1

#
# image format
#
__C.data = edict()
__C.data.img_size = [608, 608, 3]

#
# file path
#

__C.path = edict()
__C.path.train_data_path = "enhancement01_608.tfrecords"
__C.path.darknet_weights_path = "../darknet53.conv.74.npz"
__C.path.ckpt_dir = './ckpt/'
