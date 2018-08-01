from net.yolo_top import yolov3
import numpy as np
import tensorflow as tf
from net.config import cfg
from PIL import Image, ImageDraw, ImageFont
from predict.draw_box import draw_boxes
import matplotlib.pyplot as plt
import os

class YOLO_PREDICT:
    
    def __init__(self, gpu = "0"):
        
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        
        # result
        self.boxes_dict = {}
        self.scores_dict = {}
        self.classes_dict = {}
        
        # clear graph
        tf.reset_default_graph()
        
        # now, in predict mode
        self.istraining = tf.constant(False, tf.bool)
        # which size is the training size
        self.img_size = cfg.data.img_size
        # 
        self.batch_size = cfg.predict.batch_size
        self.scratch = cfg.predict.scratch
        
        self.build_model()

    def build_model(self):
        
        self.img_hw = tf.placeholder(dtype=tf.float32, shape=[2])
        self.imgs_holder = tf.placeholder(tf.float32, 
                                          shape = [None,
                                                   self.img_size[0], 
                                                   self.img_size[1], 
                                                   self.img_size[2]])
        
        self.model = yolov3(self.imgs_holder, None, self.istraining)
        self.boxes, self.scores, self.classes = self.model.pedict(self.img_hw,
                                                                  iou_threshold = cfg.predict.iou_thresh,
                                                                  score_threshold = cfg.predict.score_thresh)
        self.saver = tf.train.Saver()
        self.ckpt_dir = cfg.path.ckpt_dir
        
    def predict_imgs(self, image_data, img_id_list):

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            
            for i, single_image_data in enumerate(image_data):
                boxes_, scores_, classes_ = sess.run([self.boxes, self.scores, self.classes],
                                                     feed_dict={
                                                        self.img_hw:
                                                         [self.img_size[1], 
                                                          self.img_size[0]],
                                                        self.imgs_holder: 
                                                         np.reshape(single_image_data / 255, 
                                                            [1, 
                                                             self.img_size[0], 
                                                             self.img_size[1], 
                                                             self.img_size[2]])})

                self.boxes_dict[img_id_list[i]] = boxes_
                self.scores_dict[img_id_list[i]] = scores_
                self.classes_dict[img_id_list[i]] = classes_