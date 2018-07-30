from net.yolo_top import yolov3
from data.data_pipeline import data_pipeline
from net.config import cfg
import numpy as np
import tensorflow as tf
import os

class YOLO_TRAIN:
    
    def __init__(self, database_path, gpu = "0"):
        
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        
        # init param
        self.model = None
        self.istraining = None
        self.image_resized = cfg.train.image_resized
        self.max_batches = cfg.train.max_batches
        
        self.loss = None
        self.global_step = None
        self.lr_steps = cfg.train.lr_steps
        self.lr_scales = cfg.train.lr_scales
        self.lr = None
        self.optimizer = None
        self.update_op = None
        
        self.saver = None
        self.ckpt_dir = cfg.path.ckpt_dir
        
        # build data
        # from tfrecord format database read imgs and boxes
        self.imgs, self.true_boxes = data_pipeline(database_path, cfg.batch_size)
    
    def build_model(self):
        # set model status
        self.istraining = tf.constant(True, tf.bool)
        # build whole yolov3 model
        self.model = yolov3(self.imgs, self.true_boxes, self.istraining)
    
        # get loss
        self.loss = self.model.compute_loss()
    
        self.global_step = tf.Variable(0, trainable=False)
        # lr = tf.train.exponential_decay(0.0001, global_step=global_step, decay_steps=2e4, decay_rate=0.1)
        self.lr = tf.train.piecewise_constant(self.global_step, 
                                              self.lr_steps, 
                                              self.lr_scales)
    
        # get optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Head")
        # for var in vars_det:
        #     print(var)
        with tf.control_dependencies(self.update_op):
            self.train_op = self.optimizer.minimize(
                self.loss, global_step = self.global_step, var_list=vars_det)
        self.saver = tf.train.Saver()

    def train_model(self):   
        gs = 0
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            if (ckpt and ckpt.model_checkpoint_path):
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                sess.run(tf.assign(self.global_step, gs))
                print('Restore batch: ', gs)
            else:
                print('no checkpoint found')
                sess.run(tf.global_variables_initializer())
            for i in range(gs, self.max_batches):
                _, loss_ = sess.run([self.train_op, self.loss])
                if(i % 100 == 0):
                    print(i,': ', loss_)
                if(i % 1000 == 0):
                    self.saver.save(sess, self.ckpt_dir+'yolov3.ckpt', 
                               global_step=self.global_step, write_meta_graph=False)