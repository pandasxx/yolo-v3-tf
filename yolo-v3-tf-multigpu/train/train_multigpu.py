from net.yolo_top import yolov3
from data.data_pipeline import data_pipeline
from net.config import cfg
import numpy as np
import tensorflow as tf
import os

class YOLO_TRAIN:
    
    def __init__(self, database_path, gpus = [0], num_gpus = 1):
        # init param
        self.model = None
        self.istraining = None
        self.image_resized = cfg.train.image_resized
        self.max_batches = cfg.train.max_batches
        self.gpu_id = gpus
        self.num_gpus = num_gpus
        self.database_path = database_path

        self.mean_loss = []
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
#        imgs, true_boxes = data_pipeline(database_path, cfg.batch_size)
#       self.image_splits = tf.split(imgs, num_gpus)
#        self.label_splits = tf.split(true_boxes, num_gpus)

    def tower_loss(self, scope):
    # 数据集的路径可以在cifar10.py中的tf.app.flags.DEFINE_string中定义
        imgs, true_boxes = data_pipeline(self.database_path, cfg.batch_size)
        istraining = tf.constant(True, tf.bool)
        model = yolov3(imgs, true_boxes, istraining)
        tf.add_to_collection('losses', model.compute_loss())
        losses = tf.get_collection('losses', scope)  # 获取当前GPU上的loss(通过scope限定范围)
        total_loss = tf.add_n(losses, name='total_loss')
        return total_loss

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
    
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
    
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
    
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
    
    def build_model(self):

        with tf.Graph().as_default(), tf.device('/cpu:0'):
            global_step = tf.Variable(0, trainable=False)
            # lr = tf.train.exponential_decay(0.0001, global_step=global_step, decay_steps=2e4, decay_rate=0.1)
            lr = tf.train.piecewise_constant(global_step, self.lr_steps, self.lr_scales)
#            opt = tf.train.AdamOptimizer(learning_rate = lr)
            opt = tf.train.GradientDescentOptimizer(lr)

            tower_grads = []
            for i in range(self.num_gpus):
                with tf.device('/gpu:%d' % self.gpu_id[i]):
                    with tf.name_scope('%s_%d' % ("TOWER", i)) as scope:
                        loss = self.tower_loss(scope)
                        tf.get_variable_scope().reuse_variables()                    
                        grads = opt.compute_gradients(loss)

                        print(loss)
                        print("****************************************")
                        print(type(loss))
                        print("#############################")
                        print(grads)
                        print("****************************************")
                        print(type(grads))
                        print("#############################")

                        tower_grads.append(grads)
                print("create completed!")


            #grads = self.average_gradients(tower_grads)
            self.train_op = opt.apply_gradients(tower_grads[0], global_step=global_step)

            self.saver = tf.train.Saver()

    def train_model(self):   
        gs = 0

        with tf.Graph().as_default(), tf.device('/cpu:0'):
            init = tf.global_variables_initializer()
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                sess.run(init)
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
                    #_, loss_ = sess.run([self.train_op, self.mean_loss])
                    _, = sess.run(self.train_op)
                    if(i % 100 == 0):
                        print(i,': ', loss_)
                    if(i % 1000 == 0):
                        self.saver.save(sess, self.ckpt_dir+'yolov3.ckpt', 
                                   global_step=self.global_step, write_meta_graph=False)