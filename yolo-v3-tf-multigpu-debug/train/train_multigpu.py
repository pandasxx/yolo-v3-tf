from net.yolo_top import yolov3
from data.data_pipeline import data_pipeline
from net.config import cfg
import numpy as np
import time
import tensorflow as tf
import os

def tower_loss(scope, imgs, true_boxes, istraining):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        images: Images. 4D tensor of shape [batch_size, height, width, 3].
        labels: Labels. 1D tensor of shape [batch_size].
    Returns:
        Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    model = yolov3(imgs, true_boxes, istraining)
    _ = model.compute_loss()

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)
        
    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    
    return total_loss

def average_gradients(tower_grads):
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


def train(num_gpus):

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.Variable(0, trainable=False)

        #lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, 
        #                                 global_step,
        #                                 decay_steps,
        #                                 LEARNING_RATE_DECAY_FACTOR,
        #                                 staircase=True)

        lr = tf.train.piecewise_constant(global_step, 
                                         cfg.train.lr_steps,
                                         cfg.train.lr_scales)
        
        # Create an optimizer that performs gradient descent.
        
#        opt = tf.train.GradientDescentOptimizer(lr)
        
        opt = tf.train.AdamOptimizer(learning_rate = lr)
        
        tower_grads = []
        
        imgs, true_boxes = data_pipeline(
            [cfg.path.train_data_path, cfg.path.train_data_path], cfg.batch_size)
        
        print(imgs.shape)
        print(true_boxes.shape)
        
        imgs = tf.reshape(imgs, 
                          [cfg.batch_size, imgs.shape[1], imgs.shape[2], imgs.shape[3]])
        true_boxes = tf.reshape(true_boxes, 
                                [cfg.batch_size, true_boxes.shape[1], true_boxes.shape[2]])
        
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [imgs, true_boxes], capacity=2 * num_gpus)
        
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        istraining = tf.constant(True, tf.bool)
                        
                        imgs_batch, true_boxes_batch = batch_queue.dequeue()
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        
                        loss = tower_loss(scope, 
                                          imgs_batch, 
                                          true_boxes_batch,
                                          istraining)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)
                        
                        print('ops________________________________________________')
                        for i in grads:
                            print(i[0])
                        print ('variables________________________________________________')
                        for i in grads:
                            print (i[1])

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.

        grads = average_gradients(tower_grads)

        print('average ops________________________________________________')
        for i in grads:
            print(i[0])
        print ('average variables________________________________________________')
        for i in grads:
            print (i[1])

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        
        #update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Head")
        
        #with tf.control_dependencies(update_op):
        #    apply_gradient_op = opt.minimize(loss, 
        #                                     global_step = global_step, 
        #                                     var_list = vars_det)
        
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Group all updates to into a single train op.
        train_op = apply_gradient_op

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        
        gs = 0
        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))
        
        
        ckpt = tf.train.get_checkpoint_state(cfg.path.ckpt_dir)
        if (ckpt and ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            sess.run(tf.assign(global_step, gs))
            print('Restore batch: ', gs)
        else:
            print('no checkpoint found')
            sess.run(init)
        
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        
        for i in range(gs, cfg.train.max_batches):
            #start_time = time.time()
            _ = sess.run(train_op)
            #duration = time.time() - start_time
            #print(duration)
            if(i % 10 == 0):
                loss_ = sess.run(loss)
                print(i,': ', loss_)
            if(i % 2000 == 0):
                saver.save(sess, 
                           cfg.path.ckpt_dir + 'yolov3.ckpt', 
                           global_step=global_step, 
                           write_meta_graph=False)

        print("Complete!!")