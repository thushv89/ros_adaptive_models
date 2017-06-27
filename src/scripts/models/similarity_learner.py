import numpy as np
import tensorflow as tf
import logging
import sys
from math import ceil
import os
from PIL import Image
import config

logging_level = logging.DEBUG
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

logger = logging.getLogger('Logger')
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter(logging_format))
console.setLevel(logging_level)
fileHandler = logging.FileHandler('main.log', mode='w')
fileHandler.setFormatter(logging.Formatter(logging_format))
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.addHandler(fileHandler)

graph = tf.get_default_graph()
sess = tf.InteractiveSession(graph=graph)

def build_input_pipeline(filenames,batch_size):
    global sess, graph
    global logger
    logger.info('Received filenames: %s', filenames)
    with sess.as_default() and graph.as_default() and tf.name_scope('preprocess'):
        # FIFO Queue of file names
        # creates a FIFO queue until the reader needs them
        filename_queue = tf.train.string_input_producer(filenames, capacity=10, shuffle=False)

        # Reader which takes a filename queue and read() which outputs data one by one
        reader = tf.WholeFileReader()
        _, image_buffer = reader.read(filename_queue, name='image_read_op')

        # return uint8
        dec_image = tf.image.decode_png(contents=image_buffer, channels=3, name='decode_jpg')
        # convert to float32
        float_image = tf.image.convert_image_dtype(dec_image, dtype=tf.float32, name='float_image')
        # resize image

        # standardize image
        std_image = tf.image.per_image_standardization(dec_image)
        # https://stackoverflow.com/questions/37126108/how-to-read-data-into-tensorflow-batches-from-example-queue

        # The batching mechanism that takes a output produced by reader (with preprocessing) and outputs a batch tensor
        # [batch_size, height, width, depth] 4D tensor
        image_batch = tf.train.batch([std_image], batch_size=batch_size, capacity=10, name='image_batch')

        # to use record reader we need to use a Queue either random

    print('Preprocessing done\n')
    return image_batch


def build_tensorflw_variables(variable_shapes):
    '''
    Build the required tensorflow variables to populate the VGG-16 model
    All are initialized with zeros (initialization doesn't matter in this case as we assign exact values later)
    :param variable_shapes: Shapes of the variables
    :return:
    '''
    global logger,sess,graph

    logger.info("Building Tensorflow Variables (Tensorflow)...")
    with sess.as_default and graph.as_default():
        for si,scope in enumerate(config.TF_SCOPES):
            with tf.variable_scope(scope) as sc:

                # Try Except because if you try get_variable with an intializer and
                # the variable exists, you will get a ValueError saying the variable exists
                #
                try:
                    weights = tf.get_variable(config.TF_WEIGHTS_STR, config.TF_VAR_SHAPES[si],
                                              initializer=tf.truncated_normal(config.TF_VAR_SHAPES[si],stddev=0.02,dtyple=tf.float32))
                    bias = tf.get_variable(config.TF_BIAS_STR, config.TF_VAR_SHAPES[si][-1],
                                           initializer = tf.constant_initializer(0.001))
                    with tf.variable_scope(config.TF_DECONV_STR) as deconv_sc:
                        deconv_bias = tf.get_variable(config.TF_BIAS_STR, config.TF_VAR_SHAPES[si][-2],
                                                      initializer = tf.constant_initializer(0.001))
                    sess.run(tf.variables_initializer([weights,bias]))

                except ValueError:
                    logger.debug('Variables in scope %s already initialized\n'%scope)


def infererence(tf_inputs):
    '''
    Inferencing the model. The input (tf_inputs) is propagated through convolutions poolings
    fully-connected layers to obtain the final softmax output
    :param tf_inputs: a batch of images (tensorflow placeholder)
    :return:
    '''
    global logger

    with tf.name_scope('forward'):
        for si, scope in enumerate(config.TF_SCOPES):
            with tf.variable_scope(scope,reuse=True) as sc:
                weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)

                if 'fc' not in scope:
                    if si == 0:
                        h = tf.nn.relu(tf.nn.conv2d(tf_inputs,weight,strides=config.TF_STRIDES[sc],padding='SAME')+bias,name='hidden')
                    else:
                        h = tf.nn.relu(tf.nn.conv2d(h, weight, strides=config.TF_STRIDES[sc], padding='SAME') + bias,
                                       name='hidden')
                else:
                    # Reshaping required for the first fulcon layer
                    if scope == 'fc1':
                        h_shape = h.get_shape().as_list()
                        h = tf.reshape(h,[h_shape[0], h_shape[1] * h_shape[2] * h_shape[3]])
                    h = tf.nn.relu(tf.matmul(h, weight) + bias, name= 'hidden')

    with tf.name_scope('backward'):
        for si, scope in enumerate(reversed(config.TF_SCOPES)):
            with tf.variable_scope(scope, reuse=True) as sc:
                weight = tf.get_variable(config.TF_WEIGHTS_STR)
                with tf.variable_scope(config.TF_DECONV_STR, reuse=True) as sc:
                    deconv_bias = tf.get_variable(config.TF_BIAS_STR)

                if 'fc' not in scope:
                    if si == len(config.TF_SCOPES)-1:
                        out = tf.nn.sigmoid(
                            tf.nn.conv2d_transpose(h, weight, strides=config.TF_STRIDES[sc], padding='SAME') + deconv_bias,
                            name='hidden')
                    else:
                        h = tf.nn.relu(tf.nn.conv2d_transpose(h, weight, strides=config.TF_STRIDES[sc], padding='SAME') + deconv_bias,
                                       name='hidden')
                else:
                    # Reshaping required for the first fulcon layer
                    if scope == 'fc1':
                        h = tf.nn.relu(tf.matmul(h, tf.transpose(weight)) + deconv_bias, name='hidden')
                        h = tf.reshape(h, [h_shape[0], h_shape[1], h_shape[2], h_shape[3]])

                    h = tf.nn.relu(tf.matmul(h, weight) + deconv_bias, name='hidden')

    return out


def calculate_loss(tf_inputs):

    out = infererence(tf_inputs)

    loss = tf.reduce_mean(tf.reduce_sum((tf_inputs - out)**2,axis=[1,2,3]))

    return loss

def optimize_model(loss):

    optimize = tf.train.MomentumOptimizer(momentum=0.9,learning_rate=0.01).minimize(loss)
    return optimize


if __name__ == '__main__':

    filenames = ['..'+os.sep+'sample'+os.sep+'image_%d.png'%i for i in range(25)]
    batch_size = 5
    with sess.as_default() and graph.as_default():
        tf_images = build_input_pipeline(filenames,batch_size)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        #tf_out = infererence(tf_images)
        tf_loss = calculate_loss(tf_images)
        tf_optimize = optimize_model(tf_loss)

        _ = sess.run(tf_optimize)

        coord.request_stop()
        coord.join(threads)
