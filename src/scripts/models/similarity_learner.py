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


def build_input_pipeline(filenames, batch_size):
    global sess, graph
    global logger
    logger.info('Received filenames: %s', filenames)
    with tf.name_scope('sim_preprocess'):
        # FIFO Queue of file names
        # creates a FIFO queue until the reader needs them
        filename_queue = tf.train.string_input_producer(filenames, capacity=2, shuffle=True,name='string_input_producer')

        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)
            else:
                logger.info('File %s found.',f)
        # Reader which takes a filename queue and read() which outputs data one by one
        reader = tf.TFRecordReader()

        key, serial_example = reader.read(filename_queue, name='image_read_op')

        features = tf.parse_single_example(
            serial_example,
            features = {config.FEAT_IMG_RAW : tf.FixedLenFeature([], tf.string),
                        config.FEAT_LABEL : tf.FixedLenFeature([], tf.float32)}
        )

        image = tf.decode_raw(features[config.FEAT_IMG_RAW], tf.float32)
        image = tf.cast(image,tf.float32)
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32, name='float_image')
        image = tf.reshape(image,config.TF_INPUT_SIZE)
        image.set_shape(config.TF_INPUT_SIZE)

        #image = tf.image.resize_images(image,config.TF_INPUT_RESIZE_SIZE)

        label = tf.cast(features[config.FEAT_LABEL], tf.float32)
        # standardize image
        std_image = tf.image.per_image_standardization(image)

        # https://stackoverflow.com/questions/37126108/how-to-read-data-into-tensorflow-batches-from-example-queue

        # The batching mechanism that takes a output produced by reader (with preprocessing) and outputs a batch tensor
        # [batch_size, height, width, depth] 4D tensor
        image_batch = tf.train.batch([std_image], batch_size=batch_size,
                                     capacity=10, name='image_batch', allow_smaller_final_batch=True)

        record_count = reader.num_records_produced()
        # to use record reader we need to use a Queue either random

    print('Preprocessing done\n')
    return image_batch


def build_tensorflw_variables():
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
                    weights = tf.get_variable(config.TF_WEIGHTS_STR,shape=config.TF_VAR_SHAPES[scope],
                                              initializer=tf.truncated_normal_initializer(stddev=0.02,dtype=tf.float32))
                    bias = tf.get_variable(config.TF_BIAS_STR, config.TF_VAR_SHAPES[scope][-1],
                                           initializer = tf.constant_initializer(0.001,dtype=tf.float32))

                except ValueError as e:
                    logger.critical(e)
                    logger.debug('Variables in scope %s already initialized\n'%scope)

        with tf.variable_scope(config.TF_DECONV_STR):

            for si, scope in enumerate(config.TF_SCOPES):
                with tf.variable_scope(scope) as sc:

                    # Try Except because if you try get_variable with an intializer and
                    # the variable exists, you will get a ValueError saying the variable exists
                    #
                    try:
                        deconv_weights = tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_VAR_SHAPES[scope],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.02,
                                                                                              dtype=tf.float32))
                        deconv_bias = tf.get_variable(config.TF_BIAS_STR, config.TF_VAR_SHAPES[scope][-2],
                                               initializer=tf.constant_initializer(0.001, dtype=tf.float32))

                    except ValueError as e:
                        logger.critical(e)
                        logger.debug('Variables in scope %s already initialized\n' % scope)


        print([v.name for v in tf.global_variables()])

def get_hidden_rep(tf_inputs):
    '''
        Inferencing the model. The input (tf_inputs) is propagated through convolutions poolings
        fully-connected layers to obtain the final softmax output
        :param tf_inputs: a batch of images (tensorflow placeholder)
        :return:
        '''
    global logger
    logger.info('Defining inference ops ...')
    logger.info('\tDefining the forward pass ...')
    with tf.name_scope('forward'):
        for si, scope in enumerate(config.TF_SCOPES):
            with tf.variable_scope(scope, reuse=True) as sc:
                weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)

                if 'fc' not in scope:
                    logger.info('\t\tConvolution with ReLU activation for %s', scope)
                    if si == 0:
                        h = tf.nn.relu(
                            tf.nn.conv2d(tf_inputs, weight, strides=config.TF_STRIDES[scope], padding='SAME') + bias,
                            name='hidden')
                    else:
                        h = tf.nn.relu(tf.nn.conv2d(h, weight, strides=config.TF_STRIDES[scope], padding='SAME') + bias,
                                       name='hidden')
                else:
                    # Reshaping required for the first fulcon layer
                    if scope == 'fc1':
                        logger.info('\t\tFully-connected with ReLU activation for %s', scope)
                        h_shape = h.get_shape().as_list()
                        logger.info('\t\t\tReshaing the input before feeding to %s', scope)
                        h = tf.reshape(h, [h_shape[0], h_shape[1] * h_shape[2] * h_shape[3]])
                        h = tf.nn.relu(tf.matmul(h, weight) + bias, name='hidden')
                    else:
                        raise NotImplementedError

    return h


def infererence(tf_inputs):
    '''
    Inferencing the model. The input (tf_inputs) is propagated through convolutions poolings
    fully-connected layers to obtain the final softmax output
    :param tf_inputs: a batch of images (tensorflow placeholder)
    :return:
    '''
    global logger
    logger.info('Defining inference ops ...')
    logger.info('\tDefining the forward pass ...')
    with tf.name_scope('forward'):
        for si, scope in enumerate(config.TF_SCOPES):
            with tf.variable_scope(scope,reuse=True) as sc:
                weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)

                if 'fc' not in scope:
                    logger.info('\t\tConvolution with ReLU activation for %s',scope)
                    if si == 0:
                        h = tf.nn.relu(tf.nn.conv2d(tf_inputs,weight,strides=config.TF_STRIDES[scope],padding='SAME')+bias,name='hidden')
                    else:
                        h = tf.nn.relu(tf.nn.conv2d(h, weight, strides=config.TF_STRIDES[scope], padding='SAME') + bias,
                                       name='hidden')
                else:
                    # Reshaping required for the first fulcon layer
                    if scope == 'fc1':
                        logger.info('\t\tFully-connected with ReLU activation for %s',scope)
                        h_shape = h.get_shape().as_list()
                        logger.info('\t\t\tReshaing the input before feeding to %s',scope)
                        h = tf.reshape(h,[h_shape[0], h_shape[1] * h_shape[2] * h_shape[3]])
                        h = tf.nn.relu(tf.matmul(h, weight) + bias, name= 'hidden')
                    else:
                        raise NotImplementedError

    logger.info('\tDefining the backward pass ...')
    with tf.name_scope('backward'):
        with tf.variable_scope(config.TF_DECONV_STR, reuse=True) as sc:

            for si, scope in enumerate(reversed(config.TF_SCOPES)):
                with tf.variable_scope(scope, reuse=True) as sc:

                    weight = tf.get_variable(config.TF_WEIGHTS_STR)

                    deconv_bias = tf.get_variable(config.TF_BIAS_STR)

                    if 'fc' not in scope:
                        logger.info('\t\tConvolution with ReLU activation for %s', scope)
                        if si == len(config.TF_SCOPES)-1:
                            logger.info('\t\t\tLast layer of the backward pass (%s) Using Sigmoid', scope)
                            out = tf.nn.tanh(
                                tf.nn.conv2d_transpose(h, weight, strides=config.TF_STRIDES[scope], output_shape=config.TF_OUTPUT_SHAPES[scope],padding='SAME') + deconv_bias,
                                name='hidden')
                        else:
                            h = tf.nn.relu(tf.nn.conv2d_transpose(h, weight, strides=config.TF_STRIDES[scope], output_shape=config.TF_OUTPUT_SHAPES[scope], padding='SAME') + deconv_bias,
                                           name='hidden')
                    else:
                        # Reshaping required for the first fulcon layer
                        if scope == 'fc1':
                            logger.info('\t\tFully-connected with ReLU activation for %s', scope)
                            h = tf.nn.relu(tf.matmul(h, tf.transpose(weight)) + deconv_bias, name='hidden')
                            h = tf.reshape(h, [h_shape[0], h_shape[1], h_shape[2], h_shape[3]])
                        else:
                            # doesn't support multiple fully-connected layers
                            raise NotImplementedError
                        #h = tf.nn.relu(tf.matmul(h, weight) + deconv_bias, name='hidden')

    return out


def calculate_loss(tf_inputs,tf_outputs):

    out = infererence(tf_inputs)

    loss = tf.reduce_mean(tf.reduce_sum((tf_outputs - out)**2,axis=[1,2,3]))

    return loss

def optimize_model(loss):

    optimize = tf.train.AdamOptimizer(beta1=0.5,learning_rate=0.0002).minimize(loss)
    return optimize


if __name__ == '__main__':

    filenames = ['..'+os.sep+'sample'+os.sep+'image_angle-%d.tfrecords'%i for i in range(4)]
    batch_size = 5
    with sess.as_default() and graph.as_default():
        build_tensorflw_variables()

        tf_images = build_input_pipeline(filenames,batch_size+2)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf_inputs = tf.placeholder(shape=[batch_size,config.TF_INPUT_SIZE[0],config.TF_INPUT_SIZE[1],config.TF_INPUT_SIZE[2]],dtype=tf.float32)
        tf_outputs = tf.placeholder(
            shape=[batch_size, config.TF_INPUT_SIZE[0], config.TF_INPUT_SIZE[1], config.TF_INPUT_SIZE[2]], dtype=tf.float32)

        tf_loss = calculate_loss(tf_inputs, tf_outputs)
        tf_optimize = optimize_model(tf_loss)

        tf_hidden_rep = get_hidden_rep(tf_inputs)

        tf.global_variables_initializer().run()
        for epoch in range(10):
            for step in range(18):
                print('Running step %d ...'%step)
                norm_images = sess.run(tf_images)
                l1,_ = sess.run([tf_loss,tf_optimize],feed_dict={tf_inputs:norm_images[1:batch_size+1,:,:,:],
                                                tf_outputs:norm_images[:batch_size,:,:,:]})
                l2,_ = sess.run([tf_loss, tf_optimize], feed_dict={tf_inputs: norm_images[1:batch_size + 1, :, :, :],
                                                 tf_outputs: norm_images[2:, :, :, :]})

                print('\tLoss: %.3f'%((l1+l2)/2.0))

        coord.request_stop()
        coord.join(threads)
