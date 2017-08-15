import config
import tensorflow as tf
import pickle
import os
import models_utils
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import ceil


logger = logging.getLogger('CNNVisualizationLogger')
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter('[%(name)s] [%(funcName)s] %(message)s'))
console.setLevel(logging.INFO)
logger.addHandler(console)


def save_cnn_hyperparameters(main_dir,weight_sizes_dict,stride_dict, hypeparam_filepath):

    if config.WEIGHT_SAVE_DIR and not os.path.exists(main_dir + os.sep + config.WEIGHT_SAVE_DIR):
        os.mkdir(main_dir + os.sep + config.WEIGHT_SAVE_DIR)

    hyperparam_dict = {'layers': config.TF_ANG_SCOPES,
                       'activations': config.ACTIVATION}

    for scope in config.TF_ANG_SCOPES:
        if 'fc' not in scope and 'out' not in scope:
            hyperparam_dict[scope] = {'weights_size': weight_sizes_dict[scope],
                                  'stride': stride_dict[scope]}
        else:
            hyperparam_dict[scope] = {'weights_size': weight_sizes_dict[scope]}

    pickle.dump(hyperparam_dict, open(main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + hypeparam_filepath, "wb"))


def save_cnn_weights_naive(main_dir, sess,model_filepath):
    var_dict = {}

    if config.WEIGHT_SAVE_DIR and not os.path.exists(main_dir + os.sep + config.WEIGHT_SAVE_DIR):
        os.mkdir(main_dir + os.sep + config.WEIGHT_SAVE_DIR)

    for scope in config.TF_ANG_SCOPES:

        if 'pool' not in scope:
            weights_name = scope + config.TF_SCOPE_DIVIDER + config.TF_WEIGHTS_STR
            bias_name = scope + config.TF_SCOPE_DIVIDER + config.TF_BIAS_STR
            with tf.variable_scope(scope,reuse=True):
                var_dict[weights_name] = tf.get_variable(config.TF_WEIGHTS_STR)
                var_dict[bias_name] = tf.get_variable(config.TF_BIAS_STR)

    saver = tf.train.Saver(var_dict)
    saver.save(sess,main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep +  model_filepath)


def save_cnn_weights_detached(main_dir, sess,model_filepath):
    var_dict = {}
    all_directions = ['left','straight','right']
    if config.WEIGHT_SAVE_DIR and not os.path.exists(main_dir + os.sep + config.WEIGHT_SAVE_DIR):
        os.mkdir(main_dir + os.sep + config.WEIGHT_SAVE_DIR)

    for scope in config.TF_ANG_SCOPES:

        if 'conv' in scope:
            weights_name = scope + config.TF_SCOPE_DIVIDER + config.TF_WEIGHTS_STR
            bias_name = scope + config.TF_SCOPE_DIVIDER + config.TF_BIAS_STR
            with tf.variable_scope(scope,reuse=True):
                var_dict[weights_name] = tf.get_variable(config.TF_WEIGHTS_STR)
                var_dict[bias_name] = tf.get_variable(config.TF_BIAS_STR)
        elif 'fc' in scope or 'out' in scope:
            with tf.variable_scope(scope,reuse=True):
                for di in all_directions:
                    weights_name = scope + config.TF_SCOPE_DIVIDER + di + config.TF_SCOPE_DIVIDER + config.TF_WEIGHTS_STR
                    bias_name = scope + config.TF_SCOPE_DIVIDER + di + config.TF_SCOPE_DIVIDER + config.TF_BIAS_STR
                    with tf.variable_scope(di,reuse=True):
                        var_dict[weights_name] = tf.get_variable(config.TF_WEIGHTS_STR)
                        var_dict[bias_name] = tf.get_variable(config.TF_BIAS_STR)

    saver = tf.train.Saver(var_dict)
    saver.save(sess,main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + model_filepath)


def cnn_create_variables_naive_with_scope_size_stride(hyperparam_filepath):
    hyperparam_dict = pickle.load(open(hyperparam_filepath, 'rb'))
    scope_list = hyperparam_dict['layers']

    for scope in scope_list:
        with tf.variable_scope(scope,reuse=False):
            if 'pool' not in scope:
                tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope],
                                initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_NAIVE[scope][-1],
                                initializer=tf.constant_initializer(0.0, dtype=tf.float32))


def cnn_create_variables_detached_with_scope_size_stride(hyperparam_filepath):
    hyperparam_dict = pickle.load(open(hyperparam_filepath, 'rb'))
    scope_list = hyperparam_dict['layers']
    all_directions = ['left', 'straight', 'right']

    for scope in scope_list:
        with tf.variable_scope(scope,reuse=False):
            if 'conv' in scope:
                tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope],
                                initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_NAIVE[scope][-1],
                                initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            elif 'fc' in scope or 'out' in scope:
                for di in all_directions:
                    with tf.variable_scope(di):
                        tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope],
                                        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                        tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_NAIVE[scope][-1],
                                        initializer=tf.constant_initializer(0.0, dtype=tf.float32))


def cnn_visualize_activations_naive(main_dir, sess, weights_filepath, hyperparam_filepath, activation_image,batch_size):
    hyperparam_dict = pickle.load(open(hyperparam_filepath,'rb'))

    scope_list = hyperparam_dict['layers']
    activation = hyperparam_dict['activations']

    activation_dict = {}

    for scope in scope_list:

        mod_weight_string = config.TF_WEIGHTS_STR + ':0'
        mod_bias_string = config.TF_BIAS_STR + ':0'
        with tf.variable_scope(scope,reuse=True):
            if 'conv' in scope:
                weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                weight.set_shape(hyperparam_dict[scope]['weights_size'])
                bias.set_shape([hyperparam_dict[scope]['weights_size'][-1]])

                if scope=='conv1':
                    logger.info('\t\tConvolution with %s activation for %s', activation, scope)
                    h = models_utils.activate(
                        tf.nn.conv2d(activation_image, weight, strides=hyperparam_dict[scope]['stride'], padding='SAME') + bias,
                        activation, name='hidden')
                else:
                    logger.info('\t\tConvolution with %s activation for %s', activation, scope)
                    h = models_utils.activate(
                        tf.nn.conv2d(h, weight, strides=hyperparam_dict[scope]['stride'], padding='SAME') + bias,
                        activation, name='hidden')

                activation_dict[scope] = h

            elif 'pool' in scope:
                logger.info('\t\tMax pooling for %s', scope)
                h = tf.nn.max_pool(h, hyperparam_dict[scope]['weights_size'], hyperparam_dict[scope]['stride'],
                                   padding='SAME', name='pool_hidden')

            else:
                weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                # Reshaping required for the first fulcon layer
                if scope == 'out':
                    logger.info('\t\tFully-connected with output Logits for %s', scope)
                    h = tf.matmul(h, weight) + bias

                elif 'fc' in scope:
                    if scope == 'fc1':
                        h_shape = h.get_shape().as_list()
                        logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s', scope, h_shape)
                        h = tf.reshape(h, [batch_size, h_shape[1] * h_shape[2] * h_shape[3]])
                        h = models_utils.activate(tf.matmul(h, weight) + bias, activation)

                        activation_dict[scope] = h
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

    return activation_dict


def cnn_visualize_activations_detached(main_dir, sess, weights_filepath, hyperparam_filepath, activation_image,batch_size):
    hyperparam_dict = pickle.load(open(hyperparam_filepath,'rb'))

    scope_list = hyperparam_dict['layers']
    activation = hyperparam_dict['activations']

    activation_dict = {}

    for scope in scope_list:

        with tf.variable_scope(scope,reuse=True):
            if 'conv' in scope:
                weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                weight.set_shape(hyperparam_dict[scope]['weights_size'])
                bias.set_shape([hyperparam_dict[scope]['weights_size'][-1]])

                if scope=='conv1':
                    logger.info('\t\tConvolution with %s activation for %s', activation, scope)
                    h = models_utils.activate(
                        tf.nn.conv2d(activation_image, weight, strides=hyperparam_dict[scope]['stride'], padding='SAME') + bias,
                        activation, name='hidden')
                else:
                    logger.info('\t\tConvolution with %s activation for %s', activation, scope)
                    h = models_utils.activate(
                        tf.nn.conv2d(h, weight, strides=hyperparam_dict[scope]['stride'], padding='SAME') + bias,
                        activation, name='hidden')

                activation_dict[scope] = h

            elif 'pool' in scope:
                logger.info('\t\tMax pooling for %s', scope)
                h = tf.nn.max_pool(h, hyperparam_dict[scope]['weights_size'], hyperparam_dict[scope]['stride'],
                                   padding='SAME', name='pool_hidden')

            else:
                weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                # Reshaping required for the first fulcon layer
                if scope == 'out':
                    logger.info('\t\tFully-connected with output Logits for %s', scope)
                    h = tf.matmul(h, weight) + bias

                elif 'fc' in scope:
                    if scope == 'fc1':
                        h_shape = h.get_shape().as_list()
                        logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s', scope, h_shape)
                        h = tf.reshape(h, [batch_size, h_shape[1] * h_shape[2] * h_shape[3]])
                        h = models_utils.activate(tf.matmul(h, weight) + bias, activation)

                        activation_dict[scope] = h
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

    return activation_dict


def cnn_store_activations_as_image(activation_dict,orig_image,img_id, filename_prefix):

    number_of_cols = 10

    for k,v in activation_dict.items():
        tensor_depth = v.shape[-1]
        print(v.shape)
        # depth+1 for the original image
        fig, ax = plt.subplots(nrows=ceil((tensor_depth+1)*1.0/number_of_cols), ncols=number_of_cols)

        if 'fc' in k:
            continue

        for ri in range(ceil((tensor_depth+1)*1.0/number_of_cols)):
            for ci in range(number_of_cols):
                if ri==0 and ci==0:
                    norm_img = orig_image[0,:,:,:] - np.min(orig_image[0,:,:,:])
                    norm_img = (norm_img / np.max(norm_img))
                    ax[ri, ci].imshow(norm_img)
                    ax[ri,ci].axis('off')
                else:
                    index = ri*number_of_cols + ci - 1
                    if index >= tensor_depth:
                        break
                    if 'conv' in k:
                        ax[ri,ci].imshow(v[0,:,:,index])
                        ax[ri, ci].axis('off')

        fig.savefig(filename_prefix + '_activation_%s_%d.jpg'%(k,img_id))
        plt.cla()
        plt.close(fig)

