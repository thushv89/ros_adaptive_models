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
import matplotlib.gridspec as gridspec

logger = logging.getLogger('CNNVisualizationLogger')
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter('[%(name)s] [%(funcName)s] %(message)s'))
console.setLevel(logging.INFO)
logger.addHandler(console)


def save_cnn_hyperparameters(main_dir,weight_sizes_dict,stride_dict, scope_list, hypeparam_filepath):

    if config.WEIGHT_SAVE_DIR and not os.path.exists(main_dir + os.sep + config.WEIGHT_SAVE_DIR):
        os.mkdir(main_dir + os.sep + config.WEIGHT_SAVE_DIR)

    hyperparam_dict = {'layers': config.TF_ANG_SCOPES,
                       'activations': config.ACTIVATION}

    for scope in scope_list:
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
            if 'out' not in scope:
                weights_name = scope + config.TF_SCOPE_DIVIDER + config.TF_WEIGHTS_STR
                bias_name = scope + config.TF_SCOPE_DIVIDER + config.TF_BIAS_STR
                with tf.variable_scope(scope,reuse=True):
                    var_dict[weights_name] = tf.get_variable(config.TF_WEIGHTS_STR)
                    var_dict[bias_name] = tf.get_variable(config.TF_BIAS_STR)

                    with open(main_dir + os.sep + config.WEIGHT_SAVE_DIR +os.sep + 'variable_names.txt','w') as f:
                        f.write(weights_name)
                        f.write(bias_name)

            else:
                with tf.variable_scope(scope, reuse=True):
                    for di in config.TF_DIRECTION_LABELS:
                        with tf.variable_scope(di, reuse=True):
                            weights_name = scope + config.TF_SCOPE_DIVIDER + di +\
                                           config.TF_SCOPE_DIVIDER + config.TF_WEIGHTS_STR
                            bias_name = scope + config.TF_SCOPE_DIVIDER + di +\
                                        config.TF_SCOPE_DIVIDER + config.TF_BIAS_STR

                            var_dict[weights_name] = tf.get_variable(config.TF_WEIGHTS_STR)
                            var_dict[bias_name] = tf.get_variable(config.TF_BIAS_STR)

                            with open(main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + 'variable_names.txt','w') as f:
                                f.write(weights_name)
                                f.write(bias_name)

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


def save_cnn_weights_multiple(main_dir, sess, model_filepath):
    var_dict = {}
    all_directions = ['left','straight','right']
    if config.WEIGHT_SAVE_DIR and not os.path.exists(main_dir + os.sep + config.WEIGHT_SAVE_DIR):
        os.mkdir(main_dir + os.sep + config.WEIGHT_SAVE_DIR)

    for scope in config.TF_ANG_SCOPES:

        if 'conv' in scope:
            for di in all_directions:
                # Have to use the weights name like this (manually creating name) because otherwise
                # the weights are not saved with scope information
                weights_name = scope + config.TF_SCOPE_DIVIDER + di + config.TF_SCOPE_DIVIDER + config.TF_WEIGHTS_STR
                bias_name = scope + config.TF_SCOPE_DIVIDER + di + config.TF_SCOPE_DIVIDER + config.TF_BIAS_STR
                with tf.variable_scope(scope,reuse=True):
                    with tf.variable_scope(di, reuse=True):
                        var_dict[weights_name] = tf.get_variable(config.TF_WEIGHTS_STR)
                        var_dict[bias_name] = tf.get_variable(config.TF_BIAS_STR)
        elif 'fc' in scope or 'out' in scope:
            with tf.variable_scope(scope,reuse=True):
                for di in all_directions:
                    # Have to use the weights name like this because otherwise
                    # the weights are not saved with scope information
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
            if 'pool' not in scope and 'out' not in scope:
                tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope],
                                initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_NAIVE[scope][-1],
                                initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            elif 'out' == scope:
                for di in config.TF_DIRECTION_LABELS:
                    with tf.variable_scope(di,reuse=False):
                        tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope],
                                        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                        tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_NAIVE[scope][-1],
                                        initializer=tf.constant_initializer(0.0, dtype=tf.float32))

def cnn_create_variables_multiple_with_scope_size_stride(hyperparam_filepath):
    hyperparam_dict = pickle.load(open(hyperparam_filepath, 'rb'))
    scope_list = hyperparam_dict['layers']

    for scope in scope_list:
        with tf.variable_scope(scope,reuse=False):
            for di in config.TF_DIRECTION_LABELS:
                with tf.variable_scope(di,reuse=False):
                    if 'conv' in scope:
                        tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_ANG_VAR_SHAPES_MULTIPLE[scope],
                                        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                        tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_MULTIPLE[scope][-1],
                                        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                    elif 'fc' in scope or 'out' in scope:

                        tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_ANG_VAR_SHAPES_MULTIPLE[scope],
                                        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                        tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_MULTIPLE[scope][-1],
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

                # Reshaping required for the first fulcon layer
                h_per_di = []
                if scope == 'out':
                    for di in config.TF_DIRECTION_LABELS:
                        with tf.variable_scope(di, reuse=True):
                            weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                        logger.info('\t\tFully-connected with output Logits for %s', scope)
                        h_per_di.append(tf.matmul(h, weight) + bias)

                    h = tf.concat(h_per_di,axis=1)

                elif 'fc' in scope:
                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
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


def cnn_visualize_activations_multiple(main_dir, sess, weights_filepath, hyperparam_filepath, activation_image,batch_size):
    hyperparam_dict = pickle.load(open(hyperparam_filepath,'rb'))

    scope_list = hyperparam_dict['layers']
    activation = hyperparam_dict['activations']

    activation_dict = {}

    for scope in scope_list:

        with tf.variable_scope(scope,reuse=True):
            if 'conv' in scope:
                h_per_di = []
                for di in config.TF_DIRECTION_LABELS:
                    with tf.variable_scope(di):
                        key = scope + config.TF_SCOPE_DIVIDER + di
                        weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                        weight.set_shape(hyperparam_dict[scope]['weights_size'])
                        bias.set_shape([hyperparam_dict[scope]['weights_size'][-1]])

                        if scope=='conv1':
                            logger.info('\t\tConvolution with %s activation for %s', activation, scope)
                            h_local = models_utils.activate(
                                tf.nn.conv2d(activation_image, weight, strides=hyperparam_dict[scope]['stride'], padding='SAME') + bias,
                                activation, name='hidden')
                            h_per_di.append(h_local)

                        else:
                            logger.info('\t\tConvolution with %s activation for %s', activation, scope)
                            h_local = models_utils.activate(
                                tf.nn.conv2d(h, weight, strides=hyperparam_dict[scope]['stride'], padding='SAME') + bias,
                                activation, name='hidden')
                            h_per_di.append(h_local)

                        activation_dict[key] = h_local

                h = tf.concat(h_per_di,axis=3)

            elif 'pool' in scope:
                logger.info('\t\tMax pooling for %s', scope)
                h = tf.nn.max_pool(h, hyperparam_dict[scope]['weights_size'], hyperparam_dict[scope]['stride'],
                                   padding='SAME', name='pool_hidden')

            else:

                # Reshaping required for the first fulcon layer
                if scope == 'out':
                    continue

                elif 'fc' in scope:
                    if scope == 'fc1':
                        h_per_di = []
                        for di in config.TF_DIRECTION_LABELS:
                            key = scope + config.TF_SCOPE_DIVIDER + di
                            with tf.variable_scope(di):

                                weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)

                                h_shape = h.get_shape().as_list()
                                logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s', scope, h_shape)
                                h_local = tf.reshape(h, [batch_size, h_shape[1] * h_shape[2] * h_shape[3]])
                                h_local = models_utils.activate(tf.matmul(h_local, weight) + bias, activation)
                                h_per_di.append(h_local)

                            activation_dict[key] = h

                        h = tf.concat(h_per_di,axis=1)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

    return activation_dict



def cnn_store_activations_as_image(activation_dict,visualization_scopes,orig_image,img_id, filename_prefix,activation_type):

    number_of_cols_per_direction = 4
    rows_per_layer = 2
    rows_for_original = 2
    nrows, ncols = len(visualization_scopes) * rows_per_layer + rows_for_original, 3 * number_of_cols_per_direction

    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    fig = plt.figure(1)
    padding = 0.0
    fig_w = ncols * 1.0 + (ncols + 1) * padding
    fig_h = nrows * .7 + (nrows + 1) * padding
    fig.set_size_inches(fig_w, fig_h)

    gs0 = gridspec.GridSpec(nrows, ncols, wspace=0.1, hspace=0.05)

    ax = [[None for _ in range(ncols)] for _ in range(nrows)]
    for ri in range(nrows):
        for ci in range(ncols):
            ax[ri][ci] = plt.subplot(gs0[ri * ncols + ci])
            ax[ri][ci].axis('off')

    norm_img = orig_image[0, :, :, :] - np.min(orig_image[0, :, :, :])
    norm_img = (norm_img / np.max(norm_img))
    ax_original_image = plt.Subplot(fig, gs0[:2, :2])
    fig.add_subplot(ax_original_image)
    ax_original_image.imshow(norm_img, aspect='auto')
    ax_original_image.axis('off')

    for sc_i, scope in enumerate(visualization_scopes):
        if 'fc' in scope:
            raise NotImplementedError

        key = scope
        v = activation_dict[key]
        tensor_depth = v.shape[-1]
        print(v.shape)

        if activation_type == 'highest':
            best_indices = np.argsort(np.mean(v, axis=tuple(range(0, 3))).flatten())
            best_indices = best_indices.flatten()[-3*number_of_cols_per_direction*rows_per_layer:]
        elif activation_type == 'random':
            best_indices = np.random.randint(0, tensor_depth, 3*number_of_cols_per_direction*rows_per_layer)
        print('Found best indices: ', best_indices)

        for bi, best_idx in enumerate(best_indices):
            ri = ((sc_i * rows_per_layer) + rows_for_original) + (bi // (number_of_cols_per_direction*3))
            ci = bi % (number_of_cols_per_direction*3)
            #print(bi,ri,len(ax),len(ax[ri]))
            ax[ri][ci].imshow(v[0, :, :, best_idx],aspect='auto')
            ax[ri][ci].axis('off')

    # Annotations
    ax[3][0].text(-0.1,-2, 'Conv1',
            horizontalalignment='right',
            verticalalignment='center',
            rotation='vertical',fontsize=20)

    ax[5][0].text(-0.25, -2, 'Conv2',
                  horizontalalignment='right',
                  verticalalignment='center',
                  rotation='vertical',fontsize=20)

    ax[7][0].text(-0.5, -2, 'Conv3',
                  horizontalalignment='right',
                  verticalalignment='center',
                  rotation='vertical',fontsize=20)

    ax[9][0].text(-0.75, -2, 'Conv4',
                  horizontalalignment='right',
                  verticalalignment='center',
                  rotation='vertical',fontsize=20)

    #ax[1][0].annotate('Conv1', xy=(-0.1,1.0),xytext=(-0.1, 1.0),fontsize=16,arrowprops=dict(facecolor='black', shrink=0.05))
    # fig.tight_layout()
    fig.subplots_adjust(bottom=0.01, top=0.99, right=0.9, left=0.1)
    fig.savefig(filename_prefix + '_naive_activation_%d.jpg' % (img_id))
    plt.cla()
    plt.close(fig)



def cnn_store_activations_as_image_heirarchy_for_cerberus(activation_dict,visualization_scopes,orig_image,img_id, filename_prefix,activation_type):

    number_of_cols_per_direction = 4
    rows_per_layer = 2
    rows_for_original = 2
    nrows, ncols = len(visualization_scopes)*rows_per_layer + rows_for_original, 3 * number_of_cols_per_direction

    #fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    fig = plt.figure(1)
    padding = 0.0
    fig_w = ncols * 1.0 + (ncols + 1) * padding
    fig_h = nrows * 0.7 + (nrows + 1) * padding
    fig.set_size_inches(fig_w, fig_h)
    gs0 = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.0)
    sub_gs = []
    #gs0.update(wspace=0.1, hspace=0.01, left=0.1, right=0.4, bottom=0.1, top=0.9)
    for di in range(3):
        sub_gs.append(gridspec.GridSpecFromSubplotSpec(nrows, number_of_cols_per_direction, wspace=0.1, hspace=0.05, subplot_spec=gs0[di]))

    #gs1.update(wspace=0.05, hspace=0.05)  # set the spacing between axes.
    #padding = 0.1
    #fig_w = ncols * 2.0 + (ncols + 1) * padding
    #fig_h = nrows * 1.0 + (nrows + 1) * padding
    #fig.set_size_inches(fig_w, fig_h)

    ax = [[None for _ in range(ncols)] for _ in range(nrows)]
    for ri in range(nrows):
        for ci in range(ncols):
            sub_gs_index = ci//number_of_cols_per_direction
            within_sub_gs_index = ci%number_of_cols_per_direction
            ax[ri][ci]=plt.subplot(sub_gs[sub_gs_index][ri*number_of_cols_per_direction+within_sub_gs_index])
            ax[ri][ci].axis('off')

    norm_img = orig_image[0, :, :, :] - np.min(orig_image[0, :, :, :])
    norm_img = (norm_img / np.max(norm_img))
    ax_original_image = plt.Subplot(fig, sub_gs[0][:2, :2])
    fig.add_subplot(ax_original_image)
    ax_original_image.imshow(norm_img,aspect='auto')
    ax_original_image.axis('off')


    for sc_i,scope in enumerate(visualization_scopes):
        if 'fc' in scope:
            raise NotImplementedError

        for d_idx, di in enumerate(config.TF_DIRECTION_LABELS):
            key = scope + config.TF_SCOPE_DIVIDER + di
            v = activation_dict[key]
            tensor_depth = v.shape[-1]
            print(v.shape)

            if activation_type=='highest':
                best_indices = np.argsort(np.mean(v,axis=tuple(range(0, 3))).flatten())
                best_indices = best_indices.flatten()[-number_of_cols_per_direction*rows_per_layer:]
            elif activation_type=='random':
                best_indices = np.random.randint(0,tensor_depth,number_of_cols_per_direction*rows_per_layer)
            print('Found best indices: ',best_indices)

            for bi, best_idx in enumerate(best_indices):
                ri = (sc_i * rows_per_layer)+ rows_for_original + (bi//number_of_cols_per_direction)
                ci = (number_of_cols_per_direction*d_idx) + (bi%number_of_cols_per_direction)

                ax[ri][ ci].imshow(v[0, :, :, best_idx],aspect='auto')
                ax[ri][ci].axis('off')

        # Annotations
        ax[3][0].text(-0.1, -2, 'Conv1',
                      horizontalalignment='right',
                      verticalalignment='center',
                      rotation='vertical', fontsize=20)

        ax[5][0].text(-0.25, -2, 'Conv2',
                      horizontalalignment='right',
                      verticalalignment='center',
                      rotation='vertical', fontsize=20)

        ax[7][0].text(-0.5, -2, 'Conv3',
                      horizontalalignment='right',
                      verticalalignment='center',
                      rotation='vertical', fontsize=20)

        ax[9][0].text(-0.75, -2, 'Conv4',
                      horizontalalignment='right',
                      verticalalignment='center',
                      rotation='vertical', fontsize=20)

    #fig.tight_layout()
    fig.subplots_adjust(bottom=0.01, top=0.99, right=0.9, left=0.1)
    fig.savefig(filename_prefix + '_multiple_activation_%d.jpg'%(img_id))
    plt.cla()
    plt.close(fig)