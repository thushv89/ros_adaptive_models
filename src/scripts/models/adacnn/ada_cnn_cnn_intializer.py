import tensorflow as tf
import ada_cnn_constants as constants
import logging
import numpy as np
import sys

TF_WEIGHTS = constants.TF_WEIGHTS
TF_BIAS = constants.TF_BIAS
TF_TRAIN_MOMENTUM = constants.TF_TRAIN_MOMENTUM
TF_POOL_MOMENTUM = constants.TF_POOL_MOMENTUM
TF_GLOBAL_SCOPE = constants.TF_GLOBAL_SCOPE
TF_CONV_WEIGHT_SHAPE_STR = constants.TF_CONV_WEIGHT_SHAPE_STR
TF_FC_WEIGHT_IN_STR = constants.TF_FC_WEIGHT_IN_STR
TF_FC_WEIGHT_OUT_STR = constants.TF_FC_WEIGHT_OUT_STR
TF_ACTIVAIONS_STR = constants.TF_ACTIVAIONS_STR
TF_SCOPE_DIVIDER = constants.TF_SCOPE_DIVIDER



research_parameters = None
model_hyperparameters = None

init_logger = None

def set_from_main(research_params,logging_level, logging_format,model_params):
    global research_parameters,init_logger,model_hyperparameters

    research_parameters = research_params
    model_hyperparameters = model_params
    init_logger = logging.getLogger('cnn_initializer_logger')
    init_logger.propagate = False
    init_logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    init_logger.addHandler(console)


def initialize_cnn_with_ops(cnn_ops, cnn_hyps):
    '''
    Initialize the variables of the CNN (convolution filters fully-connected filters
    :param cnn_ops:
    :param cnn_hyps:
    :return:
    '''

    init_logger.info('CNN Hyperparameters')
    init_logger.info('%s\n', cnn_hyps)

    init_logger.info('Initializing the iConvNet (conv_global,pool_global,classifier)...\n')
    for op in cnn_ops:

        if 'conv' in op:
            with tf.variable_scope(op, reuse=False) as scope:
                tf.get_variable(
                    name=TF_WEIGHTS, shape=cnn_hyps[op]['weights'],
                    initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    validate_shape=False, dtype=tf.float32)
                tf.get_variable(
                    name=TF_BIAS,
                    initializer=tf.random_uniform(shape=[cnn_hyps[op]['weights'][3]],minval=-0.01, maxval=0.01),
                    validate_shape=False, dtype=tf.float32)

                if model_hyperparameters['use_batchnorm']:

                    tf.get_variable(
                        name=constants.TF_POP_MU,
                        initializer=tf.zeros(shape=[1,1,1,cnn_hyps[op]['weights'][3]],
                                             dtype=tf.float32),
                        validate_shape=False, trainable=False)

                    tf.get_variable(
                        name=constants.TF_POP_VAR,
                        initializer=tf.ones(shape=[1,1,1,cnn_hyps[op]['weights'][3]],
                                             dtype=tf.float32),
                        validate_shape=False, trainable=False)


                    tf.get_variable(
                        name=constants.TF_GAMMA,
                        initializer=tf.ones(shape=[1,1,1,cnn_hyps[op]['weights'][3]],
                                            dtype=tf.float32),
                        validate_shape=False, trainable=True)

                    tf.get_variable(
                        name=constants.TF_BETA,
                        initializer=tf.zeros(shape=[1,1,1,cnn_hyps[op]['weights'][3]],
                                            dtype=tf.float32),
                        validate_shape=False, trainable=True)

                if model_hyperparameters['save_best_model']:
                    with tf.variable_scope('best',reuse=False):
                        tf.get_variable(
                            name=TF_WEIGHTS, shape=cnn_hyps[op]['weights'],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            validate_shape=False, dtype=tf.float32)
                        tf.get_variable(
                            name=TF_BIAS,
                            initializer=tf.random_uniform(shape=[cnn_hyps[op]['weights'][3]], minval=-0.01,
                                                          maxval=0.01),
                            validate_shape=False, dtype=tf.float32)

                init_logger.debug('Weights for %s initialized with size %s', op, str(cnn_hyps[op]['weights']))
                init_logger.debug('Biases for %s initialized with size %d', op, cnn_hyps[op]['weights'][3])

        if 'fulcon' in op:

            with tf.variable_scope(op, reuse=False):
                for di in ['left','straight','right']:
                    with tf.variable_scope(di, reuse=False):
                        tf.get_variable(
                            name=TF_WEIGHTS, shape=[cnn_hyps[op]['in'], cnn_hyps[op]['out']],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            validate_shape=False, dtype=tf.float32)
                        tf.get_variable(
                            name=TF_BIAS,
                            initializer=tf.random_uniform(shape=[cnn_hyps[op]['out']], minval=-0.01, maxval=0.01),
                            validate_shape=False, dtype=tf.float32)

                        if model_hyperparameters['use_batchnorm']:
                            tf.get_variable(
                                name=constants.TF_POP_MU,
                                initializer=tf.zeros(shape=[1,cnn_hyps[op]['out']]),
                                validate_shape=False, dtype=tf.float32, trainable=False)

                            tf.get_variable(
                                name=constants.TF_POP_VAR,
                                initializer=tf.ones(shape=[1,cnn_hyps[op]['out']]),
                                validate_shape=False, dtype=tf.float32, trainable=False)

                            tf.get_variable(
                                name=constants.TF_BETA,
                                initializer=tf.zeros(shape=[1,cnn_hyps[op]['out']]),
                                validate_shape=False, dtype=tf.float32)

                            tf.get_variable(
                                name=constants.TF_GAMMA,
                                initializer=tf.ones(shape=[1,cnn_hyps[op]['out']]),
                                validate_shape=False, dtype=tf.float32)

                        if model_hyperparameters['save_best_model']:
                            with tf.variable_scope('best', reuse=False):
                                tf.get_variable(
                                    name=TF_WEIGHTS, shape=[cnn_hyps[op]['in'], cnn_hyps[op]['out']],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    validate_shape=False, dtype=tf.float32)
                                tf.get_variable(
                                    name=TF_BIAS,
                                    initializer=tf.random_uniform(shape=[cnn_hyps[op]['out']], minval=-0.01,
                                                                  maxval=0.01),
                                    validate_shape=False, dtype=tf.float32)

                        init_logger.debug('Weights for %s initialized with size %d,%d',
                                     op, cnn_hyps[op]['in'], cnn_hyps[op]['out'])
                        init_logger.debug('Biases for %s initialized with size %d', op, cnn_hyps[op]['out'])


def copy_weights_to_best_model(cnn_ops):
    update_ops = []
    for op in cnn_ops:

        if 'conv' in op:
            with tf.variable_scope(op):
                w,b = tf.get_variable(TF_WEIGHTS),tf.get_variable(TF_BIAS)
                with tf.variable_scope('best'):
                    w_dash, b_dash = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
                    update_ops.append(tf.assign(w_dash,w,validate_shape=False))
                    update_ops.append(tf.assign(b_dash, b, validate_shape=False))

        if 'fulcon' in op:
            with tf.variable_scope(op):
                for di in ['left','straight','right']:
                    with tf.variable_scope(di):
                        w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
                        with tf.variable_scope('best'):
                            w_dash, b_dash = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
                            update_ops.append(tf.assign(w_dash, w, validate_shape=False))
                            update_ops.append(tf.assign(b_dash, b, validate_shape=False))

    return update_ops

def define_velocity_vectors(main_scope, cnn_ops, cnn_hyperparameters):
    # if using momentum
    vel_var_list = []
    print(cnn_ops)
    if research_parameters['optimizer'] == 'Momentum':
        for tmp_op in cnn_ops:
            op_scope = tmp_op
            if 'conv' in tmp_op:
                with tf.variable_scope(tmp_op):

                    with tf.variable_scope(TF_WEIGHTS) as scope:
                        vel_var_list.append(
                            tf.get_variable(name=TF_TRAIN_MOMENTUM,
                                            initializer=tf.zeros(shape=cnn_hyperparameters[tmp_op]['weights'],
                                                                 dtype=tf.float32),
                                            dtype=tf.float32, trainable=False))
                        vel_var_list.append(
                            tf.get_variable(name=TF_POOL_MOMENTUM,
                                            initializer=tf.zeros(shape=cnn_hyperparameters[tmp_op]['weights'],
                                                                 dtype=tf.float32),
                                            dtype=tf.float32, trainable=False))

                    with tf.variable_scope(TF_BIAS) as scope:
                        vel_var_list.append(tf.get_variable(name=TF_TRAIN_MOMENTUM,
                                                            initializer=tf.zeros(
                                                                shape=cnn_hyperparameters[tmp_op]['weights'][3],
                                                                dtype=tf.float32),
                                                            dtype=tf.float32, trainable=False))

                        vel_var_list.append(tf.get_variable(name=TF_POOL_MOMENTUM,
                                                            initializer=tf.zeros(
                                                                shape=[cnn_hyperparameters[tmp_op]['weights'][3]],
                                                                dtype=tf.float32),
                                                            dtype=tf.float32, trainable=False))

            elif 'fulcon' in tmp_op:
                with tf.variable_scope(tmp_op):

                    for di in ['left','straight','right']:
                        with tf.variable_scope(di):
                            with tf.variable_scope(TF_WEIGHTS) as scope:
                                vel_var_list.append(tf.get_variable(name=TF_TRAIN_MOMENTUM,
                                                                    initializer=tf.zeros(
                                                                        shape=[cnn_hyperparameters[tmp_op]['in'],
                                                                               cnn_hyperparameters[tmp_op]['out']],
                                                                        dtype=tf.float32),
                                                                    dtype=tf.float32, trainable=False))
                                vel_var_list.append(tf.get_variable(name=TF_POOL_MOMENTUM,
                                                                    initializer=tf.zeros(
                                                                        shape=[cnn_hyperparameters[tmp_op]['in'],
                                                                               cnn_hyperparameters[tmp_op]['out']],
                                                                        dtype=tf.float32),
                                                                    dtype=tf.float32, trainable=False))

                            with tf.variable_scope(TF_BIAS) as scope:
                                vel_var_list.append(tf.get_variable(name=TF_TRAIN_MOMENTUM,
                                                                    initializer=tf.zeros(
                                                                        shape=[cnn_hyperparameters[tmp_op]['out']],
                                                                        dtype=tf.float32),
                                                                    dtype=tf.float32, trainable=False))

                                vel_var_list.append(tf.get_variable(name=TF_POOL_MOMENTUM,
                                                                    initializer=tf.zeros(
                                                                        shape=[cnn_hyperparameters[tmp_op]['out']],
                                                                        dtype=tf.float32),
                                                                    dtype=tf.float32, trainable=False))

    return vel_var_list


def reset_cnn(cnn_hyps,cnn_ops):
    reset_ops = []
    init_logger.info('CNN Hyperparameters')
    init_logger.info('%s\n', cnn_hyps)

    init_logger.info('Initializing the iConvNet (conv_global,pool_global,classifier)...\n')
    for op in cnn_ops:

        if 'conv' in op:
            with tf.variable_scope(op) as scope:
                weights = tf.get_variable(name=TF_WEIGHTS)
                new_weights = tf.random_uniform(cnn_hyps[op]['weights'],
                                                minval=-np.sqrt(
                                                    6. / (cnn_hyps[op]['weights'][0] * cnn_hyps[op]['weights'][1] *
                                                          (cnn_hyps[op]['weights'][-2] + cnn_hyps[op]['weights'][-1]))
                                                    ),
                                                maxval=np.sqrt(
                                                    6. / (cnn_hyps[op]['weights'][0] * cnn_hyps[op]['weights'][1] *
                                                          (cnn_hyps[op]['weights'][-2] + cnn_hyps[op]['weights'][-1]))
                                                    )
                                                )

                reset_ops.append(tf.assign(weights, new_weights, validate_shape=False))

                with tf.variable_scope(TF_WEIGHTS) as child_scope:
                    w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                    pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

                    new_w_vel = tf.zeros(shape=cnn_hyps[op]['weights'], dtype=tf.float32)
                    reset_ops.append(tf.assign(w_vel, new_w_vel, validate_shape=False))
                    reset_ops.append(tf.assign(pool_w_vel, new_w_vel, validate_shape=False))

                bias = tf.get_variable(name=TF_BIAS)
                new_bias = tf.constant(np.random.random() * 0.001, shape=[cnn_hyps[op]['weights'][3]])

                reset_ops.append(tf.assign(bias, new_bias, validate_shape=False))

                with tf.variable_scope(TF_BIAS) as child_scope:
                    b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                    pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)

                    new_b_vel = tf.zeros(shape=[cnn_hyps[op]['weights'][3]], dtype=tf.float32)
                    reset_ops.append(tf.assign(b_vel, new_b_vel, validate_shape=False))
                    reset_ops.append(tf.assign(pool_b_vel, new_b_vel, validate_shape=False))



        if 'fulcon' in op:
            with tf.variable_scope(op) as scope:
                for di in ['left','straight','right']:
                    with tf.variable_scope(di):
                        weights = tf.get_variable(name=TF_WEIGHTS)
                        new_weights = tf.random_uniform([cnn_hyps[op]['in'], cnn_hyps[op]['out']],
                                                        minval=-np.sqrt(6. / (cnn_hyps[op]['in'] + cnn_hyps[op]['out'])),
                                                        maxval=np.sqrt(6. / (cnn_hyps[op]['in'] + cnn_hyps[op]['out']))
                                                        )
                        reset_ops.append(tf.assign(weights, new_weights, validate_shape=False))

                        with tf.variable_scope(TF_WEIGHTS) as child_scope:
                            w_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                            pool_w_vel = tf.get_variable(TF_POOL_MOMENTUM)

                            new_w_vel = tf.zeros(shape=[cnn_hyps[op]['in'], cnn_hyps[op]['out']], dtype=tf.float32)
                            reset_ops.append(tf.assign(w_vel, new_w_vel, validate_shape=False))
                            reset_ops.append(tf.assign(pool_w_vel, new_w_vel, validate_shape=False))

                        bias = tf.get_variable(name=TF_BIAS)
                        new_bias = tf.constant(np.random.random() * 0.001, shape=[cnn_hyps[op]['out']])
                        reset_ops.append(tf.assign(bias, new_bias, validate_shape=False))

                        with tf.variable_scope(TF_BIAS) as child_scope:
                            b_vel = tf.get_variable(TF_TRAIN_MOMENTUM)
                            pool_b_vel = tf.get_variable(TF_POOL_MOMENTUM)

                            new_b_vel = tf.zeros(shape=[cnn_hyps[op]['out']], dtype=tf.float32)
                            reset_ops.append(tf.assign(b_vel, new_b_vel, validate_shape=False))
                            reset_ops.append(tf.assign(pool_b_vel, new_b_vel, validate_shape=False))

    return reset_ops



def init_tf_hyperparameters(cnn_ops, cnn_hyperparameters):
    tf_hyp_list = {}
    for op in cnn_ops:
        if 'conv' in op:
            with tf.variable_scope(op) as scope:
                tf_hyp_list[op] = {TF_CONV_WEIGHT_SHAPE_STR:
                                       tf.get_variable(name=TF_CONV_WEIGHT_SHAPE_STR,
                                                       initializer=cnn_hyperparameters[op]['weights'],
                                                       dtype=tf.int32, trainable=False)
                                   }

        if 'fulcon' in op:
            with tf.variable_scope(op) as scope:
                tf_hyp_list[op] = {TF_FC_WEIGHT_IN_STR: tf.get_variable(name=TF_FC_WEIGHT_IN_STR,
                                                                        initializer=cnn_hyperparameters[op]['in'],
                                                                        dtype=tf.int32, trainable=False)
                    , TF_FC_WEIGHT_OUT_STR: tf.get_variable(name=TF_FC_WEIGHT_OUT_STR,
                                                            initializer=cnn_hyperparameters[op]['out'],
                                                            dtype=tf.int32, trainable=False)
                                   }
    return tf_hyp_list
