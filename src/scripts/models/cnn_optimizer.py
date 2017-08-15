import config
import tensorflow as tf
import logging
import sys

logger = logging.getLogger('OptLogger')
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter('[%(name)s] [%(funcName)s] %(message)s'))
console.setLevel(logging.INFO)
logger.addHandler(console)


def optimize_model_detached(loss, global_step, direction):

    '''
    This method optimizes the CNN if it has separate outputs for each direction (1 tanh output per direction)
    This is important as this method will not work for the naive CNN optimizer
    :param loss: loss to optimize
    :param global_step: global step is used to decay the learning rate
    :param direction: Output weights belonging to this direction only will be optimized
    :return:
    '''
    momentum = 0.9

    mom_update_ops = []
    grads_and_vars = []
    learning_rate = tf.maximum(
        tf.train.exponential_decay(0.01, global_step, decay_steps=1, decay_rate=0.9, staircase=True,
                                   name='learning_rate_decay'), 1e-4)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    for si, scope in enumerate(config.TF_ANG_SCOPES):
        with tf.variable_scope(scope,reuse=True):

            if scope == 'fc1' or scope == 'out':

                with tf.variable_scope(direction, reuse=True):
                    w, b = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                    [(g_w, w), (g_b, b)] = optimizer.compute_gradients(loss, [w, b])

                    with tf.variable_scope(config.TF_WEIGHTS_STR,reuse=True):
                        w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                    with tf.variable_scope(config.TF_BIAS_STR,reuse=True):
                        b_vel = tf.get_variable(config.TF_MOMENTUM_STR)
            elif 'conv' in scope:

                w, b = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                [(g_w, w), (g_b, b)] = optimizer.compute_gradients(loss, [w, b])
                with tf.variable_scope(config.TF_WEIGHTS_STR, reuse=True):
                    w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                with tf.variable_scope(config.TF_BIAS_STR, reuse=True):
                    b_vel = tf.get_variable(config.TF_MOMENTUM_STR)

            mom_update_ops.append(tf.assign(w_vel, momentum * w_vel + g_w))
            grads_and_vars.append((w_vel * learning_rate, w))
            mom_update_ops.append(tf.assign(b_vel,momentum*b_vel + g_b))
            grads_and_vars.append((b_vel * learning_rate, b))


    optimize = optimizer.apply_gradients(grads_and_vars)

    return optimize, mom_update_ops, learning_rate


def optimize_model_naive(loss, global_step):
    '''
    Optimize a naive CNN model
    :param loss:
    :param tf_labels:
    :param global_step:
    :param use_masking:
    :param collision:
    :return:
    '''
    momentum = 0.0
    mom_update_ops = []
    grads_and_vars = []
    learning_rate = tf.maximum(
        tf.train.exponential_decay(0.01, global_step, decay_steps=1, decay_rate=0.9, staircase=True,
                                   name='learning_rate_decay'), 1e-4)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    for si, scope in enumerate(config.TF_ANG_SCOPES):
        if 'pool' in scope:
            continue

        with tf.variable_scope(scope,reuse=True):
            w,b = tf.get_variable(config.TF_WEIGHTS_STR),tf.get_variable(config.TF_BIAS_STR)
            [(g_w,w),(g_b,b)] = optimizer.compute_gradients(loss,[w,b])

            with tf.variable_scope(config.TF_WEIGHTS_STR,reuse=True):
                w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                mom_update_ops.append(tf.assign(w_vel, momentum * w_vel + g_w))
                grads_and_vars.append((w_vel * learning_rate, w))
            with tf.variable_scope(config.TF_BIAS_STR,reuse=True):
                # TODO: MASKING FOR BIAS
                b_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                mom_update_ops.append(tf.assign(b_vel, momentum*b_vel + g_b))
                grads_and_vars.append((b_vel * learning_rate, b))

    optimize = optimizer.apply_gradients(grads_and_vars)

    return optimize,mom_update_ops,grads_and_vars


def optimize_model_naive_no_momentum(loss, global_step,var_list=None):
    '''
    Optimize a naive CNN model with the built-in Optimizer
    :param loss:
    :param global_step:
    :return:
    '''
    learning_rate = tf.maximum(
        tf.train.exponential_decay(0.005, global_step, decay_steps=1, decay_rate=0.9, staircase=True,
                                   name='learning_rate_decay'), 1e-4)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss,var_list=var_list)

    optimize = optimizer.minimize(loss,var_list=var_list)
    return optimize,grads_and_vars


def optimize_model_dual(loss, global_step, direction,collision):
    global logger

    '''
    This method optimizes the CNN if it has separate outputs for each direction (1 tanh output per direction)
    This is important as this method will not work for the naive CNN optimizer
    :param loss: loss to optimize
    :param global_step: global step is used to decay the learning rate
    :param direction: Output weights belonging to this direction only will be optimized
    :return:
    '''
    momentum = 0.9

    mom_update_ops = []
    grads_and_vars = []
    learning_rate = tf.maximum(
        tf.train.exponential_decay(0.01, global_step, decay_steps=1, decay_rate=0.9, staircase=True,
                                   name='learning_rate_decay'), 1e-4)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    for si, scope in enumerate(config.TF_ANG_SCOPES):
            logger.info('For scope: %s',scope)

            if scope == 'out':
                if not collision:
                    with tf.variable_scope(config.TF_NONCOL_STR,reuse=True):
                        with tf.variable_scope(scope, reuse=True):
                            with tf.variable_scope(direction, reuse=True):
                                w, b = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                                [(g_w, w), (g_b, b)] = optimizer.compute_gradients(loss, [w, b])
                                logger.info('\tdefine optimize for %s',w.name)
                                with tf.variable_scope(config.TF_WEIGHTS_STR,reuse=True):
                                    w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                                with tf.variable_scope(config.TF_BIAS_STR,reuse=True):
                                    b_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                else:
                    with tf.variable_scope(config.TF_COL_STR,reuse=True):
                        with tf.variable_scope(scope, reuse=True):
                            with tf.variable_scope(direction, reuse=True):
                                w, b = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                                logger.info('\tdefine optimize for %s', w.name)
                                [(g_w, w), (g_b, b)] = optimizer.compute_gradients(loss, [w, b])

                                with tf.variable_scope(config.TF_WEIGHTS_STR, reuse=True):
                                    w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                                with tf.variable_scope(config.TF_BIAS_STR, reuse=True):
                                    b_vel = tf.get_variable(config.TF_MOMENTUM_STR)

            elif 'pool' not in scope:
                if scope=='conv1':
                    with tf.variable_scope(scope, reuse=True):
                        w, b = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                        logger.info('\tdefine optimize for %s', w.name)
                        [(g_w, w), (g_b, b)] = optimizer.compute_gradients(loss, [w, b])
                        with tf.variable_scope(config.TF_WEIGHTS_STR, reuse=True):
                            w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                        with tf.variable_scope(config.TF_BIAS_STR, reuse=True):
                            b_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                else:
                    if not collision:
                        with tf.variable_scope(config.TF_NONCOL_STR,reuse=True):
                            with tf.variable_scope(scope, reuse=True):
                                w, b = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                                logger.info('\tdefine optimize for %s', w.name)
                                [(g_w, w), (g_b, b)] = optimizer.compute_gradients(loss, [w, b])

                                with tf.variable_scope(config.TF_WEIGHTS_STR, reuse=True):
                                    w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                                with tf.variable_scope(config.TF_BIAS_STR, reuse=True):
                                    b_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                    else:
                        with tf.variable_scope(config.TF_COL_STR,reuse=True):
                            with tf.variable_scope(scope, reuse=True):
                                w, b = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                                logger.info('\tdefine optimize for %s', w.name)
                                [(g_w, w), (g_b, b)] = optimizer.compute_gradients(loss, [w, b])
                                with tf.variable_scope(config.TF_WEIGHTS_STR, reuse=True):
                                    w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                                with tf.variable_scope(config.TF_BIAS_STR, reuse=True):
                                    b_vel = tf.get_variable(config.TF_MOMENTUM_STR)

            else:
                continue

            mom_update_ops.append(tf.assign(w_vel, momentum * w_vel + g_w))
            grads_and_vars.append((w_vel * learning_rate, w))
            mom_update_ops.append(tf.assign(b_vel,momentum*b_vel + g_b))
            grads_and_vars.append((b_vel * learning_rate, b))

    optimize = optimizer.apply_gradients(grads_and_vars)

    return optimize, mom_update_ops, learning_rate


def optimize_model_dual_naive(loss, global_step,collision):
    global logger

    '''
    This method optimizes the CNN if it has separate outputs for each direction (1 tanh output per direction)
    This is important as this method will not work for the naive CNN optimizer
    :param loss: loss to optimize
    :param global_step: global step is used to decay the learning rate
    :param direction: Output weights belonging to this direction only will be optimized
    :return:
    '''
    momentum = 0.0

    mom_update_ops = []
    grads_and_vars = []
    learning_rate = tf.maximum(
        tf.train.exponential_decay(0.01, global_step, decay_steps=1, decay_rate=0.9, staircase=True,
                                   name='learning_rate_decay'), 1e-4)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    for si, scope in enumerate(config.TF_ANG_SCOPES):
            logger.info('For scope: %s',scope)

            if scope == 'out':
                if not collision:
                    with tf.variable_scope(config.TF_NONCOL_STR,reuse=True):
                        with tf.variable_scope(scope, reuse=True):
                            w, b = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                            [(g_w, w), (g_b, b)] = optimizer.compute_gradients(loss, [w, b])
                            logger.info('\tdefine optimize for %s',w.name)
                            with tf.variable_scope(config.TF_WEIGHTS_STR,reuse=True):
                                w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                            with tf.variable_scope(config.TF_BIAS_STR,reuse=True):
                                b_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                else:
                    with tf.variable_scope(config.TF_COL_STR,reuse=True):
                        with tf.variable_scope(scope, reuse=True):
                            w, b = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                            logger.info('\tdefine optimize for %s', w.name)
                            [(g_w, w), (g_b, b)] = optimizer.compute_gradients(loss, [w, b])

                            with tf.variable_scope(config.TF_WEIGHTS_STR, reuse=True):
                                w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                            with tf.variable_scope(config.TF_BIAS_STR, reuse=True):
                                b_vel = tf.get_variable(config.TF_MOMENTUM_STR)

            elif 'pool' not in scope:

                if not collision:
                    with tf.variable_scope(config.TF_NONCOL_STR,reuse=True):
                        with tf.variable_scope(scope, reuse=True):
                            w, b = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                            logger.info('\tdefine optimize for %s', w.name)
                            [(g_w, w), (g_b, b)] = optimizer.compute_gradients(loss, [w, b])

                            with tf.variable_scope(config.TF_WEIGHTS_STR, reuse=True):
                                w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                            with tf.variable_scope(config.TF_BIAS_STR, reuse=True):
                                b_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                else:
                    with tf.variable_scope(config.TF_COL_STR,reuse=True):
                        with tf.variable_scope(scope, reuse=True):
                            w, b = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                            logger.info('\tdefine optimize for %s', w.name)
                            [(g_w, w), (g_b, b)] = optimizer.compute_gradients(loss, [w, b])
                            with tf.variable_scope(config.TF_WEIGHTS_STR, reuse=True):
                                w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                            with tf.variable_scope(config.TF_BIAS_STR, reuse=True):
                                b_vel = tf.get_variable(config.TF_MOMENTUM_STR)

            else:
                continue

            mom_update_ops.append(tf.assign(w_vel, momentum * w_vel + g_w))
            grads_and_vars.append((w_vel * learning_rate, w))
            mom_update_ops.append(tf.assign(b_vel,momentum*b_vel + g_b))
            grads_and_vars.append((b_vel * learning_rate, b))

    optimize = optimizer.apply_gradients(grads_and_vars)

    return optimize, mom_update_ops, learning_rate, grads_and_vars


def optimize_model_dual_naive_builtin(loss, global_step,var_list):
    global logger

    '''
    This method optimizes the CNN if it has separate outputs for each direction (1 tanh output per direction)
    This is important as this method will not work for the naive CNN optimizer
    :param loss: loss to optimize
    :param global_step: global step is used to decay the learning rate
    :param direction: Output weights belonging to this direction only will be optimized
    :return:
    '''
    mom_update_ops = []
    grads_and_vars = []
    learning_rate = tf.maximum(
        tf.train.exponential_decay(0.001, global_step, decay_steps=1, decay_rate=0.9, staircase=True,
                                   name='learning_rate_decay'), 1e-5)

    optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, var_list=var_list)
    return optimize, mom_update_ops, learning_rate, grads_and_vars