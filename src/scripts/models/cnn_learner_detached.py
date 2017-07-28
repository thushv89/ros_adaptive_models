import numpy as np
import tensorflow as tf
import logging
import sys
from math import ceil
import os
from PIL import Image
import config
import getopt
import models_utils
import scipy.misc

'''
Network Architecture
                          > left/fc1 -> tanh (1)
                        /
conv1 -> conv2 -> conv3 --> staright/fc1 -> tanh (1)
                        \
                          > right/fc1 -> tanh(1)

'''
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

graph = None
#configp = tf.ConfigProto(allow_soft_placement=True)
sess = None

activation = 'tanh'

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
        for si,scope in enumerate(config.TF_ANG_SCOPES):
            with tf.variable_scope(scope) as sc:

                # Try Except because if you try get_variable with an intializer and
                # the variable exists, you will get a ValueError saying the variable exists

                try:
                        if not (scope=='fc1' or scope=='out'):
                            tf.get_variable(config.TF_WEIGHTS_STR,shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope],
                                                      initializer=tf.truncated_normal_initializer(stddev=0.02,dtype=tf.float32))
                            tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_DETACHED[scope][-1],
                                                   initializer = tf.constant_initializer(0.001,dtype=tf.float32))

                            with tf.variable_scope(config.TF_WEIGHTS_STR):
                                tf.get_variable(config.TF_MOMENTUM_STR,shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope],
                                                initializer = tf.constant_initializer(0,dtype=tf.float32))
                            with tf.variable_scope(config.TF_BIAS_STR):
                                tf.get_variable(config.TF_MOMENTUM_STR,shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope][-1],
                                                initializer = tf.constant_initializer(0,dtype=tf.float32))
                        else:
                            direction = ['left','straight','right']
                            for di in direction:
                                with tf.variable_scope(di):
                                    tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.02,
                                                                                                dtype=tf.float32))
                                    tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_DETACHED[scope][-1],
                                                    initializer=tf.constant_initializer(0.001, dtype=tf.float32))

                                    with tf.variable_scope(config.TF_WEIGHTS_STR):
                                        tf.get_variable(config.TF_MOMENTUM_STR, shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope],
                                                        initializer=tf.constant_initializer(0, dtype=tf.float32))
                                    with tf.variable_scope(config.TF_BIAS_STR):
                                        tf.get_variable(config.TF_MOMENTUM_STR,
                                                        shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope][-1],
                                                        initializer=tf.constant_initializer(0, dtype=tf.float32))

                except ValueError as e:
                    logger.critical(e)
                    logger.debug('Variables in scope %s already initialized\n'%scope)

        print([v.name for v in tf.global_variables()])


def logits_detached(tf_inputs, direction):
    '''
    Inferencing the model. The input (tf_inputs) is propagated through convolutions poolings
    fully-connected layers to obtain the final softmax output
    :param tf_inputs: a batch of images (tensorflow placeholder)
    :return:
    '''
    global logger
    logger.info('Defining inference ops ...')
    all_directions = ['left', 'straight', 'right']
    with tf.name_scope('infer'):
        for si, scope in enumerate(config.TF_ANG_SCOPES):
            with tf.variable_scope(scope,reuse=True) as sc:

                if 'conv' in scope:
                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                    logger.info('\t\tConvolution with ReLU activation for %s',scope)
                    if si == 0:
                        h = models_utils.activate(tf.nn.conv2d(tf_inputs,weight,strides=config.TF_ANG_STRIDES[scope],padding='SAME')+bias,activation,name='hidden')
                    else:
                        h = models_utils.activate(tf.nn.conv2d(h, weight, strides=config.TF_ANG_STRIDES[scope], padding='SAME') + bias,
                                                  activation,name='hidden')
                else:
                    # Reshaping required for the first fulcon layer
                    if scope == 'out':
                        logger.info('\t\tFully-connected with output Logits for %s',scope)
                        if direction is not None:
                            with tf.variable_scope(direction,reuse=True):
                                weights, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                                h = tf.matmul(h, weights) + bias
                                h = tf.squeeze(h,name='hidden_squeeze')

                        else:
                            h_out_list = []
                            for di in all_directions:
                                with tf.variable_scope(di, reuse=True):
                                    weights, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                        config.TF_BIAS_STR)
                                    h_out_list.append(tf.matmul(h_list[di], weights) + bias)

                            h = tf.squeeze(tf.stack(h_out_list,axis=1))

                    elif 'fc' in scope:
                        if scope == config.TF_FIRST_FC_ID:
                            h_shape = h.get_shape().as_list()
                            logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s', scope, h_shape)
                            h = tf.reshape(h, [batch_size, h_shape[1] * h_shape[2] * h_shape[3]])
                            if direction is not None:
                                with tf.variable_scope(direction, reuse=True):
                                    weights, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                        config.TF_BIAS_STR)
                                    h = models_utils.activate(tf.matmul(h, weights) + bias, activation)

                            else:
                                h_list = {}
                                for di in all_directions:
                                    with tf.variable_scope(di, reuse=True):
                                        weights, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                            config.TF_BIAS_STR)
                                        h_list[di]=models_utils.activate(tf.matmul(h, weights) + bias, activation)

                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError

    return h


def predictions_and_labels_with_logits(tf_logits,tf_labels):
    pred = tf.nn.tanh(tf_logits)
    return pred,tf_labels


def predictions_and_labels_with_inputs(tf_inputs, tf_labels):
    tf_logits = logits_detached(tf_inputs,direction=None)
    return tf.nn.tanh(tf_logits),tf_labels


def calculate_hybrid_loss(tf_noncol_logits,tf_noncol_labels,tf_col_logits,tf_col_labels):
    tf_noncol_out  = tf.nn.tanh(tf_noncol_logits)
    tf_col_out = tf.nn.tanh(tf_col_logits)

    loss = tf.reduce_mean(0.8*(tf_noncol_out - tf.cast(tf.reduce_max(tf_noncol_labels,axis=1),dtype=tf.float32))**2 +
                          0.2*(tf_col_out - tf.cast(tf.reduce_min(tf_col_labels,axis=1),dtype=tf.float32))**2)

    return loss

def calculate_loss(tf_logits,tf_labels,collision,direction):

    if collision:
        tf_labels_arg = tf.cast(tf.reduce_min(tf_labels,axis=1),dtype=tf.float32)
    else:
        tf_labels_arg = tf.cast(tf.reduce_max(tf_labels,axis=1),dtype=tf.float32)
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf_out, labels=tf_labels_arg))
    loss = tf.reduce_mean((tf.nn.tanh(tf_logits) - tf_labels_arg)**2)

    return loss


def optimize_model(loss, global_step, direction):
    momentum = 0.5
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
            else:
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
    #optimize = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=learning_rate).minimize(loss)

    return optimize, mom_update_ops, learning_rate


def inc_gstep(gstep):
    return tf.assign(gstep,gstep+1)

if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "", ["data-dir="])
    except getopt.GetoptError as error:
        print('<filename>.py --data-dir=<dirname>')
        print(error)
        sys.exit(2)

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '--data-dir':
                IMG_DIR = str(arg)

    if IMG_DIR and not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)

    config.DETACHED_TOP_LAYERS = True
    dataset_filenames = {'train_dataset': {'left': [
                                                       '..' + os.sep + 'data_indoor_1_1000' + os.sep + 'image-direction-%d-0.tfrecords' % i
                                                       for i in range(2)
                                                       ] +
                                                   [
                                                       '..' + os.sep + 'data_sandbox_1000' + os.sep + 'image-direction-%d-0.tfrecords' % i
                                                       for i in range(1)
                                                       ],
                                           'straight': [
                                                           '..' + os.sep + 'data_indoor_1_1000' + os.sep + 'image-direction-%d-1.tfrecords' % i
                                                           for i in range(3)] +
                                                       [
                                                           '..' + os.sep + 'data_sandbox_1000' + os.sep + 'image-direction-%d-1.tfrecords' % i
                                                           for i in range(4)],
                                           'right': [
                                                        '..' + os.sep + 'data_indoor_1_1000' + os.sep + 'image-direction-%d-2.tfrecords' % i
                                                        for i in range(2)] +
                                                    [
                                                        '..' + os.sep + 'data_sandbox_1000' + os.sep + 'image-direction-%d-2.tfrecords' % i
                                                        for i in range(2)]
                                           },

                         'train_bump_dataset': {
                             'left': [
                                         '..' + os.sep + 'data_indoor_1_bump_200' + os.sep + 'image-direction-%d-0.tfrecords' % i
                                         for i in range(1)] +
                                     [
                                         '..' + os.sep + 'data_sandbox_bump_200' + os.sep + 'image-direction-%d-0.tfrecords' % i
                                         for i in range(1)]
                             ,
                             'straight': [
                                             '..' + os.sep + 'data_indoor_1_bump_200' + os.sep + 'image-direction-%d-1.tfrecords' % i
                                             for i in range(2)] +
                                         [
                                             '..' + os.sep + 'data_sandbox_bump_200' + os.sep + 'image-direction-%d-1.tfrecords' % i
                                             for i in range(2)]
                             ,
                             'right': [
                                          '..' + os.sep + 'data_indoor_1_bump_200' + os.sep + 'image-direction-%d-2.tfrecords' % i
                                          for i in range(1)] +
                                      [
                                          '..' + os.sep + 'data_sandbox_bump_200' + os.sep + 'image-direction-%d-2.tfrecords' % i
                                          for i in range(1)]
                         },

                         'test_dataset': [
                                               '..' + os.sep + 'data_grande_salle_1000' + os.sep + 'image-direction-%d-0.tfrecords' % i
                                               for i in range(2)] +
                                           [
                                               '..' + os.sep + 'data_grande_salle_1000' + os.sep + 'image-direction-%d-1.tfrecords' % i
                                               for i in range(3)] +
                                           [
                                               '..' + os.sep + 'data_grande_salle_1000' + os.sep + 'image-direction-%d-2.tfrecords' % i
                                               for i in range(2)],
                         'test_bump_dataset': [
                                                  '..' + os.sep + 'data_grande_salle_bump_200' + os.sep + 'image-direction-%d-0.tfrecords' % i
                                                  for i in range(1)] +
                                              [
                                                  '..' + os.sep + 'data_grande_salle_bump_200' + os.sep + 'image-direction-%d-1.tfrecords' % i
                                                  for i in range(2)] +
                                              [
                                                  '..' + os.sep + 'data_grande_salle_bump_200' + os.sep + 'image-direction-%d-2.tfrecords' % i
                                                  for i in range(1)]
                         }

    dataset_sizes = {'train_dataset': 1000 + 1000,
                     'train_bump_dataset': 400,
                     'test_dataset': 1000,
                     'test_bump_dataset': 200}

    min_col_loss, min_noncol_loss = 10000,10000
    col_exceed_min_count, noncol_exceed_min_count = 0,0

    predictionlogger = logging.getLogger('PredictionLogger')
    predictionlogger.setLevel(logging.INFO)
    fh = logging.FileHandler(IMG_DIR + os.sep + 'predictions.log', mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    predictionlogger.addHandler(fh)
    predictionlogger.info('#ID:Actual:Predicted')

    bumpPredictionlogger = logging.getLogger('BumpPredictionLogger')
    bumpPredictionlogger.setLevel(logging.INFO)
    bumpfh = logging.FileHandler(IMG_DIR + os.sep + 'bump_predictions.log', mode='w')
    bumpfh.setFormatter(logging.Formatter('%(message)s'))
    bumpfh.setLevel(logging.INFO)
    bumpPredictionlogger.addHandler(bumpfh)
    bumpPredictionlogger.info('#ID:Actual:Predicted')

    accuracy_logger = logging.getLogger('AccuracyLogger')
    accuracy_logger.setLevel(logging.INFO)
    accuracyFH = logging.FileHandler(IMG_DIR + os.sep + 'accuracy.log', mode='w')
    accuracyFH.setFormatter(logging.Formatter('%(message)s'))
    accuracyFH.setLevel(logging.INFO)
    accuracy_logger.addHandler(accuracyFH)
    accuracy_logger.info('#Epoch;Non-collision loss; Collision loss; Non-collision Accuracy;Non-collision Accuracy(Soft);Collision Accuracy;' +
                         'Collision Accuracy (Soft);Preci-NC-L,Preci-NC-S,Preci-NC-R;Rec-NC-L,Rec-NC-S,Rec-NC-R;' +
                         'Preci-C-L,Preci-C-S,Preci-C-R;Rec-C-L,Rec-C-S,Rec-C-R')

    batch_size = 5

    graph = tf.Graph()
    configp = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.InteractiveSession(graph=graph,config=configp)

    with sess.as_default() and graph.as_default():
        build_tensorflw_variables()
        models_utils.set_from_main(sess,graph,logger)

        lr = 0.01
        noncol_global_step = tf.Variable(0,trainable=False)
        col_global_step = tf.Variable(0,trainable=False)

        inc_noncol_gstep = inc_gstep(noncol_global_step)
        inc_col_gstep = inc_gstep(col_global_step)

        tf_images, tf_labels = {},{}
        tf_bump_images, tf_bump_labels = {},{}
        tf_loss, tf_logits = {},{}
        tf_bump_loss, tf_bump_logits = {},{}
        tf_optimize,tf_mom_update_ops,tf_grads = {},{},{}
        tf_bump_optimize,tf_bump_mom_update_ops,tf_bump_grads = {},{},{}
        tf_train_predictions, tf_train_bump_predictions = {},{}
        tf_train_actuals, tf_train_bump_actuals = {}, {}

        for direction in ['left','straight','right']:
            tf_images[direction],tf_labels[direction] = models_utils.build_input_pipeline(
                dataset_filenames['train_dataset'][direction],batch_size,shuffle=True,
                training_data=True,use_opposite_label=False,inputs_for_sdae=False)

            tf_bump_images[direction], tf_bump_labels[direction] = models_utils.build_input_pipeline(
                dataset_filenames['train_bump_dataset'][direction], batch_size, shuffle=True,
                training_data=True, use_opposite_label=True,inputs_for_sdae=False)

            tf_logits[direction] = logits_detached(tf_images[direction], direction=direction)
            tf_bump_logits[direction] = logits_detached(tf_bump_images[direction], direction=direction)

            tf_train_predictions[direction],tf_train_actuals[direction] = predictions_and_labels_with_inputs(tf_images[direction],tf_labels[direction])
            tf_train_bump_predictions[direction],tf_train_bump_actuals[direction] = predictions_and_labels_with_inputs(tf_bump_images[direction],tf_labels[direction])

        for direction in ['left', 'straight', 'right']:
            if not config.OPTIMIZE_HYBRID_LOSS:
                tf_loss[direction] = calculate_loss(tf_logits[direction], tf_labels[direction],
                                                    collision=False, direction=direction)
                tf_bump_loss[direction] = calculate_loss(tf_bump_logits[direction], tf_bump_labels[direction],
                                                         collision=True, direction=direction)

                tf_optimize[direction], tf_mom_update_ops[direction], \
                tf_noncol_lr = optimize_model(tf_loss[direction],noncol_global_step,direction=direction)
                tf_bump_optimize[direction], tf_bump_mom_update_ops[direction], \
                tf_col_lr = optimize_model(tf_bump_loss[direction],col_global_step,direction=direction)

            else:
                if direction=='left':
                    bump_direction = 'right'
                elif direction=='right':
                    bump_direction = 'straight'
                else:
                    bump_direction = 'left'

                tf_loss[direction] = calculate_hybrid_loss(tf_logits[direction],tf_labels[direction],
                                                           tf_bump_logits[bump_direction],tf_bump_labels[bump_direction])

                tf_optimize[direction], tf_mom_update_ops[direction], tf_noncol_lr = optimize_model(tf_loss[direction],noncol_global_step,direction=direction)

        tf_test_images,tf_test_labels = models_utils.build_input_pipeline(dataset_filenames['test_dataset'],batch_size,shuffle=True,
                                                             training_data=False,use_opposite_label=False,inputs_for_sdae=False)
        tf_bump_test_images, tf_bump_test_labels = models_utils.build_input_pipeline(dataset_filenames['test_bump_dataset'], batch_size, shuffle=True,
                                                              training_data=False,use_opposite_label=True,inputs_for_sdae=False)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf_test_predictions,tf_test_actuals = predictions_and_labels_with_inputs(tf_test_images,tf_test_labels)
        tf_bump_test_predictions,tf_bump_test_actuals = predictions_and_labels_with_inputs(tf_bump_test_images,tf_bump_test_labels)

        tf.global_variables_initializer().run(session=sess)

        for epoch in range(250):
            avg_loss = []
            avg_bump_loss = []

            # Training Phase
            if not config.OPTIMIZE_HYBRID_LOSS:
                for step in range(dataset_sizes['train_dataset']//batch_size//config.FRACTION_OF_TRAINING_TO_USE):

                    if np.random.random()<0.6:
                        rand_direction = np.random.choice(['left', 'straight', 'right'])
                        l1, _, _ = sess.run([tf_loss[rand_direction], tf_optimize[rand_direction],tf_mom_update_ops[rand_direction],
                                                                            ])
                        avg_loss.append(l1)
                    else:
                        rand_direction = np.random.choice(['left', 'straight', 'right'])
                        bump_l1, _, _ = sess.run([tf_bump_loss[rand_direction], tf_bump_optimize[rand_direction],
                                                                                tf_bump_mom_update_ops[rand_direction]])
                        avg_bump_loss.append(bump_l1)

                if min_noncol_loss > np.mean(avg_loss):
                    min_noncol_loss = np.mean(avg_loss)
                else:
                    noncol_exceed_min_count += 1
                    logger.info('Increase noncol_exceed to %d',noncol_exceed_min_count)

                if noncol_exceed_min_count >=5:
                    logger.info('Stepping down collision learning rate')
                    sess.run(inc_col_gstep)
                    noncol_exceed_min_count = 0

                logger.info('\tAverage Loss for Epoch %d: %.5f' %(epoch,np.mean(avg_loss)))
                logger.info('\t\t Learning rate (non-collision): %.5f',sess.run(tf_noncol_lr))

                if min_col_loss > np.mean(avg_bump_loss):
                    min_col_loss = np.mean(avg_bump_loss)
                else:
                    col_exceed_min_count += 1
                    logger.info('Increase col_exceed to %d',col_exceed_min_count)

                if col_exceed_min_count >= 5:
                    logger.info('Stepping down non-collision learning rate')
                    sess.run(inc_noncol_gstep)
                    col_exceed_min_count = 0

                logger.info('\tAverage Bump Loss (Train) for Epoch %d: %.5f'%(epoch,np.mean(avg_bump_loss)))
                logger.info('\t\t Learning rate (collision): %.5f', sess.run(tf_col_lr))

            else:

                avg_hyb_loss = []
                rand_direction = np.random.choice(['left', 'straight', 'right'])
                l1, _, _ = sess.run(
                    [tf_loss[rand_direction], tf_optimize[rand_direction], tf_mom_update_ops[rand_direction]])
                avg_hyb_loss.append(l1)

                logger.info('\tAverage Hybrid Loss (Train) for Epoch %d: %.5f' % (epoch, np.mean(avg_hyb_loss)))

            avg_train_accuracy = []
            avg_bump_train_accuracy = []
            # Prediction Phase
            for step in range(dataset_sizes['train_dataset'] // batch_size):
                train_predictions,train_actuals = sess.run([tf_train_predictions[rand_direction],tf_train_actuals[rand_direction]])
                avg_train_accuracy.append(models_utils.soft_accuracy(train_predictions, train_actuals, use_argmin=False))

                if step < 2:
                    logger.debug('Predictions for Non-Collided data (%s)', rand_direction)
                    for pred, lbl in zip(train_predictions, train_actuals):
                        logger.debug('\t%s;%s', pred, lbl)

                if step < (dataset_sizes['train_bump_dataset']//batch_size):
                    train_bump_predictions, train_bump_actuals = sess.run(
                        [tf_train_bump_predictions[rand_direction], tf_train_bump_actuals[direction]])
                    avg_bump_train_accuracy.append(
                        models_utils.soft_accuracy(train_bump_predictions, train_bump_actuals, use_argmin=True))

                    if step < 2:
                        logger.debug('Predictions for Collided data (%s)', rand_direction)
                        for pred, lbl in zip(train_bump_predictions, train_bump_actuals):
                            logger.debug('\t%s;%s', pred, lbl)

            logger.info('\t\t Training accuracy: %.3f' % np.mean(avg_train_accuracy))
            logger.info(
                '\t\tAverage Bump Accuracy (Train) for Epoch %d: %.5f' % (epoch, np.mean(avg_bump_train_accuracy)))

            # Test Phase
            if (epoch+1)%5==0:
                test_accuracy = []
                soft_test_accuracy = []
                all_predictions, all_labels = None, None
                test_image_index = 0
                for step in range(dataset_sizes['test_dataset']//batch_size):
                    predicted_labels,actual_labels = sess.run([tf_test_predictions,tf_test_actuals])

                    test_accuracy.append(models_utils.accuracy(predicted_labels,actual_labels,use_argmin=False))
                    soft_test_accuracy.append(models_utils.soft_accuracy(predicted_labels,actual_labels,use_argmin=False))

                    #logger.debug('\t\t\tPrecision list %s', precision_list)
                    #logger.debug('\t\t\tRecall list %s', recall_list)
                    if all_predictions is None or all_labels is None:
                        all_predictions = predicted_labels
                        all_labels = actual_labels
                    else:
                        all_predictions = np.append(all_predictions,predicted_labels,axis=0)
                        all_labels = np.append(all_labels,actual_labels,axis=0)

                    if step<2:
                        logger.debug('Test Predictions (Non-Collisions)')
                        for pred,act in zip(predicted_labels,actual_labels):
                            pred_string = ''.join(['%.3f'%p+',' for p in pred.tolist()])
                            act_string = ''.join(['%.1f'%a +',' for a in act.tolist()])
                            predictionlogger.info('%d:%s:%s',test_image_index,act_string,pred_string)
                            if step < 2:
                                logger.debug('%d:%s:%s',test_image_index,act_string,pred_string)
                            test_image_index += 1

                test_noncol_precision = models_utils.precision_multiclass(all_predictions, all_labels, use_argmin=False)
                test_noncol_recall = models_utils.recall_multiclass(all_predictions,all_labels, use_argmin=False)
                predictionlogger.info('\n')
                print('\t\tAverage test accuracy: %.5f '%np.mean(test_accuracy))
                print('\t\tAverage test accuracy(soft): %.5f'%np.mean(soft_test_accuracy))
                print('\t\tAverage test precision: %s', test_noncol_precision)
                print('\t\tAverage test recall: %s', test_noncol_recall)

                bump_test_accuracy = []
                bump_soft_accuracy  = []

                all_bump_predictions, all_bump_labels = None,None
                test_image_index = 0
                for step in range(dataset_sizes['test_bump_dataset']//batch_size):
                    bump_predicted_labels, bump_actual_labels = sess.run([tf_bump_test_predictions, tf_bump_test_actuals])
                    bump_test_accuracy.append(models_utils.accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))
                    bump_soft_accuracy.append(models_utils.soft_accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))

                    if all_bump_predictions is None or all_bump_labels is None:
                        all_bump_predictions = bump_predicted_labels
                        all_bump_labels = bump_actual_labels
                    else:
                        all_bump_predictions = np.append(all_bump_predictions,bump_predicted_labels,axis=0)
                        all_bump_labels = np.append(all_bump_labels,bump_actual_labels,axis=0)

                    if step < 2:
                        logger.debug('Test Predictions (Collisions)')

                        for pred, act in zip(all_bump_predictions,all_bump_labels):
                            bpred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
                            bact_string = ''.join(['%.3f' % a + ',' for a in act.tolist()])
                            bumpPredictionlogger.info('%d:%s:%s', test_image_index, bact_string, bpred_string)
                            if step < 2:
                                logger.debug('%d:%s:%s', test_image_index, bact_string, bpred_string)
                            test_image_index += 1

                test_col_precision = models_utils.precision_multiclass(all_predictions, all_labels, use_argmin=True)
                test_col_recall = models_utils.recall_multiclass(all_predictions, all_labels, use_argmin=True)
                bumpPredictionlogger.info('\n')

                print('\t\tAverage bump test accuracy: %.5f ' % np.mean(bump_test_accuracy))
                print('\t\tAverage bump test (soft) accuracy: %.5f ' % np.mean(bump_soft_accuracy))
                print('\t\tAverage bump test precision: %s', test_col_precision)
                print('\t\tAverage bump test recall: %s', test_col_recall)

                precision_string = ''.join(['%.3f,'%test_noncol_precision[pi] for pi in range(3)])
                recall_string = ''.join(['%.3f,'%test_noncol_recall[ri] for ri in range(3)])

                col_precision_string = ''.join(['%.3f,' % test_col_precision[pi] for pi in range(3)])
                col_recall_string = ''.join(['%.3f,' % test_col_recall[ri] for ri in range(3)])

                accuracy_logger.info('%d;%.5f;%.5f;%.3f;%.3f;%.3f;%.3f;%s;%s;%s;%s',
                                     epoch,np.mean(avg_loss),np.mean(avg_bump_loss),np.mean(test_accuracy),np.mean(soft_test_accuracy),
                                     np.mean(bump_test_accuracy),np.mean(bump_soft_accuracy),
                                     precision_string,recall_string,col_precision_string,col_recall_string)
        coord.request_stop()
        coord.join(threads)
