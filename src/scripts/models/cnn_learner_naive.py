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
import cnn_variable_initializer
import cnn_optimizer
import cnn_model_visualizer

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

activation = config.ACTIVATION

max_thresh = 0.05
min_thresh = -0.05

def logits(tf_inputs):
    '''
    Inferencing the model. The input (tf_inputs) is propagated through convolutions poolings
    fully-connected layers to obtain the final softmax output
    :param tf_inputs: a batch of images (tensorflow placeholder)
    :return:
    '''
    global logger
    logger.info('Defining inference ops ...')
    with tf.name_scope('infer'):
        for si, scope in enumerate(config.TF_ANG_SCOPES):
            with tf.variable_scope(scope,reuse=True) as sc:

                if 'conv' in scope:
                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                    logger.info('\t\tConvolution with %s activation for %s',activation,scope)
                    if si == 0:
                        h = models_utils.activate(tf.nn.conv2d(tf_inputs,weight,strides=config.TF_ANG_STRIDES[scope],padding='SAME')+bias,activation,name='hidden')
                    else:
                        h = models_utils.activate(tf.nn.conv2d(h, weight, strides=config.TF_ANG_STRIDES[scope], padding='SAME') + bias, activation,
                                       name='hidden')
                elif 'pool' in scope:
                    logger.info('\t\tMax pooling for %s', scope)
                    h = tf.nn.max_pool(h,config.TF_ANG_VAR_SHAPES_NAIVE[scope],config.TF_ANG_STRIDES[scope],padding='SAME',name='pool_hidden')

                else:
                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                    # Reshaping required for the first fulcon layer
                    if scope == 'out':
                        logger.info('\t\tFully-connected with output Logits for %s',scope)
                        h = tf.matmul(h, weight) + bias

                    elif 'fc' in scope:
                        if scope == config.TF_FIRST_FC_ID:
                            h_shape = h.get_shape().as_list()
                            logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s', scope, h_shape)
                            h = tf.reshape(h, [config.BATCH_SIZE, h_shape[1] * h_shape[2] * h_shape[3]])
                            h = models_utils.activate(tf.matmul(h, weight) + bias, activation)

                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError

    return h

def logits_with_batchnorm(tf_inputs,is_training):
    '''
    Inferencing the model. The input (tf_inputs) is propagated through convolutions poolings
    fully-connected layers to obtain the final softmax output
    :param tf_inputs: a batch of images (tensorflow placeholder)
    :return:
    '''
    global logger
    logger.info('Defining inference ops ...')
    with tf.name_scope('infer'):
        for si, scope in enumerate(config.TF_ANG_SCOPES):
            with tf.variable_scope(scope,reuse=True) as sc:

                if 'conv' in scope:
                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                    logger.info('\t\tConvolution with %s activation for %s',activation,scope)
                    if si == 0:
                        h = tf.nn.conv2d(tf_inputs,weight,strides=config.TF_ANG_STRIDES[scope],padding='SAME')+bias
                        with tf.variable_scope('bn'):
                            beta = tf.get_variable("beta", h.get_shape().as_list()[1:], initializer=tf.constant_initializer(0.0))
                            gamma = tf.get_variable("gamma", h.get_shape().as_list()[1:], initializer=tf.constant_initializer(0.0))
                        h = tf.contrib.layers.batch_norm(h, center=True, scale=True, is_training=is_training, scope='bn')
                        h = models_utils.activate(h,activation,name='hidden')
                    else:
                        h = tf.nn.conv2d(h, weight, strides=config.TF_ANG_STRIDES[scope], padding='SAME') + bias
                        with tf.variable_scope('bn'):
                            beta = tf.get_variable("beta", h.get_shape().as_list()[1:], initializer=tf.constant_initializer(0.0))
                            gamma = tf.get_variable("gamma", h.get_shape().as_list()[1:], initializer=tf.constant_initializer(0.0))
                        h = tf.contrib.layers.batch_norm(h, center=True, scale=True, is_training=is_training,
                                                         scope='bn')
                        h = models_utils.activate(h, activation, name='hidden')
                elif 'pool' in scope:
                    logger.info('\t\tMax pooling for %s', scope)
                    h = tf.nn.max_pool(h,config.TF_ANG_VAR_SHAPES_NAIVE[scope],config.TF_ANG_STRIDES[scope],padding='SAME',name='pool_hidden')

                else:
                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                    # Reshaping required for the first fulcon layer
                    if scope == 'out':
                        logger.info('\t\tFully-connected with output Logits for %s',scope)
                        h = tf.matmul(h, weight) + bias

                    elif 'fc' in scope:
                        if scope == config.TF_FIRST_FC_ID:
                            h_shape = h.get_shape().as_list()
                            logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s', scope, h_shape)
                            h = tf.reshape(h, [config.BATCH_SIZE, h_shape[1] * h_shape[2] * h_shape[3]])

                            with tf.variable_scope('bn'):
                                beta = tf.get_variable("beta", h.get_shape().as_list()[1:],
                                                       initializer=tf.constant_initializer(0.0))
                                gamma = tf.get_variable("gamma", h.get_shape().as_list()[1:],
                                                        initializer=tf.constant_initializer(0.0))
                            h = tf.contrib.layers.batch_norm(h, center=True, scale=True, is_training=is_training,
                                                             scope='bn')

                            h = models_utils.activate(tf.matmul(h, weight) + bias, activation)

                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError

    return h


def predictions_with_logits(logits):
    pred = tf.nn.tanh(logits)
    return pred


def predictions_with_inputs(tf_inputs):
    tf_logits = logits(tf_inputs)
    return tf.nn.tanh(tf_logits)


def calculate_hybrid_loss(tf_noncol_logits,tf_noncol_labels,tf_col_logits,tf_col_labels):
    tf_col_out = tf_col_logits
    tf_noncol_out = tf_noncol_logits

    tf_col_arg = tf.cast(tf.reduce_min(tf_col_labels, axis=1), dtype=tf.float32)
    tf_noncol_arg = tf.cast(tf.reduce_max(tf_noncol_labels, axis=1), dtype=tf.float32)

    loss = tf.reduce_mean((tf.nn.tanh(tf_col_out) - tf_col_arg)**2 + (tf.nn.tanh(tf_noncol_out) - tf_noncol_arg)**2)
    return loss


def calculate_loss(tf_logits, tf_labels):
    mask_predictions = True
    random_masking = True
    tf_out = tf_logits
    tf_label_weights = tf.reduce_mean(tf_labels, axis=[0])

    if mask_predictions and not random_masking:
        masked_preds = tf.nn.tanh(tf_logits) * tf.cast(tf.not_equal(tf_labels,0.0),dtype=tf.float32)
        loss = tf.reduce_mean(tf.reduce_sum((masked_preds - tf_labels)**2 *(1.0-tf.abs(tf_label_weights)),axis=[1]),axis=[0])
    elif random_masking:
        rand_mask = tf.cast(tf.greater(tf.random_normal([config.BATCH_SIZE,3], dtype=tf.float32),0.0),dtype=tf.float32)
        masked_preds = tf.nn.tanh(tf_logits) * tf.cast(tf.not_equal(tf_labels, 0.0), dtype=tf.float32) * rand_mask
        loss = tf.reduce_mean(tf.reduce_sum((masked_preds - tf_labels) ** 2 *(1.0-tf.abs(tf_label_weights)), axis=[1]), axis=[0])
    else:
        # use appropriately to weigh output *(1-tf.abs(tf_label_weights))
        loss = tf.reduce_mean(tf.reduce_sum(((tf.nn.tanh(tf_out) - tf_labels)**2)*(1.0-tf.abs(tf_label_weights)),axis=[1]),axis=[0])

    return loss


def inc_gstep(gstep):
    return tf.assign(gstep,gstep+1)


def setup_loggers():
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
    accuracy_logger.info('#Epoch;Non-collision Accuracy;Non-collision Accuracy(Soft);Collision Accuracy;' +
                         'Collision Accuracy (Soft);Loss (Non-collision); Loss (collision); Preci-NC-L;Preci-NC-S;Preci-NC-R;;' +
                         'Rec-NC-L;Rec-NC-S;Rec-NC-R;;Preci-C-L;Preci-C-S;Preci-C-R;;Rec-C-L;Rec-C-S;Rec-C-R;')

    return predictionlogger, bumpPredictionlogger, accuracy_logger


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

    dataset_filenames = {
        'train_dataset': ['..' + os.sep + 'data_indoor_1_1000' + os.sep + 'image-direction-shuffled.tfrecords'],
        'train_bump_dataset': [
            '..' + os.sep + 'data_indoor_1_bump_200' + os.sep + 'image-direction-shuffled.tfrecords'],
        'test_dataset': ['..' + os.sep + 'data_grande_salle_1000' + os.sep + 'image-direction-shuffled.tfrecords'],
        'test_bump_dataset': [
            '..' + os.sep + 'data_grande_salle_bump_200' + os.sep + 'image-direction-shuffled.tfrecords']
    }

    dataset_sizes = {'train_dataset': 1000 + 1000,
                     'train_bump_dataset': 400,
                     'test_dataset': 1000,
                     'test_bump_dataset': 200}

    min_col_loss, min_noncol_loss = 10000, 10000
    col_exceed_min_count, noncol_exceed_min_count = 0, 0

    predictionlogger, bumpPredictionlogger, accuracy_logger = setup_loggers()

    graph = tf.Graph()
    configp = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.InteractiveSession(graph=graph,config=configp)

    with sess.as_default() and graph.as_default():
        cnn_variable_initializer.set_from_main(sess,graph)
        cnn_variable_initializer.build_tensorflw_variables_naive()
        models_utils.set_from_main(sess,graph,logger)

        noncol_global_step = tf.Variable(0, trainable=False)
        col_global_step = tf.Variable(0, trainable=False)

        inc_noncol_gstep = inc_gstep(noncol_global_step)
        inc_col_gstep = inc_gstep(col_global_step)

        all_train_files,all_bump_files = [],[]
        all_test_files,all_bump_test_files = [],[]

        all_train_files = dataset_filenames['train_dataset']
        all_bump_files = dataset_filenames['train_bump_dataset']
        all_test_files = dataset_filenames['test_dataset']
        all_bump_test_files = dataset_filenames['test_bump_dataset']

        tf_img_ids, tf_images,tf_labels = models_utils.build_input_pipeline(
            all_train_files,config.BATCH_SIZE,shuffle=True,
            training_data=True,use_opposite_label=False,inputs_for_sdae=False)

        tf_bump_img_ids, tf_bump_images, tf_bump_labels = models_utils.build_input_pipeline(
            all_bump_files, config.BATCH_SIZE, shuffle=True,
            training_data=True, use_opposite_label=True,inputs_for_sdae=False)

        tf_logits = logits(tf_images)
        tf_loss = calculate_loss(tf_logits, tf_labels)

        tf_bump_logits = logits(tf_bump_images)
        tf_bump_loss = calculate_loss(tf_bump_logits, tf_bump_labels)

        tf_optimize, tf_grads_and_vars = cnn_optimizer.optimize_model_naive_no_momentum(tf_loss, noncol_global_step,tf.global_variables())
        tf_bump_optimize, _ = cnn_optimizer.optimize_model_naive_no_momentum(tf_bump_loss, col_global_step, tf.global_variables())

        tf_train_predictions = predictions_with_inputs(tf_images)
        tf_train_bump_predictions = predictions_with_inputs(tf_bump_images)

        tf_test_img_ids, tf_test_images,tf_test_labels = models_utils.build_input_pipeline(all_test_files,config.BATCH_SIZE,shuffle=False,
                                                                          training_data=False, use_opposite_label=False, inputs_for_sdae=False)
        tf_bump_test_img_ids, tf_bump_test_images, tf_bump_test_labels = models_utils.build_input_pipeline(all_bump_test_files, config.BATCH_SIZE, shuffle=False,
                                                                                     training_data=False, use_opposite_label=True, inputs_for_sdae=False)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf_test_predictions = predictions_with_inputs(tf_test_images)
        tf_bump_test_predictions = predictions_with_inputs(tf_bump_test_images)

        tf.global_variables_initializer().run(session=sess)

        for epoch in range(150):
            avg_loss = []
            avg_train_accuracy = []
            avg_bump_loss = []
            avg_bump_train_accuracy = []

            # Training with Non-Bump Data
            for step in range(dataset_sizes['train_dataset']//config.BATCH_SIZE):

                if np.random.random()<0.7:
                    l1, _, pred,train_labels = sess.run([tf_loss, tf_optimize, tf_train_predictions,tf_labels])
                    avg_loss.append(l1)
                    avg_train_accuracy.append(models_utils.soft_accuracy(pred,train_labels,use_argmin=False, max_thresh=max_thresh, min_thresh=min_thresh))

                    if step < 2:
                        logger.debug('Predictions for Non-Collided data')
                        for pred,lbl in zip(pred,train_labels):
                            logger.debug('\t%s;%s',pred,lbl)

                else:
                    bump_l1,bump_logits, _, bump_pred, train_bump_labels = sess.run([tf_bump_loss, tf_bump_logits,
                                                                                     tf_bump_optimize,
                                                                                     tf_train_bump_predictions, tf_bump_labels])

                    avg_bump_loss.append(bump_l1)
                    avg_bump_train_accuracy.append(models_utils.soft_accuracy(bump_pred,train_bump_labels,use_argmin=True, max_thresh=max_thresh, min_thresh=min_thresh))

                    if step<2:
                        logger.debug('Predictions for Collided data')
                        for pred,lbl in zip(bump_pred,train_bump_labels):
                            logger.debug('\t%s;%s',pred,lbl)

            if min_noncol_loss > np.mean(avg_loss):
                min_noncol_loss = np.mean(avg_loss)
            else:
                noncol_exceed_min_count += 1
                logger.info('Increase noncol_exceed to %d',noncol_exceed_min_count)

            if noncol_exceed_min_count >=3:
                logger.info('Stepping down collision learning rate')
                sess.run(inc_col_gstep)
                noncol_exceed_min_count = 0

            logger.info('\tAverage Loss for Epoch %d: %.5f' %(epoch,np.mean(avg_loss)))
            logger.info('\t\t Training accuracy: %.3f'%np.mean(avg_train_accuracy))

            if min_col_loss > np.mean(avg_bump_loss):
                min_col_loss = np.mean(avg_bump_loss)
            else:
                col_exceed_min_count += 1
                logger.info('Increase col_exceed to %d',col_exceed_min_count)

            if col_exceed_min_count >= 3:
                logger.info('Stepping down non-collision learning rate')
                sess.run(inc_noncol_gstep)
                col_exceed_min_count = 0

            logger.info('\tAverage Bump Loss (Train) for Epoch %d: %.5f'%(epoch,np.mean(avg_bump_loss)))
            logger.info('\t\tAverage Bump Accuracy (Train) for Epoch %d: %.5f' % (epoch, np.mean(avg_bump_train_accuracy)))

            if (epoch+1)%5==0:
                test_accuracy = []
                soft_test_accuracy = []
                bump_test_accuracy = []
                bump_soft_accuracy  = []
                all_predictions, all_labels = None, None
                all_bump_predictions, all_bump_labels = None, None

                test_image_index = 0
                for step in range(dataset_sizes['test_dataset']//config.BATCH_SIZE):
                    predicted_labels,actual_labels = sess.run([tf_test_predictions,tf_test_labels])

                    test_accuracy.append(models_utils.accuracy(predicted_labels,actual_labels,use_argmin=False))
                    soft_test_accuracy.append(models_utils.soft_accuracy(predicted_labels,actual_labels,use_argmin=False, max_thresh=max_thresh, min_thresh=min_thresh))

                    if all_predictions is None or all_labels is None:
                        all_predictions = predicted_labels
                        all_labels = actual_labels
                    else:
                        all_predictions = np.append(all_predictions,predicted_labels,axis=0)
                        all_labels = np.append(all_labels,actual_labels,axis=0)

                    if step<5:
                        logger.debug('Test Predictions (Non-Collisions)')
                    for pred,act in zip(predicted_labels,actual_labels):
                        pred_string = ''.join(['%.3f'%p+',' for p in pred.tolist()])
                        act_string = ''.join([str(int(a))+',' for a in act.tolist()])
                        predictionlogger.info('%d:%s:%s',test_image_index,act_string,pred_string)
                        if step < 5:
                            logger.debug('%d:%s:%s',test_image_index,act_string,pred_string)
                        test_image_index += 1
                    predictionlogger.info('\n')

                for step in range(dataset_sizes['test_bump_dataset']//config.BATCH_SIZE):
                    bump_predicted_labels, bump_actual_labels = sess.run([tf_bump_test_predictions, tf_bump_test_labels])
                    bump_test_accuracy.append(models_utils.accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))
                    bump_soft_accuracy.append(models_utils.soft_accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True, max_thresh=max_thresh, min_thresh=min_thresh))

                    if all_bump_predictions is None or all_bump_labels is None:
                        all_bump_predictions = bump_predicted_labels
                        all_bump_labels = bump_actual_labels
                    else:
                        all_bump_predictions = np.append(all_bump_predictions,bump_predicted_labels,axis=0)
                        all_bump_labels = np.append(all_bump_labels,bump_actual_labels,axis=0)

                if step < 5:
                    logger.debug('Test Predictions (Collisions)')
                test_image_index =0
                for pred, act in zip(all_bump_predictions,all_bump_labels):
                    bpred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
                    bact_string = ''.join([str(int(a)) + ',' for a in act.tolist()])
                    bumpPredictionlogger.info('%d:%s:%s', test_image_index, bact_string, bpred_string)
                    if step < 5:
                        logger.debug('%d:%s:%s', test_image_index, bact_string, bpred_string)
                    test_image_index += 1

                bumpPredictionlogger.info('\n')

                test_col_precision = models_utils.precision_multiclass(all_bump_predictions, all_bump_labels, use_argmin=True,
                                                                          max_thresh=max_thresh, min_thresh=min_thresh)
                test_col_recall = models_utils.recall_multiclass(all_bump_predictions, all_bump_labels, use_argmin=True,
                                                                    max_thresh=max_thresh, min_thresh=min_thresh)

                test_noncol_precision = models_utils.precision_multiclass(all_predictions, all_labels, use_argmin=False,
                                                                          max_thresh=max_thresh, min_thresh=min_thresh)
                test_noncol_recall = models_utils.recall_multiclass(all_predictions, all_labels, use_argmin=False,
                                                                    max_thresh=max_thresh, min_thresh=min_thresh)

                print('\t\tAverage test accuracy: %.5f ' % np.mean(test_accuracy))
                print('\t\tAverage test accuracy(soft): %.5f' % np.mean(soft_test_accuracy))
                print('\t\tAverage test precision: %s', test_noncol_precision)
                print('\t\tAverage test recall: %s', test_noncol_recall)

                print('\t\tAverage bump test accuracy: %.5f ' % np.mean(bump_test_accuracy))
                print('\t\tAverage bump test (soft) accuracy: %.5f ' % np.mean(bump_soft_accuracy))
                print('\t\tAverage test bump precision: %s', test_col_precision)
                print('\t\tAverage test bump recall: %s', test_col_recall)

                noncol_precision_string = ''.join(['%.3f;'%test_noncol_precision[pi] for pi in range(3)])
                noncol_recall_string = ''.join(['%.3f;'%test_noncol_recall[ri] for ri in range(3)])
                col_precision_string = ''.join(['%.3f;' % test_col_precision[pi] for pi in range(3)])
                col_recall_string = ''.join(['%.3f;' % test_col_recall[ri] for ri in range(3)])

                accuracy_logger.info('%d;%.3f;%.3f;%.3f;%.3f;%.5f;%.5f;%s;%s;%s;%s',epoch,np.mean(test_accuracy),np.mean(soft_test_accuracy),
                                     np.mean(bump_test_accuracy),np.mean(bump_soft_accuracy),np.mean(avg_loss),np.mean(avg_bump_loss),
                                     noncol_precision_string,noncol_recall_string, col_precision_string, col_recall_string)


        logger.info('Saving CNN Model')
        cnn_model_visualizer.save_cnn_hyperparameters(IMG_DIR,config.TF_ANG_VAR_SHAPES_NAIVE,config.TF_ANG_STRIDES, 'hyperparams_%d.pickle'%epoch)
        cnn_model_visualizer.save_cnn_weights_naive(IMG_DIR, sess, 'cnn_model_%d.ckpt'%epoch)

        coord.request_stop()
        coord.join(threads)
