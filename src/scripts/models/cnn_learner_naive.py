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
import visualizer
import dataset_name_factory


logging_level = logging.INFO
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

logger = logging.getLogger('Logger')
logger.setLevel(logging_level)
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
output_activation = 'sigmoid'

max_thresh = 0.55
min_thresh = 0.45

kernel_size_dict = config.TF_ANG_VAR_SHAPES_NAIVE
stride_dict = config.TF_ANG_STRIDES
scope_list = config.TF_ANG_SCOPES


def set_scope_kernel_stride_dicts(sc_list, k_dict, s_dict):
    global kernel_size_dict,stride_dict,scope_list
    scope_list = sc_list
    kernel_size_dict = k_dict
    stride_dict = s_dict


def logits(tf_inputs,is_training):
    '''
    Inferencing the model. The input (tf_inputs) is propagated through convolutions poolings
    fully-connected layers to obtain the final softmax output
    :param tf_inputs: a batch of images (tensorflow placeholder)
    :return:
    '''
    global logger
    logger.info('Defining inference ops ...')
    with tf.name_scope('infer'):
        for si, scope in enumerate(scope_list):
            with tf.variable_scope(scope,reuse=True) as sc:

                if 'conv' in scope:
                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                    logger.info('\t\tConvolution with %s activation for %s',activation,scope)
                    if si == 0:
                        if is_training and config.USE_DROPOUT:
                            tf_inputs = tf.nn.dropout(tf_inputs,1.0 - config.IN_DROPOUT,name='input_dropout')
                        h = models_utils.activate(tf.nn.conv2d(tf_inputs,weight,strides=stride_dict[scope],padding='SAME')+bias,activation,name='hidden')
                    else:
                        h = models_utils.activate(tf.nn.conv2d(h, weight, strides=stride_dict[scope], padding='SAME') + bias, activation,
                                       name='hidden')
                elif 'pool' in scope:
                    logger.info('\t\tMax pooling for %s', scope)
                    h = tf.nn.max_pool(h,kernel_size_dict[scope],stride_dict[scope],padding='SAME',name='pool_hidden')
                    if is_training and config.USE_DROPOUT:
                        h = tf.nn.dropout(h, 1.0 - config.LAYER_DROPOUT, name='pool_dropout')
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

                            if is_training and config.USE_DROPOUT:
                                h = tf.nn.dropout(h,1.0-config.LAYER_DROPOUT, name='hidden_dropout')

                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError

    return h


def predictions_with_logits(logits):
    pred = models_utils.activate(logits,activation_type=output_activation)
    return pred


def predictions_with_inputs(tf_inputs):
    tf_logits = logits(tf_inputs,is_training=False)
    return models_utils.activate(tf_logits,activation_type=output_activation)


def calculate_hybrid_loss(tf_noncol_logits,tf_noncol_labels,tf_col_logits,tf_col_labels):
    tf_col_out = tf_col_logits
    tf_noncol_out = tf_noncol_logits

    tf_col_arg = tf.cast(tf.reduce_min(tf_col_labels, axis=1), dtype=tf.float32)
    tf_noncol_arg = tf.cast(tf.reduce_max(tf_noncol_labels, axis=1), dtype=tf.float32)

    loss = tf.reduce_mean((tf.nn.tanh(tf_col_out) - tf_col_arg)**2 + (tf.nn.tanh(tf_noncol_out) - tf_noncol_arg)**2)
    return loss


def calculate_loss(tf_logits, tf_labels, weigh_by_frequency=False):
    mask_predictions = False
    random_masking = False
    use_heuristic_weights = True

    if not use_heuristic_weights:
        tf_label_weights_inv = 1.0 - tf.reduce_mean(tf.abs(tf_labels), axis=[0])
    else:
        tf_label_weights_inv =tf_labels * (tf.constant([1.0,0.5,1.0],dtype=tf.float32)) # Because we use 1-weights for weighing

    if mask_predictions and not random_masking:
        masked_preds = models_utils.activate(tf_logits,activation_type=output_activation) * tf.cast(tf.not_equal(tf_labels,0.0),dtype=tf.float32)
        if weigh_by_frequency:
            loss = tf.reduce_mean(tf.reduce_sum((masked_preds - tf_labels)**2 *tf_label_weights_inv,axis=[1]),axis=[0])
        else:
            loss = tf.reduce_mean(tf.reduce_sum((masked_preds - tf_labels) ** 2, axis=[1]),axis=[0])
    elif random_masking:
        rand_mask = tf.cast(tf.greater(tf.random_normal([config.BATCH_SIZE,3], dtype=tf.float32),0.0),dtype=tf.float32)
        masked_preds = models_utils.activate(tf_logits,activation_type=output_activation) * (tf.cast(tf.not_equal(tf_labels, 0.0), dtype=tf.float32) + rand_mask)
        if weigh_by_frequency:
            loss = tf.reduce_mean(tf.reduce_sum((masked_preds - tf_labels) ** 2 *tf_label_weights_inv, axis=[1]), axis=[0])
        else:
            loss = tf.reduce_mean(tf.reduce_sum((masked_preds - tf_labels) ** 2 , axis=[1]), axis=[0])
    else:
        # use appropriately to weigh output *(1-tf_label_weights)
        if weigh_by_frequency:
            loss = tf.reduce_mean(tf.reduce_sum(((models_utils.activate(tf_logits,activation_type=output_activation) - tf_labels)**2)*tf_label_weights_inv,axis=[1]),axis=[0])
        else:
            loss = tf.reduce_mean(tf.reduce_sum(
                ((models_utils.activate(tf_logits, activation_type=output_activation) - tf_labels) ** 2), axis=[1]), axis=[0])

    return loss


def calculate_loss_softmax(tf_logits, tf_labels):

    tf_label_weights = tf.abs(tf.reduce_mean(tf_labels, axis=[0]))
    # use appropriately to weigh output *(1-tf_label_weights)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels,logits=tf_logits),axis=[0])

    return loss


def calculate_loss_sigmoid(tf_logits, tf_labels):

    tf_label_weights = tf.abs(tf.reduce_mean(tf_labels, axis=[0]))
    # use appropriately to weigh output *(1-tf_label_weights)
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_labels,logits=tf_logits),axis=[0])

    return loss


def inc_gstep(gstep):
    return tf.assign(gstep,gstep+1)


def setup_loggers(filename_suffix):

    trainPredictionLogger = logging.getLogger('TrainPredictionLogger-%s' % filename_suffix)
    trainPredictionLogger.setLevel(logging.INFO)
    tr_fh = logging.FileHandler(IMG_DIR + os.sep + 'train-predictions-%s.log' % filename_suffix, mode='w')
    tr_fh.setFormatter(logging.Formatter('%(message)s'))
    tr_fh.setLevel(logging.INFO)
    trainPredictionLogger.addHandler(tr_fh)
    trainPredictionLogger.info('#ID:Actual:Predicted:Correct?')

    validPredictionlogger = logging.getLogger('ValidPredictionLogger-%s' % filename_suffix)
    validPredictionlogger.setLevel(logging.INFO)
    v_fh = logging.FileHandler(IMG_DIR + os.sep + 'valid-predictions-%s.log' % filename_suffix, mode='w')
    v_fh.setFormatter(logging.Formatter('%(message)s'))
    v_fh.setLevel(logging.INFO)
    validPredictionlogger.addHandler(v_fh)
    validPredictionlogger.info('#ID:Actual:Predicted:Correct?')

    predictionlogger = logging.getLogger('TestPredictionLogger-%s'%filename_suffix)
    predictionlogger.setLevel(logging.INFO)
    fh = logging.FileHandler(IMG_DIR + os.sep + 'predictions-%s.log'%filename_suffix, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    predictionlogger.addHandler(fh)
    predictionlogger.info('#ID:Actual:Predicted:Correct?')

    TestBumpPredictionLogger = logging.getLogger('TestBumpPredictionLogger-%s'%filename_suffix)
    TestBumpPredictionLogger.setLevel(logging.INFO)
    bumpfh = logging.FileHandler(IMG_DIR + os.sep + 'bump_predictions-%s.log'%filename_suffix, mode='w')
    bumpfh.setFormatter(logging.Formatter('%(message)s'))
    bumpfh.setLevel(logging.INFO)
    TestBumpPredictionLogger.addHandler(bumpfh)
    TestBumpPredictionLogger.info('#ID:Actual:Predicted:Correct?')

    SummaryLogger = logging.getLogger('AccuracyLogger-%s'%filename_suffix)
    SummaryLogger.setLevel(logging.INFO)
    accuracyFH = logging.FileHandler(IMG_DIR + os.sep + 'accuracy-%s.log'%filename_suffix, mode='w')
    accuracyFH.setFormatter(logging.Formatter('%(message)s'))
    accuracyFH.setLevel(logging.INFO)
    SummaryLogger.addHandler(accuracyFH)
    SummaryLogger.info('#Epoch;Non-collision Accuracy;Non-collision Accuracy(Soft);Collision Accuracy;' +
                         'Collision Accuracy (Soft);Loss (Non-collision); Loss (collision); Preci-NC-L;Preci-NC-S;Preci-NC-R;;' +
                         'Rec-NC-L;Rec-NC-S;Rec-NC-R;;Preci-C-L;Preci-C-S;Preci-C-R;;Rec-C-L;Rec-C-S;Rec-C-R;')

    return {'test-col-logger':TestBumpPredictionLogger, 'test-noncol-logger':predictionlogger, 'summary-logger':SummaryLogger,
            'train-logger':trainPredictionLogger, 'valid-logger':validPredictionlogger}


def test_the_model(sess, tf_test_img_ids, tf_test_images,tf_test_labels,
                   tf_test_predictions, test_dataset_size,
                   tf_bump_test_img_ids, tf_bump_test_images, tf_bump_test_labels,
                   tf_bump_test_predictions, test_bump_dataset_size, epoch, sub_folder,include_bump_data):
    '''
    Test the Trained CNN by predicting all test non-collision data and collision data
    Things done,
    Calculate accuracies (collision and non-collision) (hard and soft)
    Calculate Precision and Recall (collision and non-collision) (for each direction)
    Save images categorized by the predicted navigationd direction

    :param tf_test_img_ids:
    :param tf_test_images:
    :param tf_test_labels:
    :param tf_test_predictions:
    :param test_dataset_size:
    :param tf_bump_test_img_ids:
    :param tf_bump_test_images:
    :param tf_bump_test_labels:
    :param tf_bump_test_predictions:
    :param test_bump_dataset_size:
    :param epoch:
    :return: Hard_Accuracy(Non-col), Soft Accuracy (Non-col), Hard_Accuracy(col), Soft Accuracy (col),
     Precision (Non-col), Recall (Non-col), Precision (col), Recall (col),


    '''

    test_results = {}
    test_accuracy = []
    soft_test_accuracy = []
    bump_test_accuracy = []
    bump_soft_accuracy = []

    all_predictions, all_labels, all_img_ids, all_images = None, None, None, None
    all_bump_predictions, all_bump_labels, all_bump_img_ids, all_bump_images = None, None, None, None

    test_image_index = 0
    for step in range(test_dataset_size // config.BATCH_SIZE ):
        predicted_labels, actual_labels, test_ids, test_images = sess.run(
            [tf_test_predictions, tf_test_labels, tf_test_img_ids, tf_test_images])

        test_accuracy.append(models_utils.accuracy(predicted_labels, actual_labels, use_argmin=False))
        soft_test_accuracy.append(
            models_utils.soft_accuracy(predicted_labels, actual_labels, use_argmin=False, max_thresh=max_thresh,
                                       min_thresh=min_thresh))

        if all_predictions is None or all_labels is None:
            all_predictions = predicted_labels
            all_labels = actual_labels
            all_img_ids = test_ids
            all_images = test_images
        else:
            all_predictions = np.append(all_predictions, predicted_labels, axis=0)
            all_labels = np.append(all_labels, actual_labels, axis=0)
            all_img_ids = np.append(all_img_ids, test_ids, axis=0)
            all_images = np.append(all_images, test_images, axis=0)

        if step < 5:
            logger.debug('Test Predictions (Non-Collisions)')
        for pred, act in zip(predicted_labels, actual_labels):
            pred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
            act_string = ''.join([str(int(a)) + ',' for a in act.tolist()])
            is_correct = np.argmax(pred)==np.argmax(act)
            TestPredictionLogger.info('%d:%s:%s:%s', test_image_index, act_string, pred_string,is_correct)

            if step < 5:
                logger.debug('%d:%s:%s', test_image_index, act_string, pred_string)
            test_image_index += 1

        TestPredictionLogger.info('\n')

    print('\t\tAverage test accuracy: %.5f ' % np.mean(test_accuracy))
    print('\t\tAverage test accuracy(soft): %.5f' % np.mean(soft_test_accuracy))

    if include_bump_data:
        for step in range(test_bump_dataset_size // config.BATCH_SIZE):
            bump_predicted_labels, bump_actual_labels, bump_test_ids, bump_test_images = sess.run(
                [tf_bump_test_predictions, tf_bump_test_labels, tf_bump_test_img_ids, tf_bump_test_images])
            bump_test_accuracy.append(
                models_utils.accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))
            bump_soft_accuracy.append(
                models_utils.soft_accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True,
                                           max_thresh=max_thresh, min_thresh=min_thresh))

            if all_bump_predictions is None or all_bump_labels is None:
                all_bump_predictions = bump_predicted_labels
                all_bump_labels = bump_actual_labels
                all_bump_img_ids = bump_test_ids
                all_bump_images = bump_test_images
            else:
                all_bump_predictions = np.append(all_bump_predictions, bump_predicted_labels, axis=0)
                all_bump_labels = np.append(all_bump_labels, bump_actual_labels, axis=0)
                all_bump_img_ids = np.append(all_bump_img_ids, bump_test_ids, axis=0)
                all_bump_images = np.append(all_bump_images, bump_test_images, axis=0)

        print('\t\tAverage bump test accuracy: %.5f ' % np.mean(bump_test_accuracy))
        print('\t\tAverage bump test (soft) accuracy: %.5f ' % np.mean(bump_soft_accuracy))


        if step < 5:
            logger.debug('Test Predictions (Collisions)')
        test_image_index = 0
        for pred, act in zip(all_bump_predictions, all_bump_labels):
            bpred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
            bact_string = ''.join([str(int(a)) + ',' for a in act.tolist()])
            is_correct = np.argmin(pred)==np.argmin(act)
            TestBumpPredictionLogger.info('%d:%s:%s:%s', test_image_index, bact_string, bpred_string,is_correct)
            if step < 5:
                logger.debug('%d:%s:%s', test_image_index, bact_string, bpred_string)
            test_image_index += 1

        TestBumpPredictionLogger.info('\n')

        test_col_precision = models_utils.precision_multiclass(all_bump_predictions, all_bump_labels,
                                                               use_argmin=True,
                                                               max_thresh=max_thresh, min_thresh=min_thresh)
        test_col_recall = models_utils.recall_multiclass(all_bump_predictions, all_bump_labels, use_argmin=True,
                                                         max_thresh=max_thresh, min_thresh=min_thresh)

        print('\t\tAverage test bump precision: %s', test_col_precision)
        print('\t\tAverage test bump recall: %s', test_col_recall)


    test_noncol_precision = models_utils.precision_multiclass(all_predictions, all_labels, use_argmin=False,
                                                              max_thresh=max_thresh, min_thresh=min_thresh)
    test_noncol_recall = models_utils.recall_multiclass(all_predictions, all_labels, use_argmin=False,
                                                        max_thresh=max_thresh, min_thresh=min_thresh)

    print('\t\tAverage test precision: %s', test_noncol_precision)
    print('\t\tAverage test recall: %s', test_noncol_recall)

    predicted_hard_ids_sorted, predicted_bump_hard_ids_sorted = {}, {}
    predicted_hard_ids_sorted_best, predicted_bump_hard_ids_sorted_best = {}, {}

    for di, direct in enumerate(['left', 'straight', 'right']):
        predicted_hard_ids_sorted_best[direct] = models_utils.get_id_vector_for_predicted_samples_best(
            all_img_ids, all_predictions, all_labels, di, True, False, max_thresh, min_thresh
        )
        predicted_hard_ids_sorted[direct] = models_utils.get_id_vector_for_predicted_samples(
            all_img_ids, all_predictions, all_labels, di, True, False
        )

        if include_bump_data:
            predicted_bump_hard_ids_sorted_best[direct] = models_utils.get_id_vector_for_predicted_samples_best(
                all_bump_img_ids, all_bump_predictions, all_bump_labels, di, True, True, max_thresh, min_thresh
            )
            predicted_bump_hard_ids_sorted[direct] = models_utils.get_id_vector_for_predicted_samples(
                all_bump_img_ids, all_bump_predictions, all_bump_labels, di, True, True
            )

    image_list = np.split(all_images, all_images.shape[0])
    id_list = all_img_ids.tolist()
    dict_id_image = dict(zip(id_list, image_list))

    if include_bump_data:
        bump_image_list = np.split(all_bump_images, all_bump_images.shape[0])
        bump_id_list = all_bump_img_ids.tolist()
        bump_dict_id_image = dict(zip(bump_id_list, bump_image_list))

    logger.info('correct hard img ids for: %s', predicted_hard_ids_sorted_best)

    visualizer.save_fig_with_predictions_for_direction(predicted_hard_ids_sorted_best, dict_id_image,
                                                       sub_folder + os.sep + 'predicted_best_hard_%d.png' % (
                                                           epoch))
    visualizer.save_fig_with_predictions_for_direction(predicted_hard_ids_sorted, dict_id_image,
                                                       sub_folder + os.sep + 'predicted_hard_%d.png' % (
                                                           epoch))

    if include_bump_data:
        visualizer.save_fig_with_predictions_for_direction(predicted_bump_hard_ids_sorted_best, bump_dict_id_image,
                                                           sub_folder + os.sep + 'predicted_best_bump_%d.png' % (
                                                               epoch))
        visualizer.save_fig_with_predictions_for_direction(predicted_bump_hard_ids_sorted, bump_dict_id_image,
                                                           sub_folder + os.sep + 'predicted_bump_%d.png' % (
                                                               epoch))

    test_results['noncol-accuracy-hard'] = np.mean(test_accuracy)
    test_results['noncol-accuracy-soft'] = np.mean(soft_test_accuracy)
    test_results['noncol-precision'] = test_noncol_precision
    test_results['noncol-recall'] = test_noncol_recall

    if include_bump_data:
        test_results['col-accuracy-hard'] = np.mean(bump_test_accuracy)
        test_results['col-accuracy-soft'] = np.mean(bump_soft_accuracy)
        test_results['col-precision'] = test_col_precision
        test_results['col-recall'] = test_col_recall

    return test_results


def train_with_non_collision(sess,
                             tf_images,          tf_labels,           dataset_size,
                             tf_valid_images,    tf_valid_labels,     valid_dataset_size,
                             tf_test_images,     tf_test_labels,      tf_test_img_ids,      test_dataset_size,
                             tf_bump_test_images,tf_bump_test_labels, tf_bump_test_img_ids, bump_test_dataset_size,
                             n_epochs, test_interval, sub_folder,include_bump_test_data, use_cross_entropy, activation):
    '''
    Train a CNN with a set of non-collision images and labels
    Returns Train accuracy (col and non-collision) Train Loss (collision and non-collision)
    :param tf_images:
    :param tf_labels:
    :param dataset_size:
    :param tf_bump_images:
    :param tf_bump_labels:
    :param bump_dataset_size:
    :param tf_test_images:
    :param tf_test_labels:
    :param test_dataset_size:
    :param tf_bump_test_images:
    :param tf_bump_test_labels:
    :param test_bump_dataset_size:
    :param n_epochs:
    :param test_interval:
    :param filename_suffix:
    :return:
    '''
    global TrainLogger, ValidLogger

    train_results = {}
    min_noncol_loss = 10000
    noncol_exceed_min_count = 0

    if not os.path.exists(sub_folder):
        os.mkdir(sub_folder)

    noncol_global_step = tf.Variable(0, trainable=False)

    inc_noncol_gstep = inc_gstep(noncol_global_step)

    tf_logits = logits(tf_images, is_training=True)

    if use_cross_entropy:
        if activation=='softmax':
            tf_loss = calculate_loss_softmax(tf_logits, tf_labels)
        elif activation=='sigmoid':
            tf_loss = calculate_loss_sigmoid(tf_logits, tf_labels)
        else:
            raise NotImplementedError
    else:
        tf_loss = calculate_loss(tf_logits, tf_labels, weigh_by_frequency=True)

    tf_optimize, tf_mom_update_ops, tf_grads_and_vars = cnn_optimizer.optimize_model_naive(tf_loss, noncol_global_step, collision=False)

    tf_train_predictions = predictions_with_inputs(tf_images)
    tf_valid_predictions = predictions_with_inputs(tf_valid_images)
    tf_test_predictions = predictions_with_inputs(tf_test_images)

    if include_bump_test_data:
        tf_bump_test_predictions = predictions_with_inputs(tf_bump_test_images)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    tf.global_variables_initializer().run()

    test_accuracies_all_epochs = []
    test_soft_accuracies_all_epochs = []

    max_valid_accuracy = 0
    n_valid_accuracy_staturated = 0
    valid_saturation_threshold = 3

    for epoch in range(n_epochs):
        avg_loss = []
        avg_train_accuracy = []


        # ============================================================================
        # Training Phase
        # ============================================================================
        for step in range( int(config.FACTOR_OF_TRAINING_TO_USE * dataset_size) // config.BATCH_SIZE):

            l1, _,_, pred, train_labels = sess.run([tf_loss, tf_optimize, tf_mom_update_ops, tf_train_predictions, tf_labels])
            avg_loss.append(l1)
            avg_train_accuracy.append(
                models_utils.accuracy(pred, train_labels, use_argmin=False)
            )

            if step < 2:
                logger.debug('Predictions for Non-Collided data')
                for pred, lbl in zip(pred, train_labels):
                    logger.debug('\t%s;%s', pred, lbl)

            if step < 10:
                for pred_item, label_item in zip(pred,train_labels):
                    is_currect = np.argmax(pred_item) == np.argmax(label_item)
                    TrainLogger.info('%s:%s:%s',pred_item,label_item,is_currect)

        print_start_of_new_input_pipline_to_some_logger(TrainLogger,'================== END ==========================')

        logger.info('\tAverage Loss for Epoch %d: %.5f' % (epoch, np.mean(avg_loss)))
        logger.info('\t\t Training accuracy: %.3f' % np.mean(avg_train_accuracy))

        # =========================================
        # Validation Phase
        # =======================================

        avg_valid_accuracy = []
        for step in range(valid_dataset_size // config.BATCH_SIZE):
            v_pred, v_labels = sess.run(
                [tf_valid_predictions, tf_valid_labels])
            avg_valid_accuracy.append(
                models_utils.accuracy(v_pred, v_labels, use_argmin=False)
            )
            for pred_item, label_item in zip(v_pred, v_labels):
                is_currect = np.argmax(pred_item) == np.argmax(label_item)
                ValidLogger.info('%s:%s:%s', pred_item, label_item, is_currect)

        if np.mean(avg_valid_accuracy) > max_valid_accuracy:
            max_valid_accuracy = np.mean(avg_valid_accuracy)
        else:
            n_valid_accuracy_staturated += 1
            print('Increasing valid_saturated count to: %d', n_valid_accuracy_staturated)

        if n_valid_accuracy_staturated>=valid_saturation_threshold:
            print('Increasing global step. Validation Accuracy Saturated')
            sess.run(inc_noncol_gstep)
            n_valid_accuracy_staturated = 0

        print_start_of_new_input_pipline_to_some_logger(ValidLogger,
                                                        'Valid Accuracy For Above: %.3f' % np.mean(avg_valid_accuracy))

        # ============================================================================
        # Testing Phase
        # ============================================================================
        if (epoch + 1) % test_interval == 0:


            test_results = test_the_model(sess, tf_test_img_ids, tf_test_images,tf_test_labels,
                                          tf_test_predictions, test_dataset_size,
                                          tf_bump_test_img_ids, tf_bump_test_images, tf_bump_test_labels,
                                          tf_bump_test_predictions, bump_test_dataset_size, epoch, sub_folder, include_bump_data=include_bump_test_data)

            test_accuracies_all_epochs.append(test_results['noncol-accuracy-hard'])
            test_soft_accuracies_all_epochs.append(test_results['noncol-accuracy-soft'])

    # ============================================================================
    # Persisting data
    # ============================================================================
    logger.info('Saving CNN Model')
    cnn_model_visualizer.save_cnn_hyperparameters(sub_folder, kernel_size_dict, stride_dict,
                                                  'hyperparams_%d.pickle' %(epoch))
    cnn_model_visualizer.save_cnn_weights_naive(sub_folder, sess, 'cnn_model_%d.ckpt' %(epoch))

    coord.request_stop()
    coord.join(threads)

    train_results['noncol-loss'] = np.mean(avg_loss)
    train_results['noncol-accuracy'] = np.mean(avg_train_accuracy)

    # average out the accuracies:
    test_results['noncol-accuracy-hard'] = np.mean(test_accuracies_all_epochs[-5:])
    test_results['noncol-accuracy-soft'] = np.mean(test_soft_accuracies_all_epochs[-5:])

    return train_results, test_results


def train_with_one_non_collision_collision_combination(sess,tf_images,tf_labels,dataset_size,
                                                       tf_test_images,tf_test_labels, tf_test_img_ids, test_dataset_size,
                                                       tf_bump_test_images, tf_bump_test_labels, tf_bump_test_img_ids, test_bump_dataset_size,
                                                       n_epochs, test_interval, sub_folder):
    '''
    Train a CNN with a set of collision and non-collision images and labels
    Returns Train accuracy (col and non-collision) Train Loss (collision and non-collision)
    :param tf_images:
    :param tf_labels:
    :param dataset_size:
    :param tf_bump_images:
    :param tf_bump_labels:
    :param bump_dataset_size:
    :param tf_test_images:
    :param tf_test_labels:
    :param test_dataset_size:
    :param tf_bump_test_images:
    :param tf_bump_test_labels:
    :param test_bump_dataset_size:
    :param n_epochs:
    :param test_interval:
    :param filename_suffix:
    :return:
    '''

    train_results = {}
    min_col_loss, min_noncol_loss = 10000, 10000
    col_exceed_min_count, noncol_exceed_min_count = 0, 0

    if not os.path.exists(sub_folder):
        os.mkdir(sub_folder)

    noncol_global_step = tf.Variable(0, trainable=False)
    col_global_step = tf.Variable(0, trainable=False)

    inc_noncol_gstep = inc_gstep(noncol_global_step)
    inc_col_gstep = inc_gstep(col_global_step)

    tf_logits = logits(tf_images, is_training=True)
    tf_loss = calculate_loss(tf_logits, tf_labels)

    tf_optimize, tf_mom_update_ops, tf_grads_and_vars = cnn_optimizer.optimize_model_naive(tf_loss, noncol_global_step, collision=False)

    tf_train_predictions = predictions_with_inputs(tf_images)

    tf_test_predictions = predictions_with_inputs(tf_test_images)
    tf_bump_test_predictions = predictions_with_inputs(tf_bump_test_images)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    tf.global_variables_initializer().run()

    test_accuracies_all_epochs = []
    bump_test_accuracies_all_epochs = []
    test_soft_accuracies_all_epochs = []
    bump_test_soft_accuracies_all_epochs = []
    for epoch in range(n_epochs):
        avg_loss = []
        avg_train_accuracy = []

        # ============================================================================
        # Training Phase
        # ============================================================================
        for step in range(int(config.FACTOR_OF_TRAINING_TO_USE * dataset_size) // config.BATCH_SIZE ):
            l1, _,_, pred, train_labels = sess.run([tf_loss, tf_optimize, tf_mom_update_ops, tf_train_predictions, tf_labels])
            avg_loss.append(l1)
            avg_train_accuracy.append(
                models_utils.accuracy(pred, train_labels, use_argmin=False))

            if step < 2:
                logger.debug('Predictions for data')
                for pred, lbl in zip(pred, train_labels):
                    logger.debug('\t%s;%s', pred, lbl)

        # ============================================================================
        # Calculations related to decaying learning rate
        # ============================================================================
        if min_noncol_loss > np.mean(avg_loss):
            min_noncol_loss = np.mean(avg_loss)
        else:
            noncol_exceed_min_count += 1
            logger.info('Increase noncol_exceed to %d', noncol_exceed_min_count)

        if noncol_exceed_min_count >= 3:
            logger.info('Stepping down collision learning rate')
            sess.run(inc_noncol_gstep)
            noncol_exceed_min_count = 0

        logger.info('\tAverage Loss for Epoch %d: %.5f' % (epoch, np.mean(avg_loss)))
        logger.info('\t\t Training accuracy: %.3f' % np.mean(avg_train_accuracy))

        # ============================================================================
        # Testing Phase
        # ============================================================================
        if (epoch + 1) % test_interval == 0:
            test_results = test_the_model(sess, tf_test_img_ids, tf_test_images,tf_test_labels,
                   tf_test_predictions, test_dataset_size,
                   tf_bump_test_img_ids, tf_bump_test_images, tf_bump_test_labels,
                   tf_bump_test_predictions, test_bump_dataset_size, epoch, sub_folder,include_bump_data=True)

            test_accuracies_all_epochs.append(test_results['noncol-accuracy-hard'])
            test_soft_accuracies_all_epochs.append(test_results['noncol-accuracy-soft'])
            bump_test_accuracies_all_epochs.append(test_results['col-accuracy-hard'])
            bump_test_soft_accuracies_all_epochs.append(test_results['col-accuracy-soft'])

    # ============================================================================
    # Persisting data
    # ============================================================================
    logger.info('Saving CNN Model')
    cnn_model_visualizer.save_cnn_hyperparameters(sub_folder, kernel_size_dict, stride_dict,
                                                  'hyperparams_%d.pickle' %(epoch))
    cnn_model_visualizer.save_cnn_weights_naive(sub_folder, sess, 'cnn_model_%d.ckpt' %(epoch))

    coord.request_stop()
    coord.join(threads)

    train_results['noncol-loss'] = np.mean(avg_loss)
    train_results['noncol-accuracy'] = np.mean(avg_train_accuracy)

    # average out the accuracies:
    test_results['noncol-accuracy-hard'] = np.mean(test_accuracies_all_epochs[-5:])
    test_results['noncol-accuracy-soft'] = np.mean(test_soft_accuracies_all_epochs[-5:])
    test_results['col-accuracy-hard'] = np.mean(bump_test_accuracies_all_epochs[-5:])
    test_results['col-accuracy-soft'] = np.mean(bump_test_soft_accuracies_all_epochs[-5:])

    return train_results, test_results

def print_start_of_new_input_pipline_to_all_loggers(input_description):
    global TestPredictionLogger, TestBumpPredictionLogger, SummaryLogger

    TrainLogger.info('=' * 80)
    TrainLogger.info(input_description)
    TrainLogger.info('=' * 80)

    ValidLogger.info('=' * 80)
    ValidLogger.info(input_description)
    ValidLogger.info('=' * 80)

    TestPredictionLogger.info('='*80)
    TestPredictionLogger.info(input_description)
    TestPredictionLogger.info('=' * 80)

    TestBumpPredictionLogger.info('='*80)
    TestBumpPredictionLogger.info(input_description)
    TestBumpPredictionLogger.info('=' * 80)

    SummaryLogger.info('=' * 80)
    SummaryLogger.info(input_description)
    SummaryLogger.info('=' * 80)

def print_start_of_new_input_pipline_to_some_logger(logger, input_description):

    logger.info('=' * 80)
    logger.info(input_description)
    logger.info('=' * 80)

# =================================================================
# Loggers
# =================================================================

sess = None

def train_using_different_fractions_of_training_data(configp, n_epochs, main_dir, test_interval, include_bump_optimize):
    '''
    Train a CNN with different factions of data. Data is written in the following.
    Non-collision data: Written in the chronological order to tfrecords with 250 samples in each
    Collision data: Written in the chronological order to tfrecords with 50 samples in each
    :param configp:
    :param n_epochs:
    :param main_dir:
    :param test_interval:
    :return:
    '''
    global sess,output_activation,min_thresh,max_thresh

    if include_bump_optimize:
        dataset_filenames, dataset_sizes = dataset_name_factory.old_get_col_noncol_train_and_col_noncol_test_data()
    else:
        dataset_filenames, dataset_sizes = dataset_name_factory.new_get_noncol_train_data_col_noncol_test_data()

    for di in range(len(dataset_filenames['train_dataset'])-1,len(dataset_filenames['train_dataset'])):
        tf.reset_default_graph()
        sess = tf.InteractiveSession(config=configp)
        with sess.as_default():

            cnn_variable_initializer.set_from_main(sess)
            cnn_variable_initializer.build_tensorflw_variables_naive(separate_collision_momentum=True)
            models_utils.set_from_main(sess, logger)

            used_data_percentage = int((di+1)*100.0/len(dataset_filenames['train_dataset']))
            dataset_filenames_chunk = dataset_filenames['train_dataset'][:di+1]
            bump_dataset_filenames_chunk = dataset_filenames['train_bump_dataset'][:di + 1]
            dsize = sum(dataset_sizes['train_dataset'][:di+1])
            bump_dsize = sum(dataset_sizes['train_bump_dataset'][:di + 1])
            testdsize = sum(dataset_sizes['test_dataset'])
            bumptestdsize = sum(dataset_sizes['test_bump_dataset'])

            if include_bump_optimize:
                bump_batch_size = config.BATCH_SIZE//4
                nonbump_batch_size = config.BATCH_SIZE - bump_batch_size
                tf_img_ids, tf_images, tf_labels = models_utils.build_input_pipeline(
                    dataset_filenames_chunk, nonbump_batch_size, shuffle=True,
                    training_data=False, use_opposite_label=False, inputs_for_sdae=False,rand_valid_direction_for_bump=False)

                tf_bump_img_ids, tf_bump_images, tf_bump_labels = models_utils.build_input_pipeline(
                    bump_dataset_filenames_chunk, bump_batch_size, shuffle=True,
                    training_data=False, use_opposite_label=True, inputs_for_sdae=False,rand_valid_direction_for_bump=False)

            else:
                tf_img_ids, tf_images, tf_labels = models_utils.build_input_pipeline(
                    dataset_filenames_chunk, config.BATCH_SIZE, shuffle=True,
                    training_data=True, use_opposite_label=False, inputs_for_sdae=False,rand_valid_direction_for_bump=False)
                tf_valid_img_ids, tf_valid_images, tf_valid_labels = models_utils.build_input_pipeline(
                    dataset_filenames['valid_dataset'], config.BATCH_SIZE, shuffle=True,
                    training_data=True, use_opposite_label=False, inputs_for_sdae=False,rand_valid_direction_for_bump=False)
                validdsize = sum(dataset_sizes['valid_dataset'])

            tf_test_img_ids, tf_test_images, tf_test_labels = models_utils.build_input_pipeline(dataset_filenames['test_dataset'],
                                                                                                config.BATCH_SIZE,
                                                                                                shuffle=False,
                                                                                                training_data=False,
                                                                                                use_opposite_label=False,
                                                                                                inputs_for_sdae=False,rand_valid_direction_for_bump=False)

            tf_bump_test_img_ids, tf_bump_test_images, tf_bump_test_labels = models_utils.build_input_pipeline(
                dataset_filenames['test_bump_dataset'], config.BATCH_SIZE, shuffle=False,
                training_data=False, use_opposite_label=True, inputs_for_sdae=False,rand_valid_direction_for_bump=False)

            if include_bump_optimize:
                output_activation = 'tanh'
                min_thresh = 0.2
                max_thresh = 0.0

                tf_both_images = tf.concat([tf_images,tf_bump_images],axis=0)
                tf_both_labels = tf.concat([tf_labels,tf_bump_labels],axis=0)

                train_results, test_results = train_with_one_non_collision_collision_combination(
                    sess, tf_both_images,tf_both_labels,dsize+bump_dsize,
                    tf_test_images,tf_test_labels,tf_test_img_ids, testdsize+bump_dsize,
                    tf_bump_test_images,tf_bump_test_labels,tf_bump_test_img_ids,bumptestdsize,
                    n_epochs, test_interval,main_dir + os.sep + 'trained_with_%d_data'%used_data_percentage
                )

                test_accuracy = test_results['noncol-accuracy-hard']
                soft_test_accuracy = test_results['noncol-accuracy-soft']
                bump_test_accuracy = test_results['col-accuracy-hard']
                bump_soft_accuracy = test_results['col-accuracy-soft']
                avg_loss = train_results['noncol-loss']
                test_noncol_precision = test_results['noncol-precision']
                test_noncol_recall = test_results['noncol-recall']
                test_col_precision = test_results['col-precision']
                test_col_recall = test_results['col-recall']

                noncol_precision_string = ''.join(['%.3f;' % test_noncol_precision[pi] for pi in range(3)])
                noncol_recall_string = ''.join(['%.3f;' % test_noncol_recall[ri] for ri in range(3)])
                col_precision_string = ''.join(['%.3f;' % test_col_precision[pi] for pi in range(3)])
                col_recall_string = ''.join(['%.3f;' % test_col_recall[ri] for ri in range(3)])

                print_start_of_new_input_pipline_to_all_loggers('trained_with_%d_data' % used_data_percentage)

                SummaryLogger.info('%d;%.3f;%.3f;%.3f;%.3f;%.5f;%.5f;%s;%s;%s;%s', n_epochs, np.mean(test_accuracy),
                                     np.mean(soft_test_accuracy),
                                     np.mean(bump_test_accuracy), np.mean(bump_soft_accuracy), np.mean(avg_loss),
                                     -1.0,
                                     noncol_precision_string, noncol_recall_string, col_precision_string,
                                     col_recall_string)

            else:
                output_activation = 'softmax'
                if output_activation == 'softmax':
                    max_thresh = 0.4
                    min_thresh = 0.2

                elif output_activation == 'sigmoid':
                    max_thresh = 0.6
                    min_thresh = 0.4
                else:
                    raise NotImplementedError


                train_results, test_results = train_with_non_collision(
                    sess, tf_images, tf_labels, dsize,
                    tf_valid_images,tf_valid_labels, validdsize,
                    tf_test_images, tf_test_labels, tf_test_img_ids, testdsize,
                    tf_bump_test_images, tf_bump_test_labels, tf_bump_test_img_ids, bumptestdsize,
                    n_epochs, test_interval, main_dir + os.sep + 'trained_with_%d_data' % used_data_percentage,
                    include_bump_test_data=True, use_cross_entropy=True, activation=output_activation
                )

                test_accuracy = test_results['noncol-accuracy-hard']
                soft_test_accuracy = test_results['noncol-accuracy-soft']
                bump_test_accuracy = test_results['col-accuracy-hard']
                bump_soft_accuracy = test_results['col-accuracy-soft']
                avg_loss = train_results['noncol-loss']

                test_noncol_precision = test_results['noncol-precision']
                test_noncol_recall = test_results['noncol-recall']
                test_col_precision = test_results['col-precision']
                test_col_recall = test_results['col-recall']

                noncol_precision_string = ''.join(['%.3f;' % test_noncol_precision[pi] for pi in range(3)])
                noncol_recall_string = ''.join(['%.3f;' % test_noncol_recall[ri] for ri in range(3)])
                col_precision_string = ''.join(['%.3f;' % test_col_precision[pi] for pi in range(3)])
                col_recall_string = ''.join(['%.3f;' % test_col_recall[ri] for ri in range(3)])

                print_start_of_new_input_pipline_to_all_loggers('trained_with_%d_data' % used_data_percentage)

                SummaryLogger.info('%d;%.3f;%.3f;%.3f;%.3f;%.5f;%.5f;%s;%s;%s;%s', n_epochs, np.mean(test_accuracy),
                                     np.mean(soft_test_accuracy),
                                     np.mean(bump_test_accuracy), np.mean(bump_soft_accuracy), np.mean(avg_loss),
                                     -1,
                                     noncol_precision_string, noncol_recall_string, col_precision_string,
                                     col_recall_string)

            sess.close()


def train_using_all_data():

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

    with sess.as_default() and graph.as_default():
        cnn_variable_initializer.set_from_main(sess, graph)
        cnn_variable_initializer.build_tensorflw_variables_naive()
        models_utils.set_from_main(sess, graph, logger)

        all_train_files, all_bump_files = [], []
        all_test_files, all_bump_test_files = [], []

        all_train_files = dataset_filenames['train_dataset']
        all_bump_files = dataset_filenames['train_bump_dataset']
        all_test_files = dataset_filenames['test_dataset']
        all_bump_test_files = dataset_filenames['test_bump_dataset']

        tf_img_ids, tf_images, tf_labels = models_utils.build_input_pipeline(
            all_train_files, config.BATCH_SIZE, shuffle=True,
            training_data=True, use_opposite_label=False, inputs_for_sdae=False)

        tf_bump_img_ids, tf_bump_images, tf_bump_labels = models_utils.build_input_pipeline(
            all_bump_files, config.BATCH_SIZE, shuffle=True,
            training_data=True, use_opposite_label=True, inputs_for_sdae=False)

        tf_test_img_ids, tf_test_images, tf_test_labels = models_utils.build_input_pipeline(all_test_files,
                                                                                            config.BATCH_SIZE,
                                                                                            shuffle=False,
                                                                                            training_data=False,
                                                                                            use_opposite_label=False,
                                                                                            inputs_for_sdae=False)
        tf_bump_test_img_ids, tf_bump_test_images, tf_bump_test_labels = models_utils.build_input_pipeline(
            all_bump_test_files, config.BATCH_SIZE, shuffle=False,
            training_data=False, use_opposite_label=True, inputs_for_sdae=False)

        print('\t\tAverage test accuracy: %.5f ' % np.mean(test_accuracy))
        print('\t\tAverage test accuracy(soft): %.5f' % np.mean(soft_test_accuracy))
        print('\t\tAverage test precision: %s', test_noncol_precision)
        print('\t\tAverage test recall: %s', test_noncol_recall)

        print('\t\tAverage bump test accuracy: %.5f ' % np.mean(bump_test_accuracy))
        print('\t\tAverage bump test (soft) accuracy: %.5f ' % np.mean(bump_soft_accuracy))
        print('\t\tAverage test bump precision: %s', test_col_precision)
        print('\t\tAverage test bump recall: %s', test_col_recall)

        noncol_precision_string = ''.join(['%.3f;' % test_noncol_precision[pi] for pi in range(3)])
        noncol_recall_string = ''.join(['%.3f;' % test_noncol_recall[ri] for ri in range(3)])
        col_precision_string = ''.join(['%.3f;' % test_col_precision[pi] for pi in range(3)])
        col_recall_string = ''.join(['%.3f;' % test_col_recall[ri] for ri in range(3)])

        SummaryLogger.info('%d;%.3f;%.3f;%.3f;%.3f;%.5f;%.5f;%s;%s;%s;%s', epoch, np.mean(test_accuracy),
                             np.mean(soft_test_accuracy),
                             np.mean(bump_test_accuracy), np.mean(bump_soft_accuracy), np.mean(avg_loss),
                             np.mean(avg_bump_loss),
                             noncol_precision_string, noncol_recall_string, col_precision_string, col_recall_string)


TestPredictionLogger, TestBumpPredictionLogger, TrainLogger, ValidLogger, SummaryLogger = None, None, None, None, None
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

    logger_dict = setup_loggers('main')
    TestPredictionLogger = logger_dict['test-noncol-logger']
    TestBumpPredictionLogger = logger_dict['test-col-logger']
    TrainLogger = logger_dict['train-logger']
    ValidLogger = logger_dict['valid-logger']
    SummaryLogger = logger_dict['summary-logger']

    configp = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configp.gpu_options.per_process_gpu_memory_fraction = 0.9

    test_interval = 5
    train_using_different_fractions_of_training_data(configp, 50, IMG_DIR,test_interval,include_bump_optimize=False)