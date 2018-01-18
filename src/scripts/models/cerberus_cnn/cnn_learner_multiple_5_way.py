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
import dataset_name_factory
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

activation = 'lrelu'
out_activation = 'sigmoid'

max_thresh = 0.5
min_thresh = 0.4

batch_size = 10

kernel_size_dict = config.TF_ANG_VAR_SHAPES_NAIVE
stride_dict = config.TF_ANG_STRIDES
scope_list = config.TF_ANG_SCOPES


def set_scope_kernel_stride_dicts(sc_list, k_dict, s_dict):
    global kernel_size_dict,stride_dict,scope_list
    scope_list = sc_list
    kernel_size_dict = k_dict
    stride_dict = s_dict


def logits(tf_inputs,direction=None):
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
                        logger.info('\t\tConvolution with ReLU activation for %s',scope)
                        if si == 0:
                            h_per_di = []
                            used_labels = []
                            for di in config.TF_DIRECTION_LABELS:
                                if di in used_labels:
                                    continue

                                with tf.variable_scope(di,reuse=True):
                                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                        config.TF_BIAS_STR)
                                    logger.info('\t\t\tConvolution %s (%s)', di, weight.get_shape().as_list())
                                    if not config.USE_DILATION:
                                        h_per_di.append(
                                            models_utils.activate(
                                                tf.nn.conv2d(
                                                    tf_inputs,weight,strides=config.TF_ANG_STRIDES[scope],padding='SAME')+bias,
                                                activation,name='hidden')
                                        )
                                    else:

                                        h_per_di.append(
                                            models_utils.activate(
                                                tf.nn.convolution(tf_inputs,weight,
                                                                  dilation_rate=config.TF_DILATION[scope],padding='SAME') + bias,
                                            activation, name='dilated-hidden')
                                        )
                                used_labels.append(di)
                            h = tf.concat(values=h_per_di,axis=3)
                            logger.info('\t\tConcat Shape (%s)', h.get_shape().as_list())
                        else:
                            h_per_di = []
                            used_labels = []
                            for di in config.TF_DIRECTION_LABELS:
                                if di in used_labels:
                                    continue
                                with tf.variable_scope(di,reuse=True):
                                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                        config.TF_BIAS_STR)
                                    logger.info('\t\t\tConvolution %s (%s)', di, weight.get_shape().as_list())

                                    if not config.USE_DILATION:
                                        h_per_di.append(
                                            models_utils.activate(
                                                tf.nn.conv2d(h, weight, strides=config.TF_ANG_STRIDES[scope], padding='SAME') + bias,
                                                activation, name='hidden')
                                        )
                                    else:
                                        h_per_di.append(
                                            models_utils.activate(
                                                tf.nn.convolution(h,weight, padding='SAME',
                                                                  dilation_rate=config.TF_DILATION[scope]) + bias,
                                                activation, name='dilated-hidden'
                                            )
                                        )
                                used_labels.append(di)
                            h = tf.concat(values=h_per_di,axis=3)
                            logger.info('\t\tConcat Shape (%s)', h.get_shape().as_list())
                    elif 'pool' in scope:
                        logger.info('\t\tMax pooling for %s', scope)
                        h = tf.nn.max_pool(h,config.TF_ANG_VAR_SHAPES_MULTIPLE[scope],config.TF_ANG_STRIDES[scope],padding='SAME',name='pool_hidden')

                    else:
                        # Reshaping required for the first fulcon layer
                        if scope == 'out':
                            if direction is None:
                                logger.info('\t\tFully-connected with output Logits for %s',scope)
                                assert config.TF_ANG_VAR_SHAPES_MULTIPLE[scope][0] ==  config.TF_ANG_VAR_SHAPES_MULTIPLE['fc1'][1]*5
                                h_per_di = []

                                for di in config.TF_DIRECTION_LABELS:
                                    with tf.variable_scope(di):
                                        weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                            config.TF_BIAS_STR)
                                        h_per_di.append(tf.matmul(h, weight) + bias)

                                h = tf.concat(h_per_di,axis=1)
                            else:
                                with tf.variable_scope(direction):
                                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                        config.TF_BIAS_STR)
                                    h = tf.matmul(h, weight) + bias

                        elif 'fc' in scope:
                            if scope == config.TF_FIRST_FC_ID:
                                h_per_di = []
                                used_labels = []
                                for di in config.TF_DIRECTION_LABELS:
                                    if di in used_labels:
                                        continue
                                    with tf.variable_scope(di):
                                        weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                            config.TF_BIAS_STR)
                                        h_shape = h.get_shape().as_list()
                                        logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s', scope, h_shape)
                                        h_di = tf.reshape(h, [batch_size, h_shape[1] * h_shape[2] * h_shape[3]])
                                        h_per_di.append(models_utils.activate(tf.matmul(h_di, weight) + bias, activation))
                                    used_labels.append(di)
                                h = tf.concat(h_per_di,axis=1)
                            else:
                                raise NotImplementedError
                        else:
                            raise NotImplementedError

    return h


def predictions_with_inputs(tf_inputs):

    tf_logits = logits(tf_inputs)
    pred = models_utils.activate(tf_logits, activation_type=out_activation)
    return pred


def calculate_loss(tf_logits, tf_labels):
    use_cross_entropy = True
    if not use_cross_entropy:
        loss = tf.reduce_mean(
            tf.reduce_sum(((models_utils.activate(tf_logits,activation_type=out_activation) - tf_labels)**2),axis=[1]),axis=[0])
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_labels,logits=tf_logits))
    #loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf_logits,labels=tf_labels),axis=[1]))
    return loss


def inc_gstep(gstep):
    return tf.assign(gstep,gstep+1)



def test_the_model_5_way(sess,tf_test_labels,
                   tf_test_predictions, dataset_size_dict):
    test_accuracy = []
    soft_test_accuracy = []
    all_predictions, all_labels = None, None
    test_image_index = 0
    for step in range(dataset_size_dict['test_dataset'] // batch_size):
        predicted_labels, actual_labels = sess.run([tf_test_predictions, tf_test_labels])

        test_accuracy.append(models_utils.accuracy(predicted_labels, actual_labels, use_argmin=False))
        soft_test_accuracy.append(
            models_utils.soft_accuracy(predicted_labels, actual_labels, use_argmin=False, max_thresh=max_thresh,
                                       min_thresh=min_thresh))

        if all_predictions is None or all_labels is None:
            all_predictions = predicted_labels
            all_labels = actual_labels
        else:
            all_predictions = np.append(all_predictions, predicted_labels, axis=0)
            all_labels = np.append(all_labels, actual_labels, axis=0)

        if step < 10:
            for pred, act in zip(predicted_labels, actual_labels):
                pred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
                act_string = ''.join([str(int(a)) + ',' for a in act.tolist()])
                is_correct = np.argmax(pred) == np.argmax(act)
                TestLogger.info('%s:%s:%s', act_string, pred_string,is_correct)
                if step < 5:
                    logger.debug('%s:%s:%s', act_string, pred_string,is_correct)

    print_start_of_new_input_pipline_to_some_logger(
        TestLogger,
        'Accuracy for Above: %.3f (Hard) %.3f (Soft)' % (np.mean(test_accuracy), np.mean(soft_test_accuracy))
    )

    test_noncol_precision = models_utils.precision_multiclass(all_predictions, all_labels, use_argmin=False,
                                                              max_thresh=max_thresh, min_thresh=min_thresh)
    test_noncol_recall = models_utils.recall_multiclass(all_predictions, all_labels, use_argmin=False,
                                                        max_thresh=max_thresh, min_thresh=min_thresh)
    TestLogger.info('\n')
    print('\t\tAverage test accuracy: %.5f ' % np.mean(test_accuracy))
    print('\t\tAverage test accuracy(soft): %.5f' % np.mean(soft_test_accuracy))
    print('\t\tAverage test precision: %s', test_noncol_precision)
    print('\t\tAverage test recall: %s', test_noncol_recall)

    test_results = {}
    test_results['noncol-accuracy-hard'] = np.mean(test_accuracy)
    test_results['noncol-accuracy-soft'] = np.mean(soft_test_accuracy)
    test_results['noncol-precision'] = test_noncol_precision
    test_results['noncol-recall'] = test_noncol_recall

    return test_results


def train_cnn_multiple_epochs(sess, n_epochs, test_interval, dataset_filenames_dict, dataset_size_dict,
                              train_fraction=1.0, valid_fraction = 1.0):

    n_print_prediction_steps = 10

    noncol_global_step = tf.Variable(0, trainable=False)
    inc_noncol_gstep = inc_gstep(noncol_global_step)

    tf_img_ids, tf_images, tf_labels = {}, {}, {}
    tf_loss, tf_logits = {}, {}
    tf_bump_loss, tf_bump_logits = {}, {}
    tf_optimize, tf_mom_update_ops, tf_grads = {}, {}, {}
    tf_bump_optimize, tf_bump_mom_update_ops, tf_bump_grads = {}, {}, {}
    tf_mock_labels = tf.placeholder(shape=[batch_size, 1], dtype=tf.float32)
    tf_grads_and_vars = {}
    tf_train_predictions = {}

    for direction in config.TF_DIRECTION_LABELS:

        tf_img_ids[direction], tf_images[direction], tf_labels[direction] = models_utils.build_input_pipeline(
            dataset_filenames_dict['train_dataset'][direction], batch_size, shuffle=True,
            training_data=False, use_opposite_label=False, inputs_for_sdae=False, rand_valid_direction_for_bump=False)

        tf_logits[direction] = logits(tf_images[direction], direction)
        temp = list(config.TF_DIRECTION_LABELS)
        temp.remove(direction)

        tf_bump_logits[direction], tf_bump_loss[direction] = {}, {}
        tf_bump_optimize[direction], tf_bump_mom_update_ops[direction] = {}, {}

        # =================================================
        # Defining Optimization for Opposite Direction
        # =================================================
        for opp_direction in temp:
            bump_var_list = []
            for v in tf.global_variables():

                if opp_direction in v.name and config.TF_MOMENTUM_STR not in v.name:
                    print(v.name)
                    bump_var_list.append(v)

            tf_bump_logits[direction][opp_direction] = logits(tf_images[direction], opp_direction)
            tf_bump_loss[direction][opp_direction] = calculate_loss(tf_bump_logits[direction][opp_direction],
                                                                    tf_mock_labels)
            tf_bump_optimize[direction][opp_direction], _ = cnn_optimizer.optimize_model_naive_no_momentum(
                tf_bump_loss[direction][opp_direction], noncol_global_step, varlist=bump_var_list
            )

        tf_train_predictions[direction] = predictions_with_inputs(
            tf_images[direction])

        tf_loss[direction] = calculate_loss(tf_logits[direction], tf_mock_labels)

        var_list = []
        for v in tf.global_variables():


            if direction in v.name and config.TF_MOMENTUM_STR not in v.name:
                print(v.name)
                var_list.append(v)

        tf_optimize[direction], tf_grads_and_vars[direction] = cnn_optimizer.optimize_model_naive_no_momentum(
            tf_loss[direction], noncol_global_step, varlist=var_list
        )

    tf_valid_img_ids, tf_valid_images, tf_valid_labels = models_utils.build_input_pipeline(
        dataset_filenames_dict['valid_dataset'], batch_size, shuffle=True,
        training_data=False, use_opposite_label=False, inputs_for_sdae=False, rand_valid_direction_for_bump=False)
    tf_valid_predictions = predictions_with_inputs(tf_valid_images)

    tf_test_img_ids, tf_test_images, tf_test_labels = \
        models_utils.build_input_pipeline(dataset_filenames_dict['test_dataset'], batch_size,
                                          shuffle=False, training_data=False, use_opposite_label=False,
                                          inputs_for_sdae=False, rand_valid_direction_for_bump=False)

    tf_test_predictions = predictions_with_inputs(tf_test_images)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    tf.global_variables_initializer().run(session=sess)

    max_valid_accuracy = 0
    n_valid_saturated = 0
    valid_saturate_threshold = 3

    for epoch in range(n_epochs):

        print('='*80)
        print('Epoch ',epoch)
        print('=' * 80)

        avg_loss = []
        avg_train_accuracy = []

        # Training with Non-Bump Data
        for step in range(int(train_fraction*dataset_size_dict['train_dataset']) // batch_size):

            rand_direction = np.random.choice(config.TF_DIRECTION_LABELS)
            temp = list(config.TF_DIRECTION_LABELS)
            temp.remove(rand_direction)

            l1_noncol, _, pred, train_labels = \
                sess.run([tf_loss[rand_direction], tf_optimize[rand_direction],
                          tf_train_predictions[rand_direction], tf_labels[rand_direction]],
                         feed_dict={tf_mock_labels: np.ones(shape=(batch_size, 1), dtype=np.float32)})

            # Try doing negative suppresion for 2 times
            for _ in range(2):
                if 'hard-left' == rand_direction:
                    new_rand_direction = np.random.choice(temp, p=[0.3, 0.3, 0.2,0.2])
                elif 'soft-left' == rand_direction:
                    new_rand_direction = np.random.choice(temp, p=[0.25, 0.25, 0.25,0.25])
                elif 'soft-right' == rand_direction:
                    new_rand_direction = np.random.choice(temp, p=[0.25, 0.25, 0.25, 0.25])
                elif 'hard-right' in rand_direction:
                    new_rand_direction = np.random.choice(temp, p=[0.2, 0.2, 0.3, 0.3])
                else:
                    new_rand_direction = np.random.choice(temp)

                l1_col, _ = sess.run(
                    [tf_bump_loss[rand_direction][new_rand_direction], tf_bump_optimize[rand_direction][new_rand_direction],
                     ],
                    feed_dict={tf_mock_labels: np.zeros(shape=(batch_size, 1), dtype=np.float32)})

            avg_loss.append((l1_col + l1_noncol) / 2.0)
            avg_train_accuracy.append(models_utils.accuracy(pred, train_labels, use_argmin=False))

            if step < n_print_prediction_steps:
                for pred, lbl in zip(pred, train_labels):
                    is_correct = np.argmax(pred)==np.argmax(lbl)
                    TrainLogger.info('\t%s;%s;%s', pred, lbl, is_correct)

        logger.info('\tAverage Loss for Epoch %d: %.5f' % (epoch, np.mean(avg_loss)))
        logger.info('\t\t Training accuracy: %.3f' % np.mean(avg_train_accuracy))

        valid_accuracy = []
        for step in range(int(valid_fraction*dataset_size_dict['valid_dataset']) // batch_size):
            vpred, vlabels = sess.run([tf_valid_predictions, tf_valid_labels])
            valid_accuracy.append(models_utils.accuracy(vpred, vlabels, use_argmin=False))

            if step < n_print_prediction_steps:
                for pred, lbl in zip(vpred, vlabels):
                    is_correct = np.argmax(pred)==np.argmax(lbl)
                    ValidLogger.info('\t%s;%s;%s', pred, lbl, is_correct)

        logger.info('\tValid Accuracy: %.3f', np.mean(valid_accuracy))

        if np.mean(valid_accuracy) > max_valid_accuracy:
            max_valid_accuracy = np.mean(valid_accuracy)
        else:
            n_valid_saturated += 1
            logger.info('Increase n_valid_saturated to %d', n_valid_saturated)

        if n_valid_saturated >= valid_saturate_threshold:
            logger.info('Stepping down collision learning rate')
            sess.run(inc_noncol_gstep)
            n_valid_saturated = 0

        if (epoch + 1) % test_interval == 0:

            test_results = test_the_model_5_way(sess,
                tf_test_labels,
                tf_test_predictions, dataset_size_dict
            )

            soft_test_accuracy = test_results['noncol-accuracy-soft']
            test_accuracy = test_results['noncol-accuracy-hard']
            test_noncol_precision = test_results['noncol-precision']
            test_noncol_recall = test_results['noncol-recall']

            noncol_precision_string = ''.join(['%.3f,' % test_noncol_precision[pi] for pi in range(config.TF_NUM_CLASSES)])
            noncol_recall_string = ''.join(['%.3f,' % test_noncol_recall[ri] for ri in range(config.TF_NUM_CLASSES)])

            SummaryLogger.info('%d;%.3f;%.3f;%.5f;%.5f;%s;%s', epoch, np.mean(test_accuracy),
                               np.mean(soft_test_accuracy), np.mean(avg_loss),
                               -1,
                               noncol_precision_string, noncol_recall_string)

    coord.request_stop()
    coord.join(threads)


def loop_through_by_using_every_dataset_as_holdout_dataset(main_dir):

    hold_out_list = ['indoor-1-5-way-3000','indoor-1-my1-5-way-3000','indoor-1-my2-5-way-3000']

    for hold_index, hold_name in enumerate(hold_out_list):

        if hold_index==0:
            continue

        sub_dir = main_dir + os.sep + hold_name

        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        print_start_of_new_input_pipline_to_all_loggers('Using %s as holdout set'%hold_name)

        dataset_filenames, dataset_sizes = dataset_name_factory.new_get_train_test_data_with_holdout_5_way_half_dataset(hold_index)

        with open(sub_dir + os.sep + 'dataset_filenames_and_sizes.txt','w') as f:
            for k,v in dataset_filenames.items():
                f.write(str(k)+":"+str(v))
                f.write('\n')

            for k,v in dataset_sizes.items():
                f.write(str(k)+":"+str(v))
                f.write('\n')

        tf.reset_default_graph()
        configp = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.InteractiveSession(config=configp)

        with sess.as_default():
            cnn_variable_initializer.set_from_main(sess)
            cnn_variable_initializer.build_tensorflw_variables_multiple_5_way()
            models_utils.set_from_main(sess, logger)

            train_cnn_multiple_epochs(sess, 75, 5,dataset_filenames,dataset_sizes,train_fraction=1,valid_fraction=1)
            # ============================================================================
            # Persisting data
            # ============================================================================
            logger.info('Saving CNN Model')
            #cnn_model_visualizer.save_cnn_hyperparameters(sub_dir, kernel_size_dict, stride_dict,
            #                                              scope_list, 'hyperparams-final.pickle')
            #cnn_model_visualizer.save_cnn_weights_multiple(sub_dir, sess, 'cnn-model-final.ckpt')

            sess.close()


def print_start_of_new_input_pipline_to_all_loggers(input_description):
    '''
    Print some text to input pipeline
    :param input_description:
    :return:
    '''
    global TestLogger, TestBumpLogger, SummaryLogger

    TrainLogger.info('=' * 80)
    TrainLogger.info(input_description)
    TrainLogger.info('=' * 80)

    ValidLogger.info('=' * 80)
    ValidLogger.info(input_description)
    ValidLogger.info('=' * 80)

    TestLogger.info('='*80)
    TestLogger.info(input_description)
    TestLogger.info('=' * 80)

    TestBumpLogger.info('='*80)
    TestBumpLogger.info(input_description)
    TestBumpLogger.info('=' * 80)

    SummaryLogger.info('=' * 80)
    SummaryLogger.info(input_description)
    SummaryLogger.info('=' * 80)


def print_start_of_new_input_pipline_to_some_logger(logger, input_description):

    logger.info('=' * 80)
    logger.info(input_description)
    logger.info('=' * 80)


def setup_loggers():
    '''
    Setting up loggers
    :return:
    '''
    loggers_dict = {}

    TrainLogger = logging.getLogger('TrainLogger')
    TrainLogger.setLevel(logging.INFO)
    fh = logging.FileHandler(IMG_DIR + os.sep + 'train-predictions.log', mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    TrainLogger.addHandler(fh)
    TrainLogger.info('#ID:Actual:Predicted:Correct?')

    ValidLogger = logging.getLogger('ValidLogger')
    ValidLogger.setLevel(logging.INFO)
    fh = logging.FileHandler(IMG_DIR + os.sep + 'valid-predictions.log', mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    ValidLogger.addHandler(fh)
    ValidLogger.info('#ID:Actual:Predicted:Correct?')

    TestPredictionLogger = logging.getLogger('TestPredictionLogger')
    TestPredictionLogger.setLevel(logging.INFO)
    fh = logging.FileHandler(IMG_DIR + os.sep + 'test-predictions.log', mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    TestPredictionLogger.addHandler(fh)
    TestPredictionLogger.info('#ID:Actual:Predicted:Correct?')

    TestBumpLogger = logging.getLogger('BumpPredictionLogger')
    TestBumpLogger.setLevel(logging.INFO)
    bumpfh = logging.FileHandler(IMG_DIR + os.sep + 'test-bump-predictions.log', mode='w')
    bumpfh.setFormatter(logging.Formatter('%(message)s'))
    bumpfh.setLevel(logging.INFO)
    TestBumpLogger.addHandler(bumpfh)
    TestBumpLogger.info('#ID:Actual:Predicted:Correct?')

    SummaryLogger = logging.getLogger('AccuracyLogger')
    SummaryLogger.setLevel(logging.INFO)
    accuracyFH = logging.FileHandler(IMG_DIR + os.sep + 'accuracy.log', mode='w')
    accuracyFH.setFormatter(logging.Formatter('%(message)s'))
    accuracyFH.setLevel(logging.INFO)
    SummaryLogger.addHandler(accuracyFH)
    SummaryLogger.info('#Epoch;Non-collision Accuracy;Non-collision Accuracy(Soft);Collision Accuracy;' +
                         'Collision Accuracy (Soft);Loss (Non-collision); Loss (collision); Preci-NC-L;Preci-NC-S;Preci-NC-R;;' +
                         'Rec-NC-L;Rec-NC-S;Rec-NC-R;;Preci-C-L;Preci-C-S;Preci-C-R;;Rec-C-L;Rec-C-S;Rec-C-R;')


    loggers_dict['test-logger'] = TestPredictionLogger
    loggers_dict['bump-test-logger'] = TestBumpLogger
    loggers_dict['summary-logger'] = SummaryLogger
    loggers_dict['train-logger'] = TrainLogger
    loggers_dict['valid-logger'] = ValidLogger

    return loggers_dict

TrainLogger, ValidLogger, TestLogger, TestBumpLogger, SummaryLogger = None, None, None, None, None
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

    logger_dict = setup_loggers()
    TrainLogger = logger_dict['train-logger']
    ValidLogger = logger_dict['valid-logger']
    TestLogger = logger_dict['test-logger']
    TestBumpLogger = logger_dict['bump-test-logger']
    SummaryLogger = logger_dict['summary-logger']

    loop_through_by_using_every_dataset_as_holdout_dataset(IMG_DIR)
