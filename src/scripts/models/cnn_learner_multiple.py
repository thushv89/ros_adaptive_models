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

max_thresh = 0.6
min_thresh = 0.4

batch_size = 10

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
                            for di in (['left','straight','right']):

                                with tf.variable_scope(di,reuse=True):
                                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                        config.TF_BIAS_STR)
                                    logger.info('\t\t\tConvolution %s (%s)', di, weight.get_shape().as_list())
                                    h_per_di.append(models_utils.activate(tf.nn.conv2d(tf_inputs,weight,strides=config.TF_ANG_STRIDES[scope],padding='SAME')+bias,activation,name='hidden'))
                            h = tf.concat(values=h_per_di,axis=3)
                            logger.info('\t\tConcat Shape (%s)', h.get_shape().as_list())
                        else:
                            h_per_di = []
                            for di in (['left','straight','right']):
                                with tf.variable_scope(di,reuse=True):
                                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                        config.TF_BIAS_STR)
                                    logger.info('\t\t\tConvolution %s (%s)', di, weight.get_shape().as_list())
                                    h_per_di.append(models_utils.activate(tf.nn.conv2d(h, weight, strides=config.TF_ANG_STRIDES[scope], padding='SAME') + bias, activation,
                                           name='hidden'))
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
                                assert config.TF_ANG_VAR_SHAPES_MULTIPLE[scope][0] ==  config.TF_ANG_VAR_SHAPES_MULTIPLE['fc1'][1]*3
                                h_per_di = []
                                for di in (['left','straight','right']):
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
                                for di in (['left','straight','right']):
                                    with tf.variable_scope(di):
                                        weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                            config.TF_BIAS_STR)
                                        h_shape = h.get_shape().as_list()
                                        logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s', scope, h_shape)
                                        h_di = tf.reshape(h, [batch_size, h_shape[1] * h_shape[2] * h_shape[3]])
                                        h_per_di.append(models_utils.activate(tf.matmul(h_di, weight) + bias, activation))

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

def test_the_model(tf_test_img_ids, tf_test_images, tf_test_labels,
                   tf_test_bump_img_ids, tf_test_bump_images, tf_test_bump_labels):
    raise NotImplementedError

def train_cnn_multiple_epochs(n_epochs):

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

    for direction in ['left', 'straight', 'right']:
        tf_img_ids[direction], tf_images[direction], tf_labels[direction] = models_utils.build_input_pipeline(
            dataset_filenames['train_dataset'][direction], batch_size, shuffle=True,
            training_data=False, use_opposite_label=False, inputs_for_sdae=False, rand_valid_direction_for_bump=False)

        tf_logits[direction] = logits(tf_images[direction], direction)
        temp = ['left', 'straight', 'right']
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

    all_valid_filenames = []
    for di in ['left', 'straight', 'right']:
        all_valid_filenames += dataset_filenames['valid_dataset'][di]
    tf_valid_img_ids, tf_valid_images, tf_valid_labels = models_utils.build_input_pipeline(
        all_valid_filenames, batch_size, shuffle=True,
        training_data=False, use_opposite_label=False, inputs_for_sdae=False, rand_valid_direction_for_bump=False)
    tf_valid_predictions = predictions_with_inputs(tf_valid_images)

    tf_test_img_ids, tf_test_images, tf_test_labels = models_utils.build_input_pipeline(all_test_files, batch_size,
                                                                                        shuffle=False,
                                                                                        training_data=False,
                                                                                        use_opposite_label=False,
                                                                                        inputs_for_sdae=False,
                                                                                        rand_valid_direction_for_bump=False)
    tf_bump_test_img_ids, tf_bump_test_images, tf_bump_test_labels = models_utils.build_input_pipeline(
        all_bump_test_files, batch_size, shuffle=False,
        training_data=False, use_opposite_label=True, inputs_for_sdae=False, rand_valid_direction_for_bump=False)

    tf_test_predictions = predictions_with_inputs(tf_test_images)
    tf_bump_test_predictions = predictions_with_inputs(tf_bump_test_images)

    max_valid_accuracy = 0
    n_valid_saturated = 0
    valid_saturate_threshold = 3

    for epoch in range(100):
        avg_loss = []
        avg_train_accuracy = []

        # Training with Non-Bump Data
        for step in range(dataset_sizes['train_dataset'] // batch_size):

            rand_direction = np.random.choice(['left', 'straight', 'right'])
            temp = ['left', 'straight', 'right']
            temp.remove(rand_direction)
            if rand_direction == 'left':
                new_rand_direction = np.random.choice(temp, p=[0.6, 0.4])
            elif rand_direction == 'right':
                new_rand_direction = np.random.choice(temp, p=[0.4, 0.6])
            else:
                new_rand_direction = np.random.choice(temp)

            l1_noncol, _, pred, train_labels = \
                sess.run([tf_loss[rand_direction], tf_optimize[rand_direction],
                          tf_train_predictions[rand_direction], tf_labels[rand_direction]],
                         feed_dict={tf_mock_labels: np.ones(shape=(batch_size, 1), dtype=np.float32)})

            l1_col, _ = sess.run(
                [tf_bump_loss[rand_direction][new_rand_direction], tf_bump_optimize[rand_direction][new_rand_direction],
                 ],
                feed_dict={tf_mock_labels: np.zeros(shape=(batch_size, 1), dtype=np.float32)})

            avg_loss.append((l1_col + l1_noncol) / 2.0)
            avg_train_accuracy.append(models_utils.accuracy(pred, train_labels, use_argmin=False))

            if step < 2:
                logger.debug('Predictions for Non-Collided data')
                for pred, lbl in zip(pred, train_labels):
                    logger.debug('\t%s;%s', pred, lbl)

        logger.info('\tAverage Loss for Epoch %d: %.5f' % (epoch, np.mean(avg_loss)))
        logger.info('\t\t Training accuracy: %.3f' % np.mean(avg_train_accuracy))

        valid_accuracy = []
        for step in range(dataset_sizes['valid_dataset'] // batch_size):
            vpred, vlabels = sess.run([tf_valid_predictions, tf_valid_labels])
            valid_accuracy.append(models_utils.accuracy(vpred, vlabels, use_argmin=False))

        logger.info('\tValid Accuracy: %.3f', np.mean(valid_accuracy))

        if np.mean(valid_accuracy) > max_valid_accuracy:
            max_valid_accuracy = np.mean(valid_accuracy)
        else:
            n_valid_saturated += 1
            logger.info('Increase n_valid_saturated to %d', noncol_exceed_min_count)

        if n_valid_saturated >= valid_saturate_threshold:
            logger.info('Stepping down collision learning rate')
            sess.run(inc_noncol_gstep)
            n_valid_saturated = 0

        if (epoch + 1) % 5 == 0:
            test_accuracy = []
            soft_test_accuracy = []
            all_predictions, all_labels = None, None
            test_image_index = 0
            for step in range(dataset_sizes['test_dataset'] // batch_size):
                predicted_labels, actual_labels = sess.run([tf_test_predictions, tf_test_labels])

                test_accuracy.append(models_utils.accuracy(predicted_labels, actual_labels, use_argmin=False))
                soft_test_accuracy.append(
                    models_utils.soft_accuracy(predicted_labels, actual_labels, use_argmin=False, max_thresh=max_thresh,
                                               min_thresh=min_thresh))

                # logger.debug('\t\t\tPrecision list %s', precision_list)
                # logger.debug('\t\t\tRecall list %s', recall_list)
                if all_predictions is None or all_labels is None:
                    all_predictions = predicted_labels
                    all_labels = actual_labels
                else:
                    all_predictions = np.append(all_predictions, predicted_labels, axis=0)
                    all_labels = np.append(all_labels, actual_labels, axis=0)

                if step < 5:
                    logger.debug('Test Predictions (Non-Collisions)')
                for pred, act in zip(predicted_labels, actual_labels):
                    pred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
                    act_string = ''.join([str(int(a)) + ',' for a in act.tolist()])
                    predictionlogger.info('%d:%s:%s', test_image_index, act_string, pred_string)
                    if step < 5:
                        logger.debug('%d:%s:%s', test_image_index, act_string, pred_string)
                    test_image_index += 1

            test_noncol_precision = models_utils.precision_multiclass(all_predictions, all_labels, use_argmin=False,
                                                                      max_thresh=max_thresh, min_thresh=min_thresh)
            test_noncol_recall = models_utils.recall_multiclass(all_predictions, all_labels, use_argmin=False,
                                                                max_thresh=max_thresh, min_thresh=min_thresh)
            predictionlogger.info('\n')
            print('\t\tAverage test accuracy: %.5f ' % np.mean(test_accuracy))
            print('\t\tAverage test accuracy(soft): %.5f' % np.mean(soft_test_accuracy))
            print('\t\tAverage test precision: %s', test_noncol_precision)
            print('\t\tAverage test recall: %s', test_noncol_recall)

            bump_test_accuracy = []
            bump_soft_accuracy = []

            all_bump_predictions, all_bump_labels = None, None
            for step in range(dataset_sizes['test_bump_dataset'] // batch_size):
                bump_predicted_labels, bump_actual_labels = sess.run([tf_bump_test_predictions, tf_bump_test_labels])
                bump_test_accuracy.append(
                    models_utils.accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))
                bump_soft_accuracy.append(
                    models_utils.soft_accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True,
                                               max_thresh=max_thresh, min_thresh=min_thresh))

                if all_bump_predictions is None or all_bump_labels is None:
                    all_bump_predictions = bump_predicted_labels
                    all_bump_labels = bump_actual_labels
                else:
                    all_bump_predictions = np.append(all_bump_predictions, bump_predicted_labels, axis=0)
                    all_bump_labels = np.append(all_bump_labels, bump_actual_labels, axis=0)

            if step < 5:
                logger.debug('Test Predictions (Collisions)')
            test_image_index = 0
            for pred, act in zip(all_bump_predictions, all_bump_labels):
                bpred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
                bact_string = ''.join([str(int(a)) + ',' for a in act.tolist()])
                bumpPredictionlogger.info('%d:%s:%s', test_image_index, bact_string, bpred_string)
                if step < 5:
                    logger.debug('%d:%s:%s', test_image_index, bact_string, bpred_string)
                test_image_index += 1

            bumpPredictionlogger.info('\n')

            print('\t\tAverage bump test accuracy: %.5f ' % np.mean(bump_test_accuracy))
            print('\t\tAverage bump test (soft) accuracy: %.5f ' % np.mean(bump_soft_accuracy))

            precision_string = ''.join(['%.3f,' % test_noncol_precision[pi] for pi in range(3)])
            recall_string = ''.join(['%.3f,' % test_noncol_recall[ri] for ri in range(3)])

            accuracy_logger.info('%d;%.3f;%.3f;%.3f;%.3f;%s;%s', epoch, np.mean(test_accuracy),
                                 np.mean(soft_test_accuracy),
                                 np.mean(bump_test_accuracy), np.mean(bump_soft_accuracy), precision_string,
                                 recall_string)


def setup_loggers():
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

    bumpPredictionlogger = logging.getLogger('BumpPredictionLogger')
    bumpPredictionlogger.setLevel(logging.INFO)
    bumpfh = logging.FileHandler(IMG_DIR + os.sep + 'test-bump-predictions.log', mode='w')
    bumpfh.setFormatter(logging.Formatter('%(message)s'))
    bumpfh.setLevel(logging.INFO)
    bumpPredictionlogger.addHandler(bumpfh)
    bumpPredictionlogger.info('#ID:Actual:Predicted:Correct?')

    accuracy_logger = logging.getLogger('AccuracyLogger')
    accuracy_logger.setLevel(logging.INFO)
    accuracyFH = logging.FileHandler(IMG_DIR + os.sep + 'accuracy.log', mode='w')
    accuracyFH.setFormatter(logging.Formatter('%(message)s'))
    accuracyFH.setLevel(logging.INFO)
    accuracy_logger.addHandler(accuracyFH)
    accuracy_logger.info('#Epoch;Accuracy;Non-collision Accuracy(Soft);Collision Accuracy;' +
                         'Collision Accuracy (Soft);Preci-NC-L,Preci-NC-S,Preci-NC-R;Rec-NC-L,Rec-NC-S,Rec-NC-R')

    loggers_dict['test-logger'] = TestPredictionLogger
    loggers_dict['bump-test-logger'] = bumpPredictionlogger
    loggers_dict['summary-logger'] = accuracy_logger
    loggers_dict['train-logger'] = TrainLogger
    loggers_dict['valid-logger'] = ValidLogger

    return loggers_dict


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
    dataset_filenames, dataset_sizes = dataset_name_factory.new_get_noncol_train_data_sorted_by_direction_col_noncol_test_data()

    min_col_loss, min_noncol_loss = 10000, 10000
    col_exceed_min_count, noncol_exceed_min_count = 0, 0

    configp = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.InteractiveSession(config=configp)

    with sess.as_default():
        cnn_variable_initializer.set_from_main(sess)
        cnn_variable_initializer.build_tensorflw_variables_multiple()
        models_utils.set_from_main(sess,logger)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf.global_variables_initializer().run(session=sess)

        coord.request_stop()
        coord.join(threads)
