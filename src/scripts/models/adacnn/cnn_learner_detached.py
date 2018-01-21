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

logger = None

graph = None
#configp = tf.ConfigProto(allow_soft_placement=True)
sess = None

activation = None

max_thresh = 0.34
min_thresh = 0.1

kernel_size_dict = None

stride_dict = None
scope_list = None
conv_dropout_placeholder_dict = {}

all_directions = ['left', 'straight', 'right']

def define_conv_dropout_placeholder():
    global scope_list, conv_dropout_placeholder_dict

    for op in scope_list:
        if 'conv' in op:
            conv_dropout_placeholder_dict[op] = tf.placeholder(dtype=tf.float32,shape=[kernel_size_dict[op][-1]],name='dropout_'+op)



def logits_detached(tf_inputs,is_training):
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
                        if is_training and config.USE_DROPOUT:
                            tf_inputs = tf.nn.dropout(tf_inputs,1.0 - config.IN_DROPOUT,name='input_dropout')
                        h = models_utils.activate(tf.nn.conv2d(tf_inputs,weight,strides=stride_dict[scope],padding='SAME')+bias,activation,name='hidden')

                        if config.USE_DROPOUT:
                            h = h * tf.reshape(conv_dropout_placeholder_dict[scope],[1,1,1,-1])

                    else:
                        h = models_utils.activate(
                            tf.nn.conv2d(h, weight, strides=stride_dict[scope], padding='SAME') + bias, activation,
                            name='hidden')
                        if config.USE_DROPOUT:
                            h = h * tf.reshape(conv_dropout_placeholder_dict[scope], [1, 1, 1, -1])

                elif 'pool' in scope:
                    logger.info('\t\tMax pooling for %s', scope)
                    h = tf.nn.max_pool(h,config.TF_ANG_VAR_SHAPES_DETACHED[scope],config.TF_ANG_STRIDES[scope],padding='SAME',name='pool_hidden')
                else:
                    # Reshaping required for the first fulcon layer
                    if scope == 'out':
                        logger.info('\t\tFully-connected with output Logits for %s',scope)

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
                            h = tf.reshape(h, [config.BATCH_SIZE, h_shape[1] * h_shape[2] * h_shape[3]])

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
    tf_logits = logits_detached(tf_inputs,False)
    return tf.nn.softmax(tf_logits),tf_labels


def calculate_loss(tf_logits,tf_labels):
    global scope_list,all_directions

    loss = tf.reduce_mean(tf.reduce_sum((tf.nn.softmax(tf_logits) - tf_labels) ** 2, axis=[1]), axis=[0])

    for op in scope_list:
        if 'conv' in op:
            with tf.variable_scope(op,reuse=True):
                loss += config.L2_BETA * tf.reduce_sum(tf.get_variable(config.TF_WEIGHTS_STR)**2)
        elif 'fc' in op or 'out' in op:
            with tf.variable_scope(op,reuse=True):
                for di in all_directions:
                    with tf.variable_scope(di,reuse=True):
                        loss += config.L2_BETA * tf.reduce_sum(tf.get_variable(config.TF_WEIGHTS_STR)**2)

    return loss


def inc_gstep(gstep):
    return tf.assign(gstep,gstep+1)

def define_input_pipeline(dataset_filenames):

    tf_img_ids, tf_images, tf_labels = models_utils.build_input_pipeline(
        dataset_filenames['train_dataset'], config.BATCH_SIZE, shuffle=True,
        training_data=True, use_opposite_label=False, inputs_for_sdae=False)

    return tf_img_ids, tf_images, tf_labels

def get_dropout_placeholder_dict():
    global scope_list,conv_dropout_placeholder_dict

    placeholder_feed_dict = {}
    for scope in scope_list:
        if 'conv' in scope:
            binom_vec = np.random.binomial(1, 1.0 - config.LAYER_DROPOUT,
                                   kernel_size_dict[scope][-1])/(1.0 - config.LAYER_DROPOUT)
            placeholder_feed_dict[conv_dropout_placeholder_dict[scope]] = binom_vec

    return placeholder_feed_dict


def override_hyperparameters(hyperparams):

    for k,v in hyperparams.items():
        if 'conv' in k:
            for kk,vv in v.items():
                if 'weights' == kk:
                    config.TF_ANG_VAR_SHAPES_DETACHED[k]=vv
                    print('Changing weight_shape of %s to %s',k,vv)
                elif 'stride' == kk:
                    config.TF_ANG_STRIDES[k]=vv
                    print('Changing stride of %s to %s', k, vv)
                else:
                    raise NotImplementedError
        if 'l2_beta' in k:
            config.L2_BETA = v
            print('Changing l2_beta to %s', v)
        if 'learning_rate' in k:
            config.START_LR = v
            print('Changing learning rate to %s', v)




def define_tf_ops(tf_train_images, tf_train_labels, tf_test_images, tf_test_labels,
                  tf_valid_image=None, tf_valid_labels=None):
    global tf_logits, tf_loss, tf_optimize
    global tf_train_predictions, tf_test_predictions
    global noncol_global_step, inc_noncol_gstep

    define_conv_dropout_placeholder()

    noncol_global_step = tf.Variable(0, trainable=False)
    inc_noncol_gstep = inc_gstep(noncol_global_step)

    tf_logits = logits_detached(tf_train_images, True)

    tf_train_predictions, tf_train_actuals = predictions_and_labels_with_inputs(tf_train_images, tf_train_labels)

    tf_loss = calculate_loss(tf_logits, tf_train_labels)

    tf_optimize, _ = cnn_optimizer.optimize_model_detached(tf_loss, noncol_global_step)

    tf_test_predictions, tf_test_actuals = predictions_and_labels_with_inputs(tf_test_images, tf_test_labels)


def train_and_test(train_epochs, test_interval, separate_validation_set=False):
    global tf_logits
    for epoch in range(train_epochs):
        avg_loss = []
        avg_train_accuracy,avg_test_accuracy = [],[]

        # Training Phase
        for step in range(dataset_sizes['train_dataset'] // config.BATCH_SIZE):
            l1, _ = sess.run([tf_loss, tf_optimize], feed_dict=get_dropout_placeholder_dict())
            avg_loss.append(l1)

        logger.info('\tAverage Loss for Epoch %d: %.5f' % (epoch, np.mean(avg_loss)))



        # Train Accuracy Calculation
        for step in range(dataset_sizes['train_dataset'] // config.BATCH_SIZE):
            train_predictions, train_actuals = sess.run([tf_train_predictions, tf_train_labels])
            avg_train_accuracy.append(
                models_utils.soft_accuracy(train_predictions, train_actuals, use_argmin=False, max_thresh=max_thresh,
                                           min_thresh=min_thresh))

            if step < 2:
                logger.debug('(Train) Predictions for Non-Collided data')
                for pred, lbl in zip(train_predictions, train_actuals):
                    logger.debug('\t%s;%s', pred, lbl)

        logger.info('\t\t Training accuracy: %.3f' % np.mean(avg_train_accuracy))

        # Testing and Validation Phase
        if (epoch+1)%test_interval==0:

            # Validation Phase
            if separate_validation_set:
                raise NotImplementedError

            for step in range(dataset_sizes['test_dataset'] // config.BATCH_SIZE):
                test_predictions, test_actuals = sess.run([tf_test_predictions, tf_test_labels])
                avg_test_accuracy.append(
                    models_utils.soft_accuracy(test_predictions, test_actuals, use_argmin=False,
                                               max_thresh=max_thresh,
                                               min_thresh=min_thresh))

                if step < 2:
                    logger.debug('(Test) Predictions for Non-Collided data')
                    for pred, lbl in zip(test_predictions, test_predictions):
                        logger.debug('\t%s;%s', pred, lbl)

            logger.info('\t\t Test accuracy: %.3f' % np.mean(avg_test_accuracy))



tf_logits, tf_loss, tf_optimize = None, None, None
tf_train_predictions, tf_test_predictions = None, None
noncol_global_step, inc_noncol_gstep = None, None

def setup_loggers():
    global logger,testPredictionLogger

    logger = logging.getLogger('Logger')
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    fileHandler = logging.FileHandler(IMG_DIR + os.sep + 'main.log', mode='w')
    fileHandler.setFormatter(logging.Formatter(logging_format))
    fileHandler.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(fileHandler)

    testPredictionLogger = logging.getLogger('TestPredictionLogger')
    testPredictionLogger.setLevel(logging.DEBUG)
    testPredFH = logging.FileHandler(IMG_DIR + os.sep + 'test_predictions.log', mode='w')
    testPredFH.setFormatter(logging.Formatter(logging_format))
    testPredFH.setLevel(logging.DEBUG)
    testPredictionLogger.addHandler(testPredFH)

logger,testPredictionLogger = None, None

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


    setup_loggers()

    config.setup_user_dependent_hyperparameters(no_pooling=True,square_input=False)
    logger.info('='*80)
    logger.info('Scope list: %s',scope_list)
    logger.info('Input of size: %s',config.TF_INPUT_SIZE)
    logger.info('Input of size (After Resized): %s', config.TF_INPUT_AFTER_RESIZE)
    logger.info('=' * 80 + '\n')

    activation = config.ACTIVATION
    kernel_size_dict = config.TF_ANG_VAR_SHAPES_DETACHED
    stride_dict = config.TF_ANG_STRIDES
    scope_list = config.TF_ANG_SCOPES

    print('dataset_filenames')
    dataset_filenames,dataset_sizes = dataset_name_factory.new_get_noncol_train_data_sorted_by_direction_noncol_test_data()


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
    accuracy_logger.info('#Epoch; Non-collision Accuracy;Non-collision Accuracy(Soft);Collision Accuracy;' +
                         'Collision Accuracy (Soft); Non-collision loss; Collision loss;' +
                         'Preci-NC-L,Preci-NC-S,Preci-NC-R;;Rec-NC-L,Rec-NC-S,Rec-NC-R;;' +
                         'Preci-C-L,Preci-C-S,Preci-C-R;;Rec-C-L,Rec-C-S,Rec-C-R;')

    config.BATCH_SIZE = 10

    graph = tf.Graph()
    configp = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.InteractiveSession(graph=graph,config=configp)

    with sess.as_default() and graph.as_default():
        cnn_variable_initializer.set_from_main(sess)
        cnn_variable_initializer.build_tensorflw_variables_detached()
        models_utils.set_from_main(sess,logger)

        tf_train_img_ids, tf_train_images, tf_train_labels = define_input_pipeline(dataset_filenames)
        tf_test_img_ids, tf_test_images,tf_test_labels = models_utils.build_input_pipeline(
            dataset_filenames['test_dataset'],config.BATCH_SIZE,shuffle=True,
            training_data=False,use_opposite_label=False,inputs_for_sdae=False)

        define_tf_ops(tf_train_images, tf_train_labels, tf_test_images, tf_test_labels)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf.global_variables_initializer().run(session=sess)

        for epoch in range(250):
            avg_loss = []

            # Training Phase

            for step in range(dataset_sizes['train_dataset']//config.BATCH_SIZE):
                l1, _ = sess.run([tf_loss, tf_optimize],feed_dict=get_dropout_placeholder_dict())
                avg_loss.append(l1)

            logger.info('\tAverage Loss for Epoch %d: %.5f' %(epoch,np.mean(avg_loss)))


            avg_train_accuracy = []

            # Prediction Phase
            for step in range(dataset_sizes['train_dataset'] // config.BATCH_SIZE):
                train_predictions,train_actuals = sess.run([tf_train_predictions,tf_train_labels])
                avg_train_accuracy.append(models_utils.soft_accuracy(train_predictions, train_actuals, use_argmin=False,max_thresh=max_thresh,min_thresh=min_thresh))

                if step < 2:
                    logger.debug('Predictions for Non-Collided data')
                    for pred, lbl in zip(train_predictions, train_actuals):
                        logger.debug('\t%s;%s', pred, lbl)

            logger.info('\t\t Training accuracy: %.3f' % np.mean(avg_train_accuracy))

            # Test Phase
            '''if (epoch+1)%5==0:
                test_accuracy = []
                soft_test_accuracy = []
                all_predictions, all_labels, all_img_ids, all_images = None, None, None, None
                test_image_index = 0
                for step in range(dataset_sizes['test_dataset']//config.BATCH_SIZE):
                    predicted_labels,actual_labels,test_img_ids, test_images = sess.run([tf_test_predictions,tf_test_actuals,tf_test_img_ids, tf_test_images])

                    test_accuracy.append(models_utils.accuracy(predicted_labels,actual_labels,use_argmin=False))
                    soft_test_accuracy.append(models_utils.soft_accuracy(predicted_labels,actual_labels,use_argmin=False,max_thresh=max_thresh,min_thresh=min_thresh))

                    if all_predictions is None or all_labels is None:
                        all_predictions = predicted_labels
                        all_labels = actual_labels
                        all_img_ids = test_img_ids
                        all_images = test_images

                    else:
                        all_predictions = np.append(all_predictions,predicted_labels,axis=0)
                        all_labels = np.append(all_labels,actual_labels,axis=0)
                        all_img_ids = np.append(all_img_ids, test_img_ids, axis=0)
                        all_images = np.append(all_images, test_images, axis = 0)

                    if step<2:
                        logger.debug('Test Predictions (Non-Collisions)')
                        for pred,act in zip(predicted_labels,actual_labels):
                            pred_string = ''.join(['%.3f'%p+',' for p in pred.tolist()])
                            act_string = ''.join(['%.1f'%a +',' for a in act.tolist()])
                            predictionlogger.info('%d:%s:%s',test_image_index,act_string,pred_string)
                            if step < 2:
                                logger.debug('%d:%s:%s',test_image_index,act_string,pred_string)
                            test_image_index += 1

                testPredictionLogger.info('Predictions for Non-Collisions (Epoch %d)', epoch)
                test_image_index = 0
                for pred, act in zip(all_predictions, all_labels):
                    pred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
                    act_string = ''.join(['%.3f' % a + ',' for a in act.tolist()])
                    testPredictionLogger.info('%d:%s:%s', test_image_index, act_string, pred_string)
                    test_image_index += 1
                testPredictionLogger.info('\n')

                test_noncol_precision = models_utils.precision_multiclass(all_predictions, all_labels, use_argmin=False,max_thresh=max_thresh,min_thresh=min_thresh)
                test_noncol_recall = models_utils.recall_multiclass(all_predictions,all_labels, use_argmin=False,max_thresh=max_thresh,min_thresh=min_thresh)
                predictionlogger.info('\n')
                print('\t\tAverage test accuracy: %.5f '%np.mean(test_accuracy))
                print('\t\tAverage test accuracy(soft): %.5f'%np.mean(soft_test_accuracy))
                print('\t\tAverage test precision: %s', test_noncol_precision)
                print('\t\tAverage test recall: %s', test_noncol_recall)

                # saving figures

                # create a dictionary img_id => image
                image_list = np.split(all_images,all_images.shape[0])
                id_list = all_img_ids.tolist()
                dict_id_image = dict(zip(id_list,image_list))

                correct_hard_ids_sorted = {}
                predicted_hard_ids_sorted = {}
                for di,direct in enumerate(['left','straight','right']):
                    correct_hard_ids_sorted[direct] = models_utils.get_id_vector_for_correctly_predicted_samples(
                        all_img_ids, all_predictions, all_labels, di, enable_soft_accuracy=False, use_argmin=False)
                    predicted_hard_ids_sorted[direct] = models_utils.get_id_vector_for_predicted_samples(
                        all_img_ids, all_predictions, all_labels, di, enable_soft_accuracy=False, use_argmin=False)
                logger.info('correct hard img ids for: %s',correct_hard_ids_sorted)
                visualizer.save_fig_with_predictions_for_direction(correct_hard_ids_sorted, dict_id_image,
                                                                   IMG_DIR + os.sep + 'correct_predicted_results_hard_%d.png'%(epoch))
                visualizer.save_fig_with_predictions_for_direction(predicted_hard_ids_sorted, dict_id_image,
                                                                   IMG_DIR + os.sep + 'predicted_results_hard_%d.png' % (
                                                                   epoch))

                    #correct_soft_img_ids = models_utils.get_accuracy_vector_with_direction(
                    #    all_img_ids, all_predictions, all_labels, di, enable_soft_accuracy=True, use_argmin=False)

                testPredictionLogger.info('Predictions for Collisions (Epoch %d)',epoch)
                test_image_index = 0
                for pred, act in zip(all_bump_predictions, all_bump_labels):
                    bpred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
                    bact_string = ''.join(['%.3f' % a + ',' for a in act.tolist()])
                    testPredictionLogger.info('%d:%s:%s', test_image_index, bact_string, bpred_string)
                    test_image_index += 1
                testPredictionLogger.info('\n')


                precision_string = ''.join(['%.3f,'%test_noncol_precision[pi] for pi in range(3)])
                recall_string = ''.join(['%.3f,'%test_noncol_recall[ri] for ri in range(3)])

                col_precision_string = ''.join(['%.3f,' % test_col_precision[pi] for pi in range(3)])
                col_recall_string = ''.join(['%.3f,' % test_col_recall[ri] for ri in range(3)])

                accuracy_logger.info('%d;%.3f;%.3f;%.3f;%.3f;%.5f;%.5f;%s;%s;%s;%s',
                                     epoch,np.mean(test_accuracy),np.mean(soft_test_accuracy),
                                     np.mean(bump_test_accuracy),np.mean(bump_soft_accuracy),
                                     np.mean(avg_loss), np.mean(avg_bump_loss),
                                     precision_string,recall_string,col_precision_string,col_recall_string)'''
        coord.request_stop()
        coord.join(threads)
