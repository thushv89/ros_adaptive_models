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
import visualizer
import cnn_variable_initializer
import cnn_optimizer

'''

Dual Network Architecture

'''

logging_level = logging.DEBUG
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

logger = None

graph = None
#configp = tf.ConfigProto(allow_soft_placement=True)
sess = None

activation = config.ACTIVATION
out_activation = 'tanh'

max_thresh = 0.25
min_thresh = -0.001


def logits(tf_inputs):
    '''
    Logit inference for dual network
    :param tf_inputs:
    :param direction:
    :param collision:
    :return:
    '''
    global logger
    logger.info('Defining inference ops ...')
    with tf.name_scope('infer'):
        for si, scope in enumerate(config.TF_ANG_SCOPES):

            if 'conv' in scope:

                if scope=='conv1':
                    with tf.variable_scope(config.TF_NONCOL_STR, reuse=True):
                        with tf.variable_scope(scope, reuse=True) as sc:
                            nc_weights, nc_bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                            logger.info('\t\tConvolution with ReLU activation for %s', scope)
                            logger.info('\t\t\tWeights: %s', nc_weights.name)
                            h_nc = models_utils.activate(
                                tf.nn.conv2d(tf_inputs, nc_weights, strides=config.TF_ANG_STRIDES[scope],
                                             padding='SAME') + nc_bias, activation, name='conv_hidden_nc')
                            logger.info('\t\t\tOutput size %s', h_nc.get_shape().as_list())

                    with tf.variable_scope(config.TF_COL_STR, reuse=True):
                        with tf.variable_scope(scope, reuse=True) as sc:
                            c_weights, c_bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                config.TF_BIAS_STR)
                            logger.info('\t\tConvolution with ReLU activation for %s',scope)
                            logger.info('\t\t\tWeights: %s', c_weights.name)
                            h_c = models_utils.activate(tf.nn.conv2d(tf_inputs,c_weights,strides=config.TF_ANG_STRIDES[scope],padding='SAME')+c_bias,activation,name='conv_hidden_c')
                            logger.info('\t\t\tOutput size %s', h_c.get_shape().as_list())

                    h = tf.concat([h_nc,h_c],axis=3)
                else:
                    with tf.variable_scope(config.TF_NONCOL_STR, reuse=True):
                        with tf.variable_scope(scope, reuse=True) as sc:
                            nc_weights, nc_bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                config.TF_BIAS_STR)
                            logger.info('\t\tConvolution with ReLU activation for %s', scope)
                            logger.info('\t\t\tWeights: %s', nc_weights.name)
                            h_nc = models_utils.activate(
                                tf.nn.conv2d(h, nc_weights, strides=config.TF_ANG_STRIDES[scope],
                                             padding='SAME') + nc_bias, activation, name='conv_hidden_nc')
                            logger.info('\t\t\tOutput size %s', h_nc.get_shape().as_list())

                    with tf.variable_scope(config.TF_COL_STR, reuse=True):
                        with tf.variable_scope(scope, reuse=True) as sc:
                            c_weights, c_bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                config.TF_BIAS_STR)
                            logger.info('\t\tConvolution with ReLU activation for %s', scope)
                            logger.info('\t\t\tWeights: %s', c_weights.name)
                            h_c = models_utils.activate(
                                tf.nn.conv2d(h, c_weights, strides=config.TF_ANG_STRIDES[scope],
                                             padding='SAME') + c_bias, activation, name='conv_hidden_c')
                            logger.info('\t\t\tOutput size %s', h_c.get_shape().as_list())

                    h = tf.concat([h_nc, h_c], axis=3)

            elif 'pool' in scope:
                logger.info('\t\tMax pooling for %s', scope)
                h = tf.nn.max_pool(h,config.TF_VAR_SHAPES_DUAL_NAIVE_NONCOL[scope],config.TF_ANG_STRIDES[scope],padding='SAME',name='pool_hidden')

            elif 'fc' in scope:
                if scope == config.TF_FIRST_FC_ID:

                    h_shape = h.get_shape().as_list()
                    logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s', scope, h_shape)
                    h = tf.reshape(h, [batch_size, h_shape[1] * h_shape[2] * h_shape[3]])

                    with tf.variable_scope(scope, reuse=True) as sc:
                        weights, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                            config.TF_BIAS_STR)
                        logger.info('\t\t\tWeights: %s', weights.name)
                        h = models_utils.activate(tf.matmul(h, weights) + bias, activation)

                else:
                    raise NotImplementedError

            elif scope=='out':

                with tf.variable_scope(scope, reuse=True) as sc:
                    weights, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                        config.TF_BIAS_STR)

                    h = tf.matmul(h,weights)+bias

            else:
                raise NotImplementedError

    return h


def predictions_and_labels_with_inputs(tf_inputs, tf_labels):
    tf_logits = logits(tf_inputs)
    return models_utils.activate(tf_logits,activation_type=out_activation),tf_labels


def calculate_loss(tf_logits, tf_labels):
    mask_predictions = False
    random_masking = False
    tf_out = tf_logits
    tf_label_weights = tf.reduce_mean(tf_labels, axis=[0])

    if mask_predictions and not random_masking:
        masked_preds = models_utils.activate(tf_logits,activation_type=out_activation) * tf.cast(tf.not_equal(tf_labels, 0.0), dtype=tf.float32)
        loss = tf.reduce_mean(
            tf.reduce_sum((masked_preds - tf_labels) ** 2, axis=[1]), axis=[0])
    elif random_masking:
        rand_mask = tf.cast(tf.greater(tf.random_normal([config.BATCH_SIZE, 3], dtype=tf.float32), 0.0),
                            dtype=tf.float32)
        masked_preds = models_utils.activate(tf_logits,activation_type=out_activation) * (tf.cast(tf.not_equal(tf_labels, 0.0), dtype=tf.float32) + rand_mask)
        loss = tf.reduce_mean(
            tf.reduce_sum((masked_preds - tf_labels) ** 2, axis=[1]), axis=[0])
    else:
        # use appropriately to weigh output *(1-tf.abs(tf_label_weights))
        loss = tf.reduce_mean(
            tf.reduce_sum(((models_utils.activate(tf_logits,activation_type=out_activation) - tf_labels) ** 2), axis=[1]),
            axis=[0])

    return loss


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

    dataset_filenames = {'train_dataset': ['..' + os.sep + 'data_indoor_1_1000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(3)] +
                                          ['..' + os.sep + 'data_sandbox_1000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(3)] +
                                          ['..' + os.sep + 'data_grande_salle_1000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(3)],
                         'train_bump_dataset': ['..' + os.sep + 'data_indoor_1_bump_200' + os.sep +
                                                'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for
                                                i in
                                                range(3)] +
                                               ['..' + os.sep + 'data_sandbox_bump_200' + os.sep +
                                                'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for
                                                i in
                                                range(3)] +
                                               ['..' + os.sep + 'data_grande_salle_bump_200' + os.sep +
                                                'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for
                                                i in
                                                range(3)],
                         'test_dataset': ['..' + os.sep + 'data_indoor_1_1000' + os.sep +
                                          'data-chunks-chronological' + os.sep + 'data-chunk-3.tfrecords'] +
                                         ['..' + os.sep + 'data_sandbox_1000' + os.sep +
                                          'data-chunks-chronological' + os.sep + 'data-chunk-3.tfrecords'] +
                                         ['..' + os.sep + 'data_grande_salle_1000' + os.sep +
                                          'data-chunks-chronological' + os.sep + 'data-chunk-3.tfrecords'],
                         'test_bump_dataset': ['..' + os.sep + 'data_indoor_1_bump_200' + os.sep +
                                               'data-chunks-chronological' + os.sep + 'data-chunk-3.tfrecords'] +
                                              ['..' + os.sep + 'data_sandbox_bump_200' + os.sep +
                                               'data-chunks-chronological' + os.sep + 'data-chunk-3.tfrecords'] +
                                              ['..' + os.sep + 'data_grande_salle_bump_200' + os.sep +
                                               'data-chunks-chronological' + os.sep + 'data-chunk-3.tfrecords']
                         }

    dataset_sizes = {'train_dataset': sum([250 for i in range(9)]),
                     'train_bump_dataset': sum([50 for i in range(9)]),
                     'test_dataset': sum([250, 250, 250]),
                     'test_bump_dataset': sum([50, 50, 50])}

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
    accuracy_logger.info('#Epoch; Non-collision loss; Collision loss; Non-collision Accuracy;Non-collision Accuracy(Soft);Collision Accuracy;' +
                         'Collision Accuracy (Soft);Preci-NC-L,Preci-NC-S,Preci-NC-R;Rec-NC-L,Rec-NC-S,Rec-NC-R;' +
                         'Preci-C-L,Preci-C-S,Preci-C-R;Rec-C-L,Rec-C-S,Rec-C-R')

    batch_size = config.BATCH_SIZE


    configp = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    configp.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.InteractiveSession(config=configp)

    with sess.as_default():

        models_utils.set_from_main(sess, logger)
        cnn_variable_initializer.set_from_main(sess)

        cnn_variable_initializer.build_tensorflw_variables_dual_naive()

        noncol_global_step = tf.Variable(0,trainable=False)
        col_global_step = tf.Variable(0,trainable=False)

        inc_noncol_gstep = inc_gstep(noncol_global_step)
        inc_col_gstep = inc_gstep(col_global_step)

        # defining Tensorflow outputs

        all_train_data = dataset_filenames['train_dataset']
        all_bump_train_data = dataset_filenames['train_bump_dataset']

        print(all_train_data)
        tf_img_ids, tf_images,tf_labels = models_utils.build_input_pipeline(
            all_train_data,batch_size,shuffle=True,
            training_data=True,use_opposite_label=False,inputs_for_sdae=False)

        tf_bump_img_ids, tf_bump_images, tf_bump_labels = models_utils.build_input_pipeline(
            all_bump_train_data, batch_size, shuffle=True,
            training_data=True, use_opposite_label=True,inputs_for_sdae=False)

        tf_logits = logits(tf_images)
        tf_train_predictions,tf_train_actuals = predictions_and_labels_with_inputs(tf_images,tf_labels)

        tf_loss = calculate_loss(tf_logits, tf_labels)

        noncol_var_list, col_var_list = [],[]
        for v in tf.global_variables():
            if 'out' in v.name and config.TF_MOMENTUM_STR not in v.name:
                col_var_list.append(v)
                noncol_var_list.append(v)
            if 'fc' in v.name and config.TF_MOMENTUM_STR not in v.name:
                col_var_list.append(v)
                noncol_var_list.append(v)
            elif config.TF_NONCOL_STR in v.name and config.TF_MOMENTUM_STR not in v.name:
                noncol_var_list.append(v)
            elif config.TF_COL_STR in v.name and config.TF_MOMENTUM_STR not in v.name:
                col_var_list.append(v)

        tf_optimize, tf_mom_update_ops, \
        tf_noncol_lr, tf_grads_and_vars = cnn_optimizer.optimize_model_dual_naive_builtin(tf_loss,noncol_global_step, var_list=noncol_var_list)
        tf_bump_optimize, tf_bump_mom_update_ops, \
        tf_col_lr,_ = cnn_optimizer.optimize_model_dual_naive_builtin(tf_bump_loss,col_global_step,var_list=col_var_list)

        tf_test_img_ids, tf_test_images,tf_test_labels = models_utils.build_input_pipeline(dataset_filenames['test_dataset'],batch_size,shuffle=True,
                                                             training_data=True,use_opposite_label=False,inputs_for_sdae=False)
        tf_bump_test_img_ids, tf_bump_test_images, tf_bump_test_labels = models_utils.build_input_pipeline(dataset_filenames['test_bump_dataset'], batch_size, shuffle=True,
                                                              training_data=True,use_opposite_label=True,inputs_for_sdae=False)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf_test_predictions,tf_test_actuals = predictions_and_labels_with_inputs(tf_test_images,tf_test_labels)
        tf_bump_test_predictions,tf_bump_test_actuals = predictions_and_labels_with_inputs(tf_bump_test_images,tf_bump_test_labels)

        tf.global_variables_initializer().run(session=sess)

        rand_direction = None
        for epoch in range(100):
            avg_loss = []
            avg_bump_loss = []

            for step in range(dataset_sizes['train_dataset']//batch_size):

                if np.random.random()<0.7:
                    l1, _, _, gnv = sess.run([tf_loss, tf_optimize,tf_mom_update_ops,tf_grads_and_vars])
                    avg_loss.append(l1)

                else:
                    bump_l1, _, _ = sess.run([tf_bump_loss, tf_bump_optimize, tf_bump_mom_update_ops])
                    avg_bump_loss.append(bump_l1)

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
            logger.info('\t\t Learning rate (non-collision): %.5f',sess.run(tf_noncol_lr))

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
            logger.info('\t\t Learning rate (collision): %.5f', sess.run(tf_col_lr))


            avg_train_accuracy = []
            avg_bump_train_accuracy = []
            # Prediction Phase
            for step in range(dataset_sizes['train_dataset'] // batch_size):
                train_predictions,train_actuals = sess.run([tf_train_predictions,tf_train_actuals])
                avg_train_accuracy.append(models_utils.soft_accuracy(train_predictions, train_actuals, use_argmin=False, max_thresh=max_thresh, min_thresh=min_thresh))

                if step < 2:
                    logger.debug('Predictions for Non-Collided data')
                    for pred, lbl in zip(train_predictions, train_actuals):
                        logger.debug('\t%s;%s', pred, lbl)

                if step < (dataset_sizes['train_bump_dataset']//batch_size):
                    train_bump_predictions, train_bump_actuals = sess.run(
                        [tf_train_bump_predictions, tf_train_bump_actuals])
                    avg_bump_train_accuracy.append(
                        models_utils.soft_accuracy(train_bump_predictions, train_bump_actuals, use_argmin=False, max_thresh=max_thresh, min_thresh=min_thresh))

                    if step < 2:
                        logger.debug('Predictions for Collided data')
                        for pred, lbl in zip(train_bump_predictions, train_bump_actuals):
                            logger.debug('\t%s;%s', pred, lbl)

            logger.info('\t\t Training accuracy: %.3f' % np.mean(avg_train_accuracy))
            logger.info(
                '\t\tAverage Bump Accuracy (Train) for Epoch %d: %.5f' % (epoch, np.mean(avg_bump_train_accuracy)))

            # Test Phase
            if (epoch+1)%5==0:
                test_accuracy = []
                soft_test_accuracy = []
                all_predictions, all_labels, all_img_ids, all_images = None, None, None, None
                test_image_index = 0
                for step in range(dataset_sizes['test_dataset']//batch_size):
                    predicted_labels,actual_labels,test_img_ids, test_images = sess.run([tf_test_predictions,tf_test_actuals,tf_test_img_ids, tf_test_images])

                    test_accuracy.append(models_utils.accuracy(predicted_labels,actual_labels,use_argmin=False))
                    soft_test_accuracy.append(models_utils.soft_accuracy(predicted_labels,actual_labels,use_argmin=False, max_thresh=max_thresh, min_thresh=min_thresh))

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

                test_noncol_precision = models_utils.precision_multiclass(all_predictions, all_labels, use_argmin=False, max_thresh=max_thresh, min_thresh=min_thresh)
                test_noncol_recall = models_utils.recall_multiclass(all_predictions,all_labels, use_argmin=False, max_thresh=max_thresh, min_thresh=min_thresh)
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
                        all_img_ids, all_predictions, all_labels, di, enable_soft_accuracy=False, use_argmin=False
                    )
                logger.info('correct hard img ids for: %s',correct_hard_ids_sorted)
                visualizer.save_fig_with_predictions_for_direction(correct_hard_ids_sorted, dict_id_image,
                                                                   IMG_DIR + os.sep + 'correct_predicted_results_hard_%d.png'%(epoch))
                visualizer.save_fig_with_predictions_for_direction(predicted_hard_ids_sorted, dict_id_image,
                                                                   IMG_DIR + os.sep + 'predicted_results_hard_%d.png' % (
                                                                   epoch))

                    #correct_soft_img_ids = models_utils.get_accuracy_vector_with_direction(
                    #    all_img_ids, all_predictions, all_labels, di, enable_soft_accuracy=True, use_argmin=False)

                bump_test_accuracy = []
                bump_soft_accuracy  = []

                all_bump_img_ids, all_bump_images, all_bump_predictions, all_bump_labels = None,None,None,None
                test_image_index = 0
                for step in range(dataset_sizes['test_bump_dataset']//batch_size):
                    bump_img_ids, bump_predicted_labels, \
                    bump_actual_labels, bump_images = sess.run([tf_bump_test_img_ids, tf_bump_test_predictions,
                                                                tf_bump_test_actuals, tf_bump_test_images])
                    bump_test_accuracy.append(models_utils.accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))
                    bump_soft_accuracy.append(models_utils.soft_accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True, max_thresh=max_thresh, min_thresh=min_thresh))

                    if all_bump_predictions is None or all_bump_labels is None:
                        all_bump_predictions = bump_predicted_labels
                        all_bump_labels = bump_actual_labels
                        all_bump_images = bump_images
                        all_bump_img_ids = bump_img_ids
                    else:
                        all_bump_predictions = np.append(all_bump_predictions,bump_predicted_labels,axis=0)
                        all_bump_labels = np.append(all_bump_labels,bump_actual_labels,axis=0)
                        all_bump_images = np.append(all_bump_images,bump_images,axis=0)
                        all_bump_img_ids = np.append(all_bump_img_ids,bump_img_ids,axis=0)

                    if step < 2:
                        logger.debug('Test Predictions (Collisions)')

                        for pred, act in zip(all_bump_predictions,all_bump_labels):
                            bpred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
                            bact_string = ''.join(['%.3f' % a + ',' for a in act.tolist()])
                            bumpPredictionlogger.info('%d:%s:%s', test_image_index, bact_string, bpred_string)
                            if step < 2:
                                logger.debug('%d:%s:%s', test_image_index, bact_string, bpred_string)
                            test_image_index += 1

                testPredictionLogger.info('Predictions for Collisions (Epoch %d)',epoch)
                test_image_index = 0
                for pred, act in zip(all_bump_predictions, all_bump_labels):
                    bpred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
                    bact_string = ''.join(['%.3f' % a + ',' for a in act.tolist()])
                    testPredictionLogger.info('%d:%s:%s', test_image_index, bact_string, bpred_string)
                    test_image_index += 1
                testPredictionLogger.info('\n')

                test_col_precision = models_utils.precision_multiclass(all_predictions, all_labels, use_argmin=True, max_thresh=max_thresh, min_thresh=min_thresh)
                test_col_recall = models_utils.recall_multiclass(all_predictions, all_labels, use_argmin=True, max_thresh=max_thresh, min_thresh=min_thresh)
                bumpPredictionlogger.info('\n')

                print('\t\tAverage bump test accuracy: %.5f ' % np.mean(bump_test_accuracy))
                print('\t\tAverage bump test (soft) accuracy: %.5f ' % np.mean(bump_soft_accuracy))
                print('\t\tAverage bump test precision: %s', test_col_precision)
                print('\t\tAverage bump test recall: %s', test_col_recall)

                # create a dictionary img_id => image
                bump_image_list = np.split(all_bump_images, all_bump_images.shape[0])
                bump_id_list = all_bump_img_ids.tolist()
                bump_dict_id_image = dict(zip(bump_id_list, bump_image_list))

                bump_correct_hard_ids_sorted = {}
                bump_predicted_hard_ids_sorted = {}
                for di, direct in enumerate(['left', 'straight', 'right']):
                    bump_correct_hard_ids_sorted[direct] = models_utils.get_id_vector_for_correctly_predicted_samples(
                        all_bump_img_ids, all_bump_predictions, all_bump_labels, di, enable_soft_accuracy=False, use_argmin=True)
                    bump_predicted_hard_ids_sorted[direct] = models_utils.get_id_vector_for_predicted_samples(
                        all_bump_img_ids, all_bump_predictions, all_bump_labels, di, enable_soft_accuracy=False, use_argmin=True
                    )
                logger.info('correct hard img ids for: %s', bump_correct_hard_ids_sorted)
                visualizer.save_fig_with_predictions_for_direction(bump_correct_hard_ids_sorted, bump_dict_id_image,
                                                                   IMG_DIR + os.sep + 'bump_correct_predicted_results_hard_%d.png' % (
                                                                   epoch+1))
                visualizer.save_fig_with_predictions_for_direction(bump_predicted_hard_ids_sorted, bump_dict_id_image,
                                                                   IMG_DIR + os.sep + 'bump_predicted_results_hard_%d.png' % (
                                                                       epoch+1))

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
