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
                        tf.get_variable(config.TF_WEIGHTS_STR,shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.02,dtype=tf.float32))
                        tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_NAIVE[scope][-1],
                                               initializer = tf.constant_initializer(0.001,dtype=tf.float32))

                        with tf.variable_scope(config.TF_WEIGHTS_STR):
                            tf.get_variable(config.TF_MOMENTUM_STR,shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope],
                                            initializer = tf.constant_initializer(0,dtype=tf.float32))
                        with tf.variable_scope(config.TF_BIAS_STR):
                            tf.get_variable(config.TF_MOMENTUM_STR,shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope][-1],
                                            initializer = tf.constant_initializer(0,dtype=tf.float32))

                except ValueError as e:
                    logger.critical(e)
                    logger.debug('Variables in scope %s already initialized\n'%scope)

        print([v.name for v in tf.global_variables()])


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
                weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)

                if 'conv' in scope:
                    logger.info('\t\tConvolution with ReLU activation for %s',scope)
                    if si == 0:
                        h = tf.nn.relu(tf.nn.conv2d(tf_inputs,weight,strides=config.TF_ANG_STRIDES[scope],padding='SAME')+bias,name='hidden')
                    else:
                        h = tf.nn.relu(tf.nn.conv2d(h, weight, strides=config.TF_ANG_STRIDES[scope], padding='SAME') + bias,
                                       name='hidden')
                else:
                    # Reshaping required for the first fulcon layer
                    if scope == 'out':
                        logger.info('\t\tFully-connected with output Logits for %s',scope)
                        h = tf.matmul(h, weight) + bias

                    elif 'fc' in scope:
                        if scope == config.TF_FIRST_FC_ID:
                            h_shape = h.get_shape().as_list()
                            logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s', scope, h_shape)
                            h = tf.reshape(h, [batch_size, h_shape[1] * h_shape[2] * h_shape[3]])
                            h = tf.nn.relu(tf.matmul(h, weight) + bias)

                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError

    return h


def predictions_with_logits(logits):
    pred,_ = tf.nn.tanh(logits)
    return pred


def predictions_with_inputs(tf_inputs):
    tf_logits = logits(tf_inputs)
    return tf.nn.tanh(tf_logits)


def calculate_hybrid_loss(tf_noncol_inputs,tf_noncol_labels,tf_col_inputs,tf_col_labels):
    tf_col_out = logits(tf_col_inputs,tf_col_labels,True)
    tf_noncol_out = logits(tf_noncol_inputs,tf_noncol_labels,False)

    tf_col_arg = tf.cast(tf.reduce_min(tf_col_labels, axis=1), dtype=tf.float32)
    tf_noncol_arg = tf.cast(tf.reduce_max(tf_noncol_labels, axis=1), dtype=tf.float32)

    loss = tf.reduce_mean((tf.nn.tanh(tf_col_out) - tf_col_arg)**2 + (tf.nn.tanh(tf_noncol_out) - tf_noncol_arg)**2)
    return loss


def calculate_loss(tf_inputs, tf_labels):

    tf_out = logits(tf_inputs)
    loss = tf.reduce_mean((tf.nn.tanh(tf_out) - tf_labels)**2)
    return loss,tf_out


def optimize_model(loss, tf_labels, global_step,increment_global_step, use_masking,collision):
    momentum = 0.01
    mom_update_ops = []
    grads_and_vars = []
    learning_rate = tf.maximum(
        tf.train.exponential_decay(0.05, global_step, decay_steps=500, decay_rate=0.95, staircase=True,
                                   name='learning_rate_decay'), 1e-4)

    if use_masking:
        if not collision:
            index_to_keep = tf.cast(tf.reduce_mean(tf.argmax(tf_labels, axis=1)),
                                    dtype=tf.int32)
        else:
            index_to_keep = tf.cast(tf.reduce_mean(tf.argmin(tf_labels, axis=1)),
                                    dtype=tf.int32)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    for si, scope in enumerate(config.TF_ANG_SCOPES):
        with tf.variable_scope(scope,reuse=True):
            w,b = tf.get_variable(config.TF_WEIGHTS_STR),tf.get_variable(config.TF_BIAS_STR)
            [(g_w,w),(g_b,b)] = optimizer.compute_gradients(loss,[w,b])

            with tf.variable_scope(config.TF_WEIGHTS_STR,reuse=True):
                w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                if use_masking and scope == 'fc1':
                    mask_vel = tf.transpose(tf.scatter_nd(tf.expand_dims(
                        tf.range(index_to_keep*config.FC1_WEIGHTS,(index_to_keep+1)*config.FC1_WEIGHTS),axis=-1),
                        tf.ones([config.FC1_WEIGHTS,config.TF_ANG_VAR_SHAPES_NAIVE['fc1'][0]],dtype=tf.float32),
                        [config.TF_ANG_VAR_SHAPES_NAIVE['fc1'][1],config.TF_ANG_VAR_SHAPES_NAIVE['fc1'][0]])
                    )
                    mom_update_ops.append(tf.assign(w_vel,(momentum*w_vel + g_w)*mask_vel))
                elif use_masking and scope == 'out':
                    mask_vel = tf.transpose(
                        tf.scatter_nd([[index_to_keep]],tf.ones([1,config.FC1_WEIGHTS],dtype=tf.float32),
                                      [config.TF_ANG_VAR_SHAPES_NAIVE['out'][1],config.TF_ANG_VAR_SHAPES_NAIVE['out'][0]])
                    )
                    mom_update_ops.append(tf.assign(w_vel, (momentum * w_vel + g_w) * mask_vel))
                else:
                    mom_update_ops.append(tf.assign(w_vel, momentum * w_vel + g_w))
                grads_and_vars.append((w_vel * learning_rate, w))
            with tf.variable_scope(config.TF_BIAS_STR,reuse=True):
                # TODO: MASKING FOR BIAS
                b_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                if use_masking and scope=='fc1':
                    mask_vel = tf.scatter_nd(tf.expand_dims(
                        tf.range(index_to_keep*config.FC1_WEIGHTS,(index_to_keep+1)*config.FC1_WEIGHTS),axis=-1),
                        tf.ones([config.FC1_WEIGHTS], dtype=tf.float32),
                        [config.TF_ANG_VAR_SHAPES_NAIVE['fc1'][1]]
                    )
                    mom_update_ops.append(tf.assign(b_vel, (momentum * b_vel + g_b) * mask_vel))
                elif use_masking and scope=='out':
                    mask_vel = tf.scatter_nd([[index_to_keep]],tf.ones([1],dtype=tf.float32),[config.TF_ANG_VAR_SHAPES_NAIVE['out'][1]])
                    mom_update_ops.append(tf.assign(b_vel, (momentum * b_vel + g_b)*mask_vel))
                else:
                    mom_update_ops.append(tf.assign(b_vel,momentum*b_vel + g_b))
                grads_and_vars.append((b_vel * learning_rate, b))

    if increment_global_step:
        optimize = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    else:
        optimize = optimizer.apply_gradients(grads_and_vars)

    return optimize,mom_update_ops,grads_and_vars


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

    dataset_filenames = {'train_dataset':{'left':[
                                                  '..' + os.sep + 'sample-with-dir-1' + os.sep + 'image-direction-%d-0.tfrecords' % i
                                                  for i in range(1)
                                                  ] +
                                              [
                                                  '..' + os.sep + 'sample-with-dir-2' + os.sep + 'image-direction-%d-0.tfrecords' % i
                                                  for i in range(1)
                                                  ],
                                         'straight':[
                                                   '..' + os.sep + 'sample-with-dir-1' + os.sep + 'image-direction-%d-1.tfrecords' % i
                                                   for i in range(2)] +
                                               [
                                                   '..' + os.sep + 'sample-with-dir-2' + os.sep + 'image-direction-%d-1.tfrecords' % i
                                                   for i in range(2)],
                                         'right':[
                                                   '..' + os.sep + 'sample-with-dir-1' + os.sep + 'image-direction-%d-2.tfrecords' % i
                                                   for i in range(2)] +
                                               [
                                                   '..' + os.sep + 'sample-with-dir-2' + os.sep + 'image-direction-%d-2.tfrecords' % i
                                                   for i in range(1)]
                                          },

                         'train_bump_dataset':{
                             'left': [
                                 '..' + os.sep + 'sample-with-dir-1-bump' + os.sep + 'image-direction-%d-0.tfrecords' % i
                                 for i in range(1)],
                             'straight': [
                                '..' + os.sep + 'sample-with-dir-1-bump' + os.sep + 'image-direction-%d-1.tfrecords' % i
                                for i in range(1)],
                             'right':[
                                '..' + os.sep + 'sample-with-dir-1-bump' + os.sep + 'image-direction-%d-2.tfrecords' % i
                                for i in range(1)]
                         },

                         'test_dataset': ['..' + os.sep + 'sample-with-dir-3' + os.sep + 'image-direction-%d.tfrecords' % i for i in range(4)],
                         'test_bump_dataset': ['..' + os.sep + 'sample-with-dir-3-bump' + os.sep + 'image-direction-0.tfrecords']}

    dataset_sizes = {'train_dataset':1000+500,
                     'train_bump_dataset': 600,
                     'test_dataset': 500,
                     'test_bump_dataset': 100}

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
    accuracy_logger.info('#Epoch;Accuracy;Non-collision Accuracy(Soft);Collision Accuracy;' +
                         'Collision Accuracy (Soft);Preci-L,Preci-S,Preci-R;Rec-L,Rec-S,Rec-R')

    batch_size = 5

    graph = tf.Graph()
    configp = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.InteractiveSession(graph=graph,config=configp)

    with sess.as_default() and graph.as_default():
        build_tensorflw_variables()
        models_utils.set_from_main(sess,graph,logger)

        global_step = tf.Variable(0,trainable=False)

        all_train_files,all_bump_files = [],[]
        all_test_files,all_bump_test_files = [],[]
        for di in ['left','straight','right']:
            all_train_files.extend(dataset_filenames['train_dataset'][di])
            all_bump_files.extend(dataset_filenames['train_bump_dataset'][di])
            all_test_files.extend(dataset_filenames['test_dataset'])
            all_bump_test_files.extend(dataset_filenames['test_bump_dataset'])

        tf_images,tf_labels = models_utils.build_input_pipeline(
            all_train_files,batch_size,shuffle=True,
            training_data=False,use_opposite_label=False,inputs_for_sdae=False)

        tf_bump_images, tf_bump_labels = models_utils.build_input_pipeline(
            all_bump_files, batch_size, shuffle=True,
            training_data=False, use_opposite_label=True,inputs_for_sdae=False)

        tf_loss, tf_logits = calculate_loss(tf_images, tf_labels)

        if not config.OPTIMIZE_HYBRID_LOSS:
            tf_bump_loss, tf_bump_logits  = calculate_loss(tf_bump_images, tf_bump_labels)

            tf_optimize, tf_mom_update_ops, tf_grads = optimize_model(tf_loss, tf_labels,
                                                                                                       global_step, increment_global_step=True,
                                                                                                       use_masking=False,collision=False)
            tf_bump_optimize, tf_bump_mom_update_ops, tf_bump_grads = optimize_model(tf_bump_loss, tf_labels,
                                                                                                                      global_step,increment_global_step=False,
                                                                                                                      use_masking=False,collision=True)
        else:
            raise NotImplementedError

        tf_train_predictions = predictions_with_inputs(tf_images)
        tf_train_bump_predictions = predictions_with_inputs(tf_bump_images)

        tf_test_images,tf_test_labels = models_utils.build_input_pipeline(all_test_files,batch_size,shuffle=False,
                                                             training_data=False,use_opposite_label=False,inputs_for_sdae=False)
        tf_bump_test_images, tf_bump_test_labels = models_utils.build_input_pipeline(all_bump_test_files, batch_size, shuffle=False,
                                                              training_data=False,use_opposite_label=True,inputs_for_sdae=False)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf_test_predictions = predictions_with_inputs(tf_test_images)
        tf_bump_test_predictions = predictions_with_inputs(tf_bump_test_images)

        tf.global_variables_initializer().run(session=sess)

        for epoch in range(100):
            avg_loss = []
            avg_train_accuracy = []

            if not config.OPTIMIZE_HYBRID_LOSS:
                for step in range(dataset_sizes['train_dataset']//batch_size):

                    l1, _, _, pred,train_labels = sess.run([tf_loss, tf_optimize,tf_mom_update_ops,
                                                                        tf_train_predictions,tf_labels])

                    avg_loss.append(l1)
                    avg_train_accuracy.append(models_utils.soft_accuracy(pred,train_labels,use_argmin=False))

                    if step < 2:
                        logger.debug('Predictions for Non-Collided data')
                        for pred,lbl in zip(pred,train_labels):
                            logger.debug('\t%s;%s',pred,lbl)
                logger.info('\tAverage Loss for Epoch %d: %.5f' %(epoch,np.mean(l1)))
                logger.info('\t\t Training accuracy: %.3f'%np.mean(avg_train_accuracy))
                for it in range(3):

                    avg_bump_loss = []
                    avg_bump_train_accuracy = []
                    for step in range(dataset_sizes['train_bump_dataset']//batch_size):

                        bump_l1,bump_logits, _, _, bump_pred, train_bump_labels = sess.run([tf_bump_loss, tf_bump_logits,
                                                                                                         tf_bump_optimize, tf_bump_mom_update_ops,
                                                                                                         tf_train_bump_predictions, tf_bump_labels])
                        avg_bump_loss.append(bump_l1)
                        avg_bump_train_accuracy.append(models_utils.soft_accuracy(bump_pred,train_bump_labels,use_argmin=True))

                        if it==0 and step<2:
                            logger.debug('Predictions for Collided data')
                            for pred,lbl in zip(bump_pred,train_bump_labels):
                                logger.debug('\t%s;%s',pred,lbl)

                logger.info('\tAverage Bump Loss (Train) for Epoch %d: %.5f'%(epoch,np.mean(avg_bump_loss)))
                logger.info('\t\tAverage Bump Accuracy (Train) for Epoch %d: %.5f' % (epoch, np.mean(avg_bump_train_accuracy)))

            else:
                avg_loss = []
                avg_noncol_train_accuracy = []
                avg_col_train_accuracy = []
                for step in range(dataset_sizes['train_dataset']//batch_size):

                    l1, _, _, = sess.run([tf_loss, tf_optimize,tf_mom_update_ops])
                    noncol_pred,noncol_labels = sess.run([tf_train_predictions,tf_labels])
                    col_pred, col_labels = sess.run([tf_train_bump_predictions, tf_bump_labels])
                    avg_loss.append(l1)
                    avg_noncol_train_accuracy.append(models_utils.soft_accuracy(noncol_pred,noncol_labels, use_argmin=False))
                    avg_col_train_accuracy.append(
                        models_utils.soft_accuracy(col_pred, col_labels, use_argmin=True))

                logger.info('\tAverage Loss (Hybrid) for Epoch %d: %.5f' % (epoch, np.mean(l1)))
                logger.info('\t\t Training accuracy: %.3f' % np.mean(avg_noncol_train_accuracy))
                logger.info(
                    '\t\tAverage Bump Accuracy (Train) for Epoch %d: %.5f' % (epoch, np.mean(avg_col_train_accuracy)))

            if (epoch+1)%5==0:
                test_accuracy = []
                soft_test_accuracy = []
                all_predictions, all_labels = None, None
                test_image_index = 0
                for step in range(dataset_sizes['test_dataset']//batch_size):
                    predicted_labels,actual_labels = sess.run([tf_test_predictions,tf_test_labels])

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

                    if step<5:
                        logger.debug('Test Predictions (Non-Collisions)')
                    for pred,act in zip(predicted_labels,actual_labels):
                        pred_string = ''.join(['%.3f'%p+',' for p in pred.tolist()])
                        act_string = ''.join([str(int(a))+',' for a in act.tolist()])
                        predictionlogger.info('%d:%s:%s',test_image_index,act_string,pred_string)
                        if step < 5:
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
                for step in range(dataset_sizes['test_bump_dataset']//batch_size):
                    bump_predicted_labels, bump_actual_labels = sess.run([tf_bump_test_predictions, tf_bump_test_labels])
                    bump_test_accuracy.append(models_utils.accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))
                    bump_soft_accuracy.append(models_utils.soft_accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))

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

                print('\t\tAverage bump test accuracy: %.5f ' % np.mean(bump_test_accuracy))
                print('\t\tAverage bump test (soft) accuracy: %.5f ' % np.mean(bump_soft_accuracy))

                precision_string = ''.join(['%.3f,'%test_noncol_precision[pi] for pi in range(3)])
                recall_string = ''.join(['%.3f,'%test_noncol_recall[ri] for ri in range(3)])

                accuracy_logger.info('%d;%.3f;%.3f;%.3f;%.3f;%s;%s',epoch,np.mean(test_accuracy),np.mean(soft_test_accuracy),
                                     np.mean(bump_test_accuracy),np.mean(bump_soft_accuracy),precision_string,recall_string)
        coord.request_stop()
        coord.join(threads)
