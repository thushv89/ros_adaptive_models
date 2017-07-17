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

                        tf.get_variable(config.TF_WEIGHTS_STR,shape=config.TF_ANG_VAR_SHAPES[scope],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.02,dtype=tf.float32))
                        tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES[scope][-1],
                                               initializer = tf.constant_initializer(0.001,dtype=tf.float32))

                        with tf.variable_scope(config.TF_WEIGHTS_STR):
                            tf.get_variable(config.TF_MOMENTUM_STR,shape=config.TF_ANG_VAR_SHAPES[scope],
                                            initializer = tf.constant_initializer(0,dtype=tf.float32))
                        with tf.variable_scope(config.TF_BIAS_STR):
                            tf.get_variable(config.TF_MOMENTUM_STR,shape=config.TF_ANG_VAR_SHAPES[scope][-1],
                                            initializer = tf.constant_initializer(0,dtype=tf.float32))

                except ValueError as e:
                    logger.critical(e)
                    logger.debug('Variables in scope %s already initialized\n'%scope)

        print([v.name for v in tf.global_variables()])


def logits(tf_inputs,tf_labels,collision):
    '''
    Inferencing the model. The input (tf_inputs) is propagated through convolutions poolings
    fully-connected layers to obtain the final softmax output
    :param tf_inputs: a batch of images (tensorflow placeholder)
    :return:
    '''
    global logger
    logger.info('Defining inference ops ...')
    index_to_keep = None
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
                        if tf_labels is not None:
                            if not collision:
                                index_to_keep = tf.cast(tf.reduce_mean(tf.argmax(tf_labels, axis=1)), dtype=tf.int32)
                            else:
                                index_to_keep = tf.cast(tf.reduce_mean(tf.argmin(tf_labels, axis=1)), dtype=tf.int32)

                            transposed_masked_weights = tf.gather_nd(tf.transpose(weight),[[index_to_keep]])
                            masked_bias = tf.gather(bias,[index_to_keep])
                            h = tf.matmul(h, tf.transpose(transposed_masked_weights)) + masked_bias
                            h = tf.squeeze(h,name='hidden_squeezed')

                        else:
                            h_out_list = []
                            for n_i in range(config.TF_NUM_CLASSES):
                                transposed_masked_weights = tf.gather_nd(tf.transpose(weight), [[n_i]])
                                masked_bias = tf.gather(bias, [n_i])
                                h_out_list.append(tf.matmul(h_list[n_i], tf.transpose(transposed_masked_weights)) + masked_bias)

                            h = tf.squeeze(tf.stack(h_out_list,axis=1))

                    elif 'fc' in scope:
                        if scope == config.TF_FIRST_FC_ID:
                            h_shape = h.get_shape().as_list()
                            logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s', scope, h_shape)
                            h = tf.reshape(h, [batch_size, h_shape[1] * h_shape[2] * h_shape[3]])
                            if tf_labels is not None:
                                if not collision:
                                    index_to_keep = tf.cast(tf.reduce_mean(tf.argmax(tf_labels, axis=1)),
                                                            dtype=tf.int32)
                                else:
                                    index_to_keep = tf.cast(tf.reduce_mean(tf.argmin(tf_labels, axis=1)),
                                                            dtype=tf.int32)

                                transposed_masked_weights = tf.gather_nd(tf.transpose(weight), tf.expand_dims(tf.range(index_to_keep*256,(index_to_keep+1)*256),axis=-1))
                                masked_bias = tf.gather(bias, [tf.range(index_to_keep*256,(index_to_keep+1)*256)])
                                h = tf.nn.relu(tf.matmul(h, tf.transpose(transposed_masked_weights)) + masked_bias)

                            else:
                                h_list = []
                                for n_i in range(config.TF_NUM_CLASSES):
                                    transposed_masked_weights = tf.gather_nd(tf.transpose(weight),
                                        tf.expand_dims(tf.range(n_i * 256, (n_i + 1) * 256),axis=-1))
                                    masked_bias = tf.gather(bias,
                                                            [tf.range(n_i * 256, (n_i + 1) * 256)])
                                    h_list.append(tf.nn.relu(tf.matmul(h, tf.transpose(transposed_masked_weights)) + masked_bias))

                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError

    return h,index_to_keep


def predictions_with_logits(logits):
    pred,_ = tf.nn.sigmoid(logits,None,False)
    return pred


def predictions_with_inputs(tf_inputs):
    tf_logits,_ = logits(tf_inputs,None,False)
    return tf.nn.sigmoid(tf_logits)


def calculate_loss(tf_inputs,tf_labels,collision):

    tf_out,index_to_keep = logits(tf_inputs,tf_labels,collision)
    if collision:
        tf_labels_arg = tf.cast(tf.reduce_min(tf_labels,axis=1),dtype=tf.float32)
    else:
        tf_labels_arg = tf.cast(tf.reduce_max(tf_labels,axis=1),dtype=tf.float32)
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf_out, labels=tf_labels_arg))
    loss = tf.reduce_mean((tf.nn.sigmoid(tf_out) - tf_labels_arg)**2)
    return loss,tf_out,tf_labels_arg,index_to_keep


def optimize_model(loss, global_step,increment_global_step):

    mom_update_ops = []
    grads_and_vars = []
    learning_rate = tf.maximum(
        tf.train.exponential_decay(0.01, global_step, decay_steps=500, decay_rate=0.95, staircase=True,
                                   name='learning_rate_decay'), 1e-4)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    for si, scope in enumerate(config.TF_ANG_SCOPES):
        with tf.variable_scope(scope,reuse=True):
            w,b = tf.get_variable(config.TF_WEIGHTS_STR),tf.get_variable(config.TF_BIAS_STR)
            [(g_w,w),(g_b,b)] = optimizer.compute_gradients(loss,[w,b])

            with tf.variable_scope(config.TF_WEIGHTS_STR,reuse=True):
                w_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                mom_update_ops.append(tf.assign(w_vel,0.9*w_vel + g_w))
                grads_and_vars.append((w_vel * learning_rate, w))
            with tf.variable_scope(config.TF_BIAS_STR,reuse=True):
                b_vel = tf.get_variable(config.TF_MOMENTUM_STR)
                mom_update_ops.append(tf.assign(b_vel,0.9*b_vel + g_b))
                grads_and_vars.append((b_vel * learning_rate, b))

    if increment_global_step:
        #optimize = tf.train.AdamOptimizer(beta1=0.9,learning_rate=learning_rate).minimize(loss,global_step=global_step)
        #optimize = tf.train.MomentumOptimizer(momentum=tf.constant(0.9,dtype=tf.float32), learning_rate=learning_rate).minimize(loss,global_step=global_step)
        optimize = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    else:
        #optimize = tf.train.AdamOptimizer(beta1=0.9, learning_rate=learning_rate).minimize(loss)
        #optimize = tf.train.MomentumOptimizer(momentum=tf.constant(0.9,dtype=tf.float32), learning_rate=learning_rate).minimize(loss)
        optimize = optimizer.apply_gradients(grads_and_vars)

    #optimize = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=learning_rate).minimize(loss)
    return optimize,mom_update_ops


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

                         'test_dataset: ': ['..' + os.sep + 'sample-with-dir-3' + os.sep + 'image-direction-%d.tfrecords' % i for i in range(4)],
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

        tf_images, tf_labels = {},{}
        tf_bump_images, tf_bump_labels = {},{}
        tf_loss, tf_logits, tf_labels_arg = {},{},{}
        tf_bump_loss, tf_bump_logits, tf_bump_labels_arg = {},{},{}
        tf_optimize,tf_mom_update_ops = {},{}
        tf_bump_optimize,tf_bump_mom_update_ops = {},{}
        tf_train_predictions, tf_train_bump_predictions = {},{}
        tf_index_to_keep,tf_bump_index_to_keep = {},{}

        for direction in ['left','straight','right']:
            tf_images[direction],tf_labels[direction] = models_utils.build_input_pipeline(
                dataset_filenames['train_dataset'][direction],batch_size,shuffle=True,
                training_data=False,use_opposite_label=False,inputs_for_sdae=False)

            tf_bump_images[direction], tf_bump_labels[direction] = models_utils.build_input_pipeline(
                dataset_filenames['train_bump_dataset'][direction], batch_size, shuffle=True,
                training_data=False, use_opposite_label=True,inputs_for_sdae=False)

            tf_loss[direction], tf_logits[direction], \
            tf_labels_arg[direction], tf_index_to_keep[direction] \
                = calculate_loss(tf_images[direction], tf_labels[direction], collision=False)

            tf_bump_loss[direction], tf_bump_logits[direction], \
            tf_bump_labels_arg[direction], tf_bump_index_to_keep[direction] \
                = calculate_loss(tf_bump_images[direction], tf_bump_labels[direction], collision=True)

            tf_optimize[direction], tf_mom_update_ops[direction] = optimize_model(tf_loss[direction], global_step, increment_global_step=True)
            tf_bump_optimize[direction], tf_bump_mom_update_ops[direction] = optimize_model(tf_bump_loss[direction], global_step,
                                                                      increment_global_step=False)

            tf_train_predictions[direction] = predictions_with_inputs(tf_images[direction])
            tf_train_bump_predictions[direction] = predictions_with_inputs(tf_bump_images[direction])

        tf_test_images,tf_test_labels = models_utils.build_input_pipeline(dataset_filenames['test_dataset: '],batch_size,shuffle=False,
                                                             training_data=False,use_opposite_label=False,inputs_for_sdae=False)
        tf_bump_test_images, tf_bump_test_labels = models_utils.build_input_pipeline(dataset_filenames['test_bump_dataset'], batch_size, shuffle=False,
                                                              training_data=False,use_opposite_label=True,inputs_for_sdae=False)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf_test_predictions = predictions_with_inputs(tf_test_images)
        tf_bump_test_predictions = predictions_with_inputs(tf_bump_test_images)

        tf.global_variables_initializer().run(session=sess)

        '''# due to flipping in data augmentation left and right data can swap but should be a problem for optimization
        for _ in range(5):
            for direction in ['left', 'straight', 'right']:
                tmp_labels_direction,tmp_ind_direction = sess.run([tf_labels[direction],tf_index_to_keep[direction]])
                print(direction)
                print(tmp_labels_direction)
                print(tmp_ind_direction)
            print('\n')
        sys.exit(1)'''

        for epoch in range(100):
            avg_loss = []
            avg_train_accuracy = []
            for step in range(dataset_sizes['train_dataset']//batch_size):
                rand_direction = np.random.choice(['left','straight','right'])
                l1, _, _, pred,train_labels,ind_to_keep = sess.run([tf_loss[rand_direction], tf_optimize[rand_direction],tf_mom_update_ops[rand_direction],
                                                        tf_train_predictions[rand_direction],tf_labels[rand_direction],tf_index_to_keep[rand_direction]])
                #l1, pred, train_labels = sess.run(
                #    [tf_loss, tf_train_predictions, tf_labels])
                #logger.debug('%s;%d',rand_direction,ind_to_keep)
                #logger.debug(train_labels)
                #logger.debug('\n')
                avg_loss.append(l1)
                avg_train_accuracy.append(models_utils.soft_accuracy(pred,train_labels,use_argmin=False))

                if step < 2:
                    logger.debug('Predictions for Non-Collided data')
                    for pred,lbl in zip(pred,train_labels):
                        logger.debug('\t%s;%s',pred,lbl)
            logger.info('\tAverage Loss for Epoch %d: %.5f' %(epoch,np.mean(l1)))
            logger.info('\t\t Training accuracy: %.3f'%np.mean(avg_train_accuracy))
            for it in range(2):

                avg_bump_loss = []
                avg_bump_train_accuracy = []
                for step in range(dataset_sizes['train_bump_dataset']//batch_size):
                    rand_direction = np.random.choice(['left', 'straight', 'right'])
                    bump_l1,bump_logits,bump_label_args, _, _, bump_pred, train_bump_labels = sess.run([tf_bump_loss[direction], tf_bump_logits[direction], tf_bump_labels_arg[direction],
                                                                                                     tf_bump_optimize[direction], tf_bump_mom_update_ops[direction],
                                                                                                     tf_train_bump_predictions[direction], tf_bump_labels[direction]])
                    avg_bump_loss.append(bump_l1)
                    avg_bump_train_accuracy.append(models_utils.soft_accuracy(bump_pred,train_bump_labels,use_argmin=True))

                    if it==0 and step<2:
                        logger.debug('Predictions for Collided data')
                        for pred,lbl in zip(bump_pred,train_bump_labels):
                            logger.debug('\t%s;%s',pred,lbl)

            logger.info('\tAverage Bump Loss (Train) for Epoch %d: %.5f'%(epoch,np.mean(avg_bump_loss)))
            logger.info('\t\tAverage Bump Accuracy (Train) for Epoch %d: %.5f' % (epoch, np.mean(avg_bump_train_accuracy)))
                #print('\t\t Training accuracy: %.3f' % soft_accuracy(pred, train_labels, use_argmin=False))
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

                    logger.debug('Test Predictions (Non-Collisions)')
                    for pred,act in zip(predicted_labels,actual_labels):
                        pred_string = ''.join(['%.3f'%p+',' for p in pred.tolist()])
                        act_string = ''.join([str(int(a))+',' for a in act.tolist()])
                        predictionlogger.info('%d:%s:%s',test_image_index,act_string,pred_string)
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

                test_image_index =0
                for pred, act in zip(all_bump_predictions,all_bump_labels):
                    bpred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
                    bact_string = ''.join([str(int(a)) + ',' for a in act.tolist()])
                    bumpPredictionlogger.info('%d:%s:%s', test_image_index, bact_string, bpred_string)
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
