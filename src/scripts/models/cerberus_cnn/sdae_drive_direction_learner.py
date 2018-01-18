import numpy as np
import tensorflow as tf
import logging
import sys
from math import ceil
import os
from PIL import Image
import getopt
import models_utils
import config

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

graph = tf.get_default_graph()
sess = tf.InteractiveSession(graph=graph)


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
        for si,scope in enumerate(config.TF_SDAE_ANG_SCOPES):
            with tf.variable_scope(scope) as sc:

                # Try Except because if you try get_variable with an intializer and
                # the variable exists, you will get a ValueError saying the variable exists

                try:
                    weights = tf.get_variable(config.TF_WEIGHTS_STR,shape=config.TF_SDAE_ANG_VAR_SHAPES[scope],
                                              initializer=tf.truncated_normal_initializer(stddev=0.02,dtype=tf.float32))
                    bias = tf.get_variable(config.TF_BIAS_STR, config.TF_SDAE_ANG_VAR_SHAPES[scope][-1],
                                           initializer = tf.constant_initializer(0.001,dtype=tf.float32))

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
        for si, scope in enumerate(config.TF_SDAE_ANG_SCOPES):
            with tf.variable_scope(scope,reuse=True) as sc:
                weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)

                if 'fc' in scope:
                    logger.info('\t\tFulcon with ReLU activation for %s',scope)
                    if si == 0:
                        h = tf.nn.relu(tf.matmul(tf_inputs,weight)+bias,name='hidden')
                    else:
                        h = tf.nn.relu(tf.matmul(h, weight) + bias, name='hidden')
                else:
                    # Reshaping required for the first fulcon layer
                    if scope == 'out':
                        logger.info('\t\tFully-connected with output Logits for %s',scope)
                        h = tf.matmul(h, weight) + bias
                    else:
                        raise NotImplementedError
    return h


def predictions_with_logits(logits):
    pred = tf.nn.sigmoid(logits)
    return pred


def predictions_with_inputs(tf_inputs):
    tf_logits = logits(tf_inputs)
    return tf.nn.sigmoid(tf_logits)


def calculate_loss(tf_inputs,tf_outputs):

    tf_out = logits(tf_inputs)
    if config.ENABLE_RANDOM_MASKING:
        random_mask = tf.cast(tf.equal(tf_outputs,tf.constant(0,dtype=tf.float32)),tf.float32) * \
                      tf.cast(tf.greater(tf.random_normal([batch_size,config.TF_NUM_CLASSES]),
                                         tf.constant(value=-0.25,shape=[batch_size,config.TF_NUM_CLASSES])),tf.float32)
        random_mask = random_mask + tf_outputs
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=random_mask*tf_out,labels=tf_outputs))
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf_out, labels=tf_outputs))

    return loss


def optimize_model(loss,global_step,increment_global_step):

    learning_rate = tf.minimum(
        tf.train.exponential_decay(0.002, global_step, decay_steps=100, decay_rate=0.9, staircase=True,
                                   name='learning_rate_decay'), 0.0001)
    if increment_global_step:
        optimize = tf.train.AdamOptimizer(beta1=0.9,learning_rate=learning_rate).minimize(loss,global_step=global_step)
    else:
        optimize = tf.train.AdamOptimizer(beta1=0.9, learning_rate=learning_rate).minimize(loss)
    #optimize = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=learning_rate).minimize(loss)
    return optimize


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

    dataset_filenames = {'train_dataset':['..' + os.sep + 'sample-with-dir-1' + os.sep + 'image-direction-%d.tfrecords' % i for i in range(10)] +
                                         ['..' + os.sep + 'sample-with-dir-2' + os.sep + 'image-direction-%d.tfrecords' % i for i in range(5)],
                         'train_bump_dataset':['..' + os.sep + 'sample-with-dir-1-bump' + os.sep + 'image-direction-%d.tfrecords' % i for i in range(1)],
                         'test_dataset: ': ['..' + os.sep + 'sample-with-dir-3' + os.sep + 'image-direction-%d.tfrecords' % i for i in range(4)],
                         'test_bump_dataset': ['..' + os.sep + 'sample-with-dir-3-bump' + os.sep + 'image-direction-0.tfrecords']}

    dataset_sizes = {'train_dataset':1000+500,
                     'train_bump_dataset': 100,
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
    accuracy_logger.info('#Epoch:Accuracy:Accuracy(Soft):Bump Accuracy: Bump Accuracy (Soft)')

    batch_size = 25
    with sess.as_default() and graph.as_default():
        build_tensorflw_variables()
        models_utils.set_from_main(sess,graph,logger)

        global_step = tf.Variable(0,trainable=False)
        tf_images,tf_labels = models_utils.build_input_pipeline(dataset_filenames['train_dataset'], batch_size,shuffle=True,
                                                   training_data=True,use_opposite_label=False,inputs_for_sdae=True)
        tf_bump_images, tf_bump_labels = models_utils.build_input_pipeline(dataset_filenames['train_bump_dataset'], batch_size, shuffle=True,
                                                              training_data=True, use_opposite_label=True,inputs_for_sdae=True)
        tf_test_images,tf_test_labels = models_utils.build_input_pipeline(dataset_filenames['test_dataset: '],batch_size,shuffle=False,
                                                             training_data=False,use_opposite_label=False,inputs_for_sdae=True)
        tf_bump_test_images, tf_bump_test_labels = models_utils.build_input_pipeline(dataset_filenames['test_bump_dataset'], batch_size, shuffle=False,
                                                              training_data=False,use_opposite_label=True,inputs_for_sdae=True)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf_loss = calculate_loss(tf_images, tf_labels)
        tf_bump_loss = calculate_loss(tf_bump_images,tf_bump_labels)

        tf_optimize = optimize_model(tf_loss,global_step,increment_global_step=True)
        tf_bump_optimize = optimize_model(tf_bump_loss,global_step,increment_global_step=False)
        tf_train_predictions = predictions_with_inputs(tf_images)

        tf_test_predictions = predictions_with_inputs(tf_test_images)
        tf_bump_test_predictions = predictions_with_inputs(tf_bump_test_images)

        tf.global_variables_initializer().run()

        for epoch in range(50):
            avg_loss = []
            avg_train_accuracy = []
            for step in range(dataset_sizes['train_dataset']//batch_size):
                l1, _, pred,train_labels = sess.run([tf_loss, tf_optimize,tf_train_predictions,tf_labels])
                avg_loss.append(l1)
                avg_train_accuracy.append(models_utils.soft_accuracy(pred,train_labels,use_argmin=False))
            print('\tAverage Loss for Epoch %d: %.5f' %(epoch,np.mean(l1)))
            print('\t\t Training accuracy: %.3f'%np.mean(avg_train_accuracy))
            if (epoch+1)%3==0:
                for step in range(dataset_sizes['train_bump_dataset']//batch_size):
                    sess.run(tf_bump_optimize)
                #print('\t\t Training accuracy: %.3f' % soft_accuracy(pred, train_labels, use_argmin=False))
            if (epoch+1)%5==0:
                test_accuracy = []
                soft_test_accuracy = []
                test_image_index = 0
                for step in range(dataset_sizes['test_dataset']//batch_size):
                    predicted_labels,actual_labels = sess.run([tf_test_predictions,tf_test_labels])
                    test_accuracy.append(models_utils.accuracy(predicted_labels,actual_labels,use_argmin=False))
                    soft_test_accuracy.append(models_utils.soft_accuracy(predicted_labels,actual_labels,use_argmin=False))

                    for pred,act in zip(predicted_labels,actual_labels):
                        pred_string = ''.join(['%.3f'%p+',' for p in pred.tolist()])
                        act_string = ''.join([str(int(a))+',' for a in act.tolist()])
                        predictionlogger.info('%d:%s:%s',test_image_index,act_string,pred_string)
                        test_image_index += 1
                predictionlogger.info('\n')
                print('\t\tAverage test accuracy: %.5f '%np.mean(test_accuracy))
                print('\t\tAverage test accuracy(soft): %.5f'%np.mean(soft_test_accuracy))

                bump_test_accuracy = []
                bump_soft_accuracy  = []
                bump_test_image_index = 0
                for step in range(dataset_sizes['test_bump_dataset']//batch_size):
                    bump_predicted_labels, bump_actual_labels = sess.run([tf_bump_test_predictions, tf_bump_test_labels])
                    bump_test_accuracy.append(models_utils.accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))
                    bump_soft_accuracy.append(models_utils.soft_accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))

                    for pred, act in zip(bump_predicted_labels, bump_actual_labels):
                        bpred_string = ''.join(['%.3f' % p + ',' for p in pred.tolist()])
                        bact_string = ''.join([str(int(a)) + ',' for a in act.tolist()])
                        bumpPredictionlogger.info('%d:%s:%s', test_image_index, bact_string, bpred_string)
                        bump_test_image_index += 1
                    bumpPredictionlogger.info('\n')

                print('\t\tAverage bump test accuracy: %.5f ' % np.mean(bump_test_accuracy))
                print('\t\tAverage bump test (soft) accuracy: %.5f ' % np.mean(bump_soft_accuracy))

                accuracy_logger.info('%d:%.3f:%.3f:%.3f:%.3f',epoch,np.mean(test_accuracy),np.mean(soft_test_accuracy),np.mean(bump_test_accuracy),np.mean(bump_soft_accuracy))
        coord.request_stop()
        coord.join(threads)
