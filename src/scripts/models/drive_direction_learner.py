import numpy as np
import tensorflow as tf
import logging
import sys
from math import ceil
import os
from PIL import Image
import config
import getopt

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

def build_input_pipeline(filenames, batch_size, shuffle, training_data, use_opposite_label):
    '''

    :param filenames: Filenames as a list
    :param batch_size:
    :param shuffle: Shuffle the data when returning
    :param training_data: Use data augmentation (brightness/contrast/flipping) if True
    :param use_opposite_label: This is for bump data (in which we invert labels e.g. if label is 001 we return 110
    :return:
    '''
    global sess, graph
    global logger
    logger.info('Received filenames: %s', filenames)
    with tf.name_scope('sim_preprocess'):
        # FIFO Queue of file names
        # creates a FIFO queue until the reader needs them
        filename_queue = tf.train.string_input_producer(filenames, capacity=2, shuffle=shuffle,name='string_input_producer')

        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)
            else:
                logger.info('File %s found.',f)
        # Reader which takes a filename queue and read() which outputs data one by one
        reader = tf.TFRecordReader()

        key, serial_example = reader.read(filename_queue, name='image_read_op')

        features = tf.parse_single_example(
            serial_example,
            features = {config.FEAT_IMG_RAW : tf.FixedLenFeature([], tf.string),
                        config.FEAT_LABEL : tf.FixedLenFeature([], tf.int64)}
        )

        image = tf.decode_raw(features[config.FEAT_IMG_RAW], tf.float32)
        image = tf.cast(image,tf.float32)
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32, name='float_image')
        image = tf.reshape(image,config.TF_INPUT_SIZE)
        image.set_shape(config.TF_INPUT_SIZE)

        if config.USE_GRAYSCALE:
            image = tf.image.rgb_to_grayscale(image,name='grayscale_image')
        if training_data:
            image = tf.image.random_brightness(image, 0.5, seed=13345432)
            image = tf.image.random_contrast(image, 0, 0.5, seed=2353252)

        label = tf.cast(features[config.FEAT_LABEL], tf.int32)
        if not use_opposite_label:
            if training_data and config.ENABLE_SOFT_CLASSIFICATION:
                one_hot_label = tf.one_hot(label,config.TF_NUM_CLASSES,dtype=tf.float32,on_value=0.9,off_value=0.1)
            else:
                one_hot_label = tf.one_hot(label,config.TF_NUM_CLASSES,dtype=tf.float32)
        else:
            if training_data and config.ENABLE_SOFT_CLASSIFICATION:
                one_hot_label = tf.one_hot(label,config.TF_NUM_CLASSES,dtype=tf.float32,on_value=0.0, off_value=0.9)
            else:
                one_hot_label = tf.one_hot(label, config.TF_NUM_CLASSES, dtype=tf.float32, on_value=0.0, off_value=1.0)

        if training_data:
            flip_rand = tf.random_uniform(shape=[1],minval=0, maxval=1.0, seed=154324654)
            image = tf.cond(flip_rand[0] > 0.75, lambda: tf.image.flip_left_right(image), lambda: image)
            one_hot_label = tf.cond(flip_rand[0] > 0.75, lambda: tf.reverse(one_hot_label,axis=[0]),lambda: one_hot_label)

        # standardize image
        std_image = tf.image.per_image_standardization(image)

        # https://stackoverflow.com/questions/37126108/how-to-read-data-into-tensorflow-batches-from-example-queue

        # The batching mechanism that takes a output produced by reader (with preprocessing) and outputs a batch tensor
        # [batch_size, height, width, depth] 4D tensor
        image_batch,label_batch = tf.train.batch([std_image,one_hot_label], batch_size=batch_size,
                                     capacity=10, name='image_batch', allow_smaller_final_batch=True)

        record_count = reader.num_records_produced()
        # to use record reader we need to use a Queue either random

    print('Preprocessing done\n')
    return image_batch,label_batch


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
                    weights = tf.get_variable(config.TF_WEIGHTS_STR,shape=config.TF_ANG_VAR_SHAPES[scope],
                                              initializer=tf.truncated_normal_initializer(stddev=0.02,dtype=tf.float32))
                    bias = tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES[scope][-1],
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
                        h_shape = h.get_shape().as_list()
                        logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s',scope,h_shape)

                        h = tf.reshape(h,[batch_size, h_shape[1] * h_shape[2] * h_shape[3]])
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
        tf.train.exponential_decay(0.002, global_step, decay_steps=250, decay_rate=0.9, staircase=True,
                                   name='learning_rate_decay'), 0.0001)
    if increment_global_step:
        optimize = tf.train.AdamOptimizer(beta1=0.9,learning_rate=learning_rate).minimize(loss,global_step=global_step)
    else:
        optimize = tf.train.AdamOptimizer(beta1=0.9, learning_rate=learning_rate).minimize(loss)
    #optimize = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=learning_rate).minimize(loss)
    return optimize


def accuracy(pred,ohe_labels,use_argmin):
    '''

    :param pred: Prediction vectors
    :param ohe_labels: One-hot encoded actual labels
    :param use_argmin: If true returns the bump accuracy which checks that the predictions have the lowest value where the actual label is zero
    :return:
    '''
    if not use_argmin:
        return np.sum(np.argmax(pred,axis=1)==np.argmax(ohe_labels,axis=1))*100.0/(pred.shape[0]*1.0)
    else:
        return np.sum(np.argmin(pred, axis=1) == np.argmin(ohe_labels, axis=1)) * 100.0 / (pred.shape[0] * 1.0)


def soft_accuracy(pred,ohe_labels, use_argmin):
    if not use_argmin:
        label_indices = list(np.argmax(ohe_labels,axis=1).flatten())
        correct_indices = list(np.where(pred[np.arange(pred.shape[0]),label_indices]>0.51)[0])
        correct_boolean = pred[np.arange(pred.shape[0]),np.argmax(ohe_labels,axis=1).flatten()]>0.51
        correct_boolean_wrt_max = np.argmax(pred,axis=1)==np.argmax(ohe_labels,axis=1)
        return np.sum(np.logical_or(correct_boolean,correct_boolean_wrt_max))*100.0/pred.shape[0]
        #return len(correct_indices)*100.0/pred.shape[0]
    else:
        label_indices = list(np.argmin(ohe_labels, axis=1).flatten())
        correct_indices = list(np.where(pred[np.arange(pred.shape[0]), label_indices] < 0.49)[0])
        correct_boolean = pred[np.arange(pred.shape[0]), np.argmax(ohe_labels, axis=1).flatten()] < 0.49
        correct_boolean_wrt_max = np.argmin(pred, axis=1) == np.argmin(ohe_labels, axis=1)
        return np.sum(np.logical_or(correct_boolean,correct_boolean_wrt_max))*100.0/pred.shape[0]
        #return len(correct_indices) * 100.0 / pred.shape[0]


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

    batch_size = 10
    with sess.as_default() and graph.as_default():
        build_tensorflw_variables()

        global_step = tf.Variable(0,trainable=False)
        tf_images,tf_labels = build_input_pipeline(dataset_filenames['train_dataset'], batch_size,shuffle=True,training_data=True,use_opposite_label=False)
        tf_bump_images, tf_bump_labels = build_input_pipeline(dataset_filenames['train_bump_dataset'], batch_size, shuffle=True, training_data=True, use_opposite_label=True)
        tf_test_images,tf_test_labels = build_input_pipeline(dataset_filenames['test_dataset: '],batch_size,shuffle=False,training_data=False,use_opposite_label=False)
        tf_bump_test_images, tf_bump_test_labels = build_input_pipeline(dataset_filenames['test_bump_dataset'], batch_size, shuffle=False,
                                                              training_data=False,use_opposite_label=True)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf_loss = calculate_loss(tf_images, tf_labels)
        tf_bump_loss = calculate_loss(tf_bump_images,tf_bump_labels)

        tf_optimize = optimize_model(tf_loss,global_step,increment_global_step=True)
        tf_bump_optimize = optimize_model(tf_bump_loss,global_step,increment_global_step=False)

        tf_test_predictions = predictions_with_inputs(tf_test_images)
        tf_bump_test_predictions = predictions_with_inputs(tf_bump_test_images)

        tf.global_variables_initializer().run()

        for epoch in range(250):
            avg_loss = []
            for step in range(50):
                l1, _ = sess.run([tf_loss, tf_optimize])
                avg_loss.append(l1)
            print('\tAverage Loss for Epoch %d: %.5f' %(epoch,np.mean(l1)))

            if (epoch+1)%3==0:
                for step in range(10):
                    sess.run(tf_bump_optimize)

            if (epoch+1)%5==0:
                test_accuracy = []
                soft_test_accuracy = []
                test_image_index = 0
                for step in range(dataset_sizes['test_dataset']//batch_size):
                    predicted_labels,actual_labels = sess.run([tf_test_predictions,tf_test_labels])
                    test_accuracy.append(accuracy(predicted_labels,actual_labels,use_argmin=False))
                    soft_test_accuracy.append(soft_accuracy(predicted_labels,actual_labels,use_argmin=False))

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
                    bump_test_accuracy.append(accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))
                    bump_soft_accuracy.append(soft_accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))

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
