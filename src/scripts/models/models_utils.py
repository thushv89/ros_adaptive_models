import numpy as np
import tensorflow as tf
import config
import logging
import sys

sess,graph,logger = None,None,None

logging_level = logging.DEBUG
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

logger = logging.getLogger('ModelsUtilsLogger')
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter(logging_format))
console.setLevel(logging_level)
fileHandler = logging.FileHandler('models_utils.log', mode='w')
fileHandler.setFormatter(logging.Formatter(logging_format))
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.addHandler(fileHandler)

def set_from_main(main_sess,main_graph, main_logger):
    global sess,graph,logger
    sess = main_sess
    graph = main_graph
    #logger = main_logger

def build_input_pipeline(filenames, batch_size, shuffle, training_data, use_opposite_label, inputs_for_sdae):
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
        image = tf.reshape(image,config.TF_INPUT_SIZE,name='reshape_1d_to_3d')
        image.set_shape(config.TF_INPUT_SIZE)

        if config.USE_GRAYSCALE:
            image = tf.image.rgb_to_grayscale(image,name='grayscale_image')
        if training_data:
            image = tf.image.random_brightness(image, 0.5, seed=13345432)
            image = tf.image.random_contrast(image, 0, 0.5, seed=2353252)

        label = tf.cast(features[config.FEAT_LABEL], tf.int32)
        if not use_opposite_label:
            if training_data and config.ENABLE_SOFT_CLASSIFICATION:
                one_hot_label = tf.one_hot(label,config.TF_NUM_CLASSES,dtype=tf.float32,on_value=1.0,off_value=config.SOFT_COLLISION_LABEL)
            else:
                one_hot_label = tf.one_hot(label,config.TF_NUM_CLASSES,dtype=tf.float32)
        else:
            if training_data and config.ENABLE_SOFT_CLASSIFICATION:
                one_hot_label = tf.one_hot(label,config.TF_NUM_CLASSES,dtype=tf.float32,on_value=0.0, off_value=1.0)
            else:
                one_hot_label = tf.one_hot(label, config.TF_NUM_CLASSES, dtype=tf.float32, on_value=0.0, off_value=1.0)

        # standardize image
        image = tf.image.per_image_standardization(image)

        if inputs_for_sdae:
            image = tf.image.resize_image_with_crop_or_pad(image, config.TF_RESIZE_TO[0], config.TF_RESIZE_TO[1])
            image = tf.reshape(image,[config.TF_RESIZE_TO[0]*config.TF_RESIZE_TO[1]*config.TF_RESIZE_TO[2]])

        # https://stackoverflow.com/questions/37126108/how-to-read-data-into-tensorflow-batches-from-example-queue

        # The batching mechanism that takes a output produced by reader (with preprocessing) and outputs a batch tensor
        # [batch_size, height, width, depth] 4D tensor
        if not shuffle:
            image_batch,label_batch = tf.train.batch([image,one_hot_label], batch_size=batch_size,
                                     capacity=10, name='image_batch', allow_smaller_final_batch=True)
        else:
            image_batch,label_batch = tf.train.shuffle_batch([image,one_hot_label], batch_size=batch_size,
                                     capacity=10, name='image_batch', allow_smaller_final_batch=True,min_after_dequeue=5)
        if training_data:
            flip_rand = tf.random_uniform(shape=[1],minval=0, maxval=1.0, seed=154324654)
            image_batch = tf.cond(flip_rand[0] > 0.5, lambda: tf.reverse(image_batch,axis=[1]), lambda: image_batch)
            label_batch = tf.cond(flip_rand[0] > 0.5, lambda: tf.reverse(label_batch,axis=[1]),lambda: label_batch)
        record_count = reader.num_records_produced()
        # to use record reader we need to use a Queue either random

    print('Preprocessing done\n')
    return image_batch,label_batch


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
        correct_boolean = pred[np.arange(pred.shape[0]),np.argmax(ohe_labels,axis=1).flatten()]>0.5
        correct_boolean_wrt_max = np.argmax(pred,axis=1)==np.argmax(ohe_labels,axis=1)
        return np.sum(np.logical_or(correct_boolean,correct_boolean_wrt_max))*100.0/pred.shape[0]
        #return len(correct_indices)*100.0/pred.shape[0]
    else:
        label_indices = list(np.argmin(ohe_labels, axis=1).flatten())
        correct_indices = list(np.where(pred[np.arange(pred.shape[0]), label_indices] < 0.49)[0])
        correct_boolean = pred[np.arange(pred.shape[0]), np.argmax(ohe_labels, axis=1).flatten()] < 0.5
        correct_boolean_wrt_max = np.argmin(pred, axis=1) == np.argmin(ohe_labels, axis=1)
        return np.sum(np.logical_or(correct_boolean,correct_boolean_wrt_max))*100.0/pred.shape[0]
        #return len(correct_indices) * 100.0 / pred.shape[0]


def precision_multiclass(pred,ohe_labels, use_argmin):
    #     T POS (Correctly Classified Positive)
    # --------------
    # T POS + F POS (All classfied Positive)
    if not use_argmin:
        logger.debug('Predictions')
        logger.debug(pred)
        logger.debug('')
        logger.debug('OHE labels')
        logger.debug(ohe_labels)
        logger.debug('')

        label_indices = np.argmax(ohe_labels, axis=1).flatten()
        logger.debug('Label indices')
        logger.debug(label_indices)
        logger.debug('')
        # list with each item being an array corresponding to a single direction
        # where array items are the indices of that direction
        label_indices_binned_to_direct = [np.where(label_indices == i)[0] for i in range(3)]
        logger.debug('Label indices binned by the label')
        logger.debug(label_indices_binned_to_direct)
        logger.debug('')
        prediction_indices_binned_to_direct = [ np.where(
            np.logical_or(
                pred[np.arange(pred.shape[0]), np.ones(pred.shape[0],dtype=np.int32)*i] > 0.51,
                np.argmax(pred, axis=1) == np.ones(pred.shape[0],dtype=np.int32)*i)==True)[0]
            for i in range(3)
        ]
        logger.debug('Prediction indices binned by the label (SOFT)')
        logger.debug(prediction_indices_binned_to_direct)
        logger.debug('')
        precision_array = []
        for bi,pred_dir_bin in enumerate(prediction_indices_binned_to_direct):
            logger.debug('True Positive for label %d',bi)
            t_pos = np.intersect1d(pred_dir_bin, label_indices_binned_to_direct[bi]).size
            logger.debug('\t%d',t_pos)
            precision_i = t_pos*1.0/np.asscalar(np.sum([pred_i.size for pred_i in prediction_indices_binned_to_direct]))
            precision_array.append(precision_i)
        logger.debug('')
        logger.debug('Precision array')
        logger.debug(precision_array)
        logger.debug('')
        return precision_array

    else:

        label_indices = np.argmin(ohe_labels, axis=1).flatten()
        # list with each item being an array corresponding to a single direction
        # where array items are the indices of that direction
        label_indices_binned_to_direct = [np.where(label_indices == i)[0] for i in range(3)]
        prediction_indices_binned_to_direct = [np.where(np.logical_or(
            pred[np.arange(pred.shape[0]), np.ones(pred.shape[0], dtype=np.int32) * i] < 0.49,
            np.argmin(pred, axis=1) == np.ones(pred.shape[0], dtype=np.int32) * i) == True)[0]
                                               for i in range(3)
                                               ]

        precision_array = []
        for bi, pred_dir_bin in enumerate(prediction_indices_binned_to_direct):
            t_pos = np.intersect1d(pred_dir_bin, label_indices_binned_to_direct[bi]).size
            precision_i = t_pos * 1.0 / np.asscalar(np.sum([pred_i.size for pred_i in prediction_indices_binned_to_direct]))
            precision_array.append(precision_i)

        return precision_array


def recall_multiclass(pred, ohe_labels, use_argmin):
    #      T POS   (Correctly classified postives)
    # ----------------
    #  T POS + F NEG (Total actual positives)
    if not use_argmin:
        label_indices = np.argmax(ohe_labels, axis=1).flatten()
        # list with each item being an array corresponding to a single direction
        # where array items are the indices of that direction
        label_indices_binned_to_direct = [np.where(label_indices == i)[0] for i in range(3)]
        prediction_indices_binned_to_direct = [np.where(np.logical_or(
            pred[np.arange(pred.shape[0]), np.ones(pred.shape[0], dtype=np.int32) * i] > 0.51,
            np.argmax(pred, axis=1) == np.ones(pred.shape[0], dtype=np.int32) * i) == True)[0]
                                               for i in range(3)
                                               ]

        recall_array = []
        for bi, pred_dir_bin in enumerate(prediction_indices_binned_to_direct):
            t_pos = np.intersect1d(pred_dir_bin, label_indices_binned_to_direct[bi]).size
            recall_i = t_pos * 1.0 / np.asscalar(np.sum([actual_i.size for actual_i in label_indices_binned_to_direct]))
            recall_array.append(recall_i)

        return recall_array

    else:

        label_indices = np.argmin(ohe_labels, axis=1).flatten()
        # list with each item being an array corresponding to a single direction
        # where array items are the indices of that direction
        label_indices_binned_to_direct = [np.where(label_indices == i)[0] for i in range(3)]
        prediction_indices_binned_to_direct = [np.where(np.logical_or(
            pred[np.arange(pred.shape[0]), np.ones(pred.shape[0], dtype=np.int32) * i] < 0.49,
            np.argmin(pred, axis=1) == np.ones(pred.shape[0], dtype=np.int32) * i) == True)[0]
                                               for i in range(3)
                                               ]

        recall_array = []
        for bi, pred_dir_bin in enumerate(prediction_indices_binned_to_direct):
            t_pos = np.intersect1d(pred_dir_bin, label_indices_binned_to_direct[bi]).size
            recall_i = t_pos * 1.0 / np.asscalar(np.sum([actual_i.size for actual_i in label_indices_binned_to_direct]))
            recall_array.append(recall_i)

        return recall_array