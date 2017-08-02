import numpy as np
import tensorflow as tf
import config
import logging
import sys

sess,graph,logger = None,None,None

logging_level = logging.INFO
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


def lrelu(x, leak=0.01, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def activate(x,activation_type,name='activation'):

    if activation_type=='tanh':
        return tf.nn.tanh(x,name=name)
    elif activation_type=='relu':
        return tf.nn.relu(x,name=name)
    elif activation_type=='lrelu':
        return lrelu(x,name=name)
    else:
        raise NotImplementedError

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
            features = {
                config.FEAT_IMG_ID : tf.FixedLenFeature([], tf.int64),
                config.FEAT_IMG_RAW : tf.FixedLenFeature([], tf.string),
                config.FEAT_LABEL : tf.FixedLenFeature([], tf.int64)
            }
        )

        image = tf.decode_raw(features[config.FEAT_IMG_RAW], tf.float32)
        image = tf.cast(image,tf.float32)
        image = tf.reshape(image,config.TF_INPUT_SIZE,name='reshape_1d_to_3d')
        image.set_shape(config.TF_INPUT_SIZE)

        if config.USE_GRAYSCALE:
            image = tf.image.rgb_to_grayscale(image,name='grayscale_image')
        if training_data:
            image = tf.image.random_brightness(image, 0.5, seed=13345432)
            image = tf.image.random_contrast(image, lower=0.1, upper=1.8, seed=2353252)

        label = tf.cast(features[config.FEAT_LABEL], tf.int32)
        ids = tf.cast(features[config.FEAT_IMG_ID], tf.int32)

        if not use_opposite_label:
            if config.ENABLE_SOFT_CLASSIFICATION:
                one_hot_label = tf.one_hot(label,config.TF_NUM_CLASSES,dtype=tf.float32,on_value=0.5,off_value=0.0)
            else:
                one_hot_label = tf.one_hot(label, config.TF_NUM_CLASSES, dtype=tf.float32, on_value=1.0, off_value=0.0)
        else:
            if config.ENABLE_SOFT_CLASSIFICATION:
                one_hot_label = tf.one_hot(label, config.TF_NUM_CLASSES, dtype=tf.float32, on_value=-0.5, off_value=0.0)
            else:
                one_hot_label = tf.one_hot(label, config.TF_NUM_CLASSES, dtype=tf.float32, on_value=-1.0, off_value=0.0)

        # standardize image
        image = tf.image.per_image_standardization(image)

        if inputs_for_sdae:
            image = tf.image.resize_image_with_crop_or_pad(image, config.TF_RESIZE_TO[0], config.TF_RESIZE_TO[1])
            image = tf.reshape(image,[config.TF_RESIZE_TO[0]*config.TF_RESIZE_TO[1]*config.TF_RESIZE_TO[2]])

        # https://stackoverflow.com/questions/37126108/how-to-read-data-into-tensorflow-batches-from-example-queue

        # The batching mechanism that takes a output produced by reader (with preprocessing) and outputs a batch tensor
        # [batch_size, height, width, depth] 4D tensor
        if not shuffle:
            img_id_batch, image_batch,label_batch = tf.train.batch([ids, image,one_hot_label], batch_size=batch_size,
                                     capacity=10, name='image_batch', allow_smaller_final_batch=True)
        else:
            img_id_batch, image_batch,label_batch = tf.train.shuffle_batch([ids, image,one_hot_label], batch_size=batch_size,
                                     capacity=10, name='image_batch', allow_smaller_final_batch=True,min_after_dequeue=5)
        if training_data:
            flip_rand = tf.random_uniform(shape=[1], minval=0, maxval=1.0, seed=154324654)
            image_batch = tf.cond(flip_rand[0] > 0.5, lambda: tf.reverse(image_batch,axis=[2]), lambda: image_batch)
            label_batch = tf.cond(flip_rand[0] > 0.5, lambda: tf.reverse(label_batch,axis=[1]),lambda: label_batch)

        record_count = reader.num_records_produced()
        # to use record reader we need to use a Queue either random

    print('Preprocessing done\n')
    return img_id_batch, image_batch,label_batch


def get_id_vector_for_correctly_predicted_samples(img_ids, pred, ohe_labels, direction_index, enable_soft_accuracy, use_argmin):

    if not enable_soft_accuracy:
        if not use_argmin:
            label_indices_to_consider = np.where(np.argmax(ohe_labels,axis=1)==direction_index)[0]
            correct_indices_to_consider = np.where(np.argmax(pred[label_indices_to_consider,:],axis=1)==direction_index)[0]
        else:
            label_indices_to_consider = np.where(np.argmin(ohe_labels,axis=1)==direction_index)[0]
            correct_indices_to_consider = \
            np.where(np.argmin(pred[label_indices_to_consider, :], axis=1) == direction_index)[0]
    else:
        if not use_argmin:
            label_indices_to_consider = np.where(np.argmax(ohe_labels,axis=1)==direction_index)[0]
            correct_indices_to_consider = np.where(np.argmax(pred[label_indices_to_consider,:],axis=1)==direction_index)[0]
            pred_indices_above_threshold = np.where(pred[label_indices_to_consider,np.ones([pred.shape[0]],dtype=np.int32)*direction_index]>0.01)[0]
            correct_indices_to_consider = np.union1d(correct_indices_to_consider,pred_indices_above_threshold)
        else:
            label_indices_to_consider = np.where(np.argmin(ohe_labels,axis=1)==direction_index)[0]
            correct_indices_to_consider = \
            np.where(np.argmin(pred[label_indices_to_consider, :], axis=1) == direction_index)[0]
            pred_indices_above_threshold = np.where(pred[label_indices_to_consider,np.ones([pred.shape[0]],dtype=np.int32)*direction_index]<-0.01)[0]
            correct_indices_to_consider = np.union1d(correct_indices_to_consider,pred_indices_above_threshold)

    return list(img_ids[correct_indices_to_consider].flatten())


def get_id_vector_for_predicted_samples(img_ids, pred, ohe_labels, direction_index, enable_soft_accuracy, use_argmin):

    if not enable_soft_accuracy:
        if not use_argmin:
            indices_to_consider = np.where(np.argmax(pred,axis=1)==direction_index)[0]
        else:
            indices_to_consider = np.where(np.argmin(pred,axis=1)==direction_index)[0]

    else:
        if not use_argmin:
            indices_to_consider = np.where(np.argmax(pred,axis=1)==direction_index)[0]
            pred_indices_above_threshold = np.where(pred[np.arange(pred.shape[0]),np.ones([pred.shape[0]],dtype=np.int32)*direction_index]>0.01)[0]
            indices_to_consider = np.union1d(indices_to_consider,pred_indices_above_threshold)
        else:
            indices_to_consider = np.where(np.argmin(pred,axis=1)==direction_index)[0]
            pred_indices_above_threshold = np.where(pred[np.arange(pred.shape[0]),np.ones([pred.shape[0]],dtype=np.int32)*direction_index]<-0.01)[0]
            indices_to_consider = np.union1d(indices_to_consider,pred_indices_above_threshold)

    return list(img_ids[indices_to_consider].flatten())


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
        correct_boolean = pred[np.arange(pred.shape[0]),np.argmax(ohe_labels,axis=1).flatten()] > 0.01
        correct_boolean_wrt_max = np.argmax(pred,axis=1)==np.argmax(ohe_labels,axis=1)
        return np.sum(np.logical_or(correct_boolean,correct_boolean_wrt_max))*100.0/pred.shape[0]
        #return len(correct_indices)*100.0/pred.shape[0]
    else:
        correct_boolean = pred[np.arange(pred.shape[0]), np.argmin(ohe_labels, axis=1).flatten()] < -0.01
        correct_boolean_wrt_min = np.argmin(pred, axis=1) == np.argmin(ohe_labels, axis=1)
        return np.sum(np.logical_or(correct_boolean,correct_boolean_wrt_min))*100.0/pred.shape[0]
        #return len(correct_indices) * 100.0 / pred.shape[0]


def precision_multiclass(pred,ohe_labels, use_argmin):
    #     T POS (Correctly Classified Positive)
    # --------------
    # T POS + F POS (All classfied Positive)
    if not use_argmin:
        logger.debug('Predictions and Labels side by side')
        for ei,(l,p) in enumerate(zip(ohe_labels,pred)):
            if ei<25:
                logger.debug('\t%s;%s',l,p)

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

        #prediction_indices_binned_to_direct = []
        #for di in range(3):
        #    di_greater_than_threshold =
        prediction_indices_binned_to_direct = [ np.where(
            np.logical_or(
                pred[np.arange(pred.shape[0]), np.ones(pred.shape[0],dtype=np.int32)*i] > 0.01,
                np.argmax(pred, axis=1) == np.ones(pred.shape[0],dtype=np.int32)*i
            )==True)[0] for i in range(3)
        ]

        logger.debug('Prediction indices binned by the label (SOFT)')
        logger.debug(prediction_indices_binned_to_direct)
        logger.debug('')
        precision_array = []
        for bi,pred_dir_bin in enumerate(prediction_indices_binned_to_direct):
            logger.debug('True Positive for label (Labels,Predictions,True Pos) %d',bi)
            logger.debug('\tLabels')
            logger.debug('\t%s',label_indices_binned_to_direct[bi])
            logger.debug('\tPredictions')
            logger.debug('\t%s',pred_dir_bin)
            t_pos = np.intersect1d(pred_dir_bin, label_indices_binned_to_direct[bi]).size
            logger.debug('\tIntersection')
            logger.debug('\t%s',np.intersect1d(pred_dir_bin, label_indices_binned_to_direct[bi]))
            logger.debug('\tTrue Positives: %d',t_pos)
            precision_i = t_pos*1.0/max([pred_dir_bin.size,1.0])
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
            pred[np.arange(pred.shape[0]), np.ones(pred.shape[0], dtype=np.int32) * i] < -0.01,
            np.argmin(pred, axis=1) == np.ones(pred.shape[0], dtype=np.int32) * i) == True)[0]
                                               for i in range(3)
                                               ]

        precision_array = []
        for bi, pred_dir_bin in enumerate(prediction_indices_binned_to_direct):
            t_pos = np.intersect1d(pred_dir_bin, label_indices_binned_to_direct[bi]).size
            precision_i = t_pos * 1.0 / max([pred_dir_bin.size,1.0])
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
            pred[np.arange(pred.shape[0]), np.ones(pred.shape[0], dtype=np.int32) * i] > 0.01,
            np.argmax(pred, axis=1) == np.ones(pred.shape[0], dtype=np.int32) * i) == True)[0]
                                               for i in range(3)
                                               ]

        recall_array = []
        for bi, pred_dir_bin in enumerate(prediction_indices_binned_to_direct):
            t_pos = np.intersect1d(pred_dir_bin, label_indices_binned_to_direct[bi]).size
            recall_i = t_pos * 1.0 / max([label_indices_binned_to_direct[bi].size,1.0])
            recall_array.append(recall_i)

        return recall_array

    else:

        label_indices = np.argmin(ohe_labels, axis=1).flatten()
        # list with each item being an array corresponding to a single direction
        # where array items are the indices of that direction
        label_indices_binned_to_direct = [np.where(label_indices == i)[0] for i in range(3)]
        prediction_indices_binned_to_direct = [np.where(np.logical_or(
            pred[np.arange(pred.shape[0]), np.ones(pred.shape[0], dtype=np.int32) * i] < -0.01,
            np.argmin(pred, axis=1) == np.ones(pred.shape[0], dtype=np.int32) * i) == True)[0]
                                               for i in range(3)
                                               ]

        recall_array = []
        for bi, pred_dir_bin in enumerate(prediction_indices_binned_to_direct):
            t_pos = np.intersect1d(pred_dir_bin, label_indices_binned_to_direct[bi]).size
            recall_i = t_pos * 1.0 / max([label_indices_binned_to_direct[bi].size,1.0])
            recall_array.append(recall_i)

        return recall_array