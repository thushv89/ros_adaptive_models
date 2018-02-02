__author__ = 'Thushan Ganegedara'

import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import os
from math import ceil, floor
import logging
import sys
import ada_cnn_qlearner
from ada_cnn_data_pool import Pool
from collections import Counter
from scipy.misc import imsave
import getopt
import time
import ada_cnn_utils as utils
import queue
from multiprocessing import Pool as MPPool
import copy
import ada_cnn_constants as constants
import ada_cnn_cnn_hyperparameters_getter as cnn_hyperparameters_getter
import ada_cnn_cnn_optimizer as cnn_optimizer
import ada_cnn_cnn_intializer as cnn_intializer
import ada_cnn_adapter
import data_generator
import h5py
import dataset_name_factory

logging_level = logging.INFO
logging_format = '[%(funcName)s] %(message)s'

dataset_dir,output_dir = None,None
adapt_structure = False
rigid_pooling = False
rigid_naive = False

act_decay = 0.99

interval_parameters, research_parameters, model_hyperparameters, dataset_info = None, None, None, None
image_size, num_channels,resize_to = None,None,None
original_image_size = None

n_epochs, n_iterations, iterations_per_batch, num_labels, train_size, test_size, n_slices, data_fluctuation = None,None,None,None,None,None,None,None
cnn_string, filter_vector = None,None

batch_size = None
start_lr, decay_learning_rate, decay_rate, decay_steps = None,None,None,None
beta, include_l2_loss = None,None
use_dropout, in_dropout_rate, dropout_rate, current_adaptive_dropout = None, None, None,None
pool_size = None

# Constant Strings
TF_WEIGHTS = constants.TF_WEIGHTS
TF_BIAS = constants.TF_BIAS
TF_ACTIVAIONS_STR = constants.TF_ACTIVAIONS_STR
TF_FC_WEIGHT_IN_STR = constants.TF_FC_WEIGHT_IN_STR
TF_FC_WEIGHT_OUT_STR = constants.TF_FC_WEIGHT_OUT_STR
TF_LOSS_VEC_STR = constants.TF_LOSS_VEC_STR
TF_GLOBAL_SCOPE = constants.TF_GLOBAL_SCOPE
TOWER_NAME = constants.TOWER_NAME
TF_ADAPTATION_NAME_SCOPE = constants.TF_ADAPTATION_NAME_SCOPE
TF_SCOPE_DIVIDER = constants.TF_SCOPE_DIVIDER


start_eps = None
eps_decay = None
valid_acc_decay = None

n_tasks = None
prune_min_bound, prune_max_bound = None, None

def set_varialbes_with_input_arguments(adapt_structure, use_rigid_pooling):
    global interval_parameters, model_hyperparameters, research_parameters, dataset_info, cnn_string, filter_vector
    global image_size, num_channels, resize_to,original_image_size
    global n_epochs, n_iterations, iterations_per_batch, num_labels, train_size, test_size, n_slices, data_fluctuation
    global start_lr, decay_learning_rate, decay_rate, decay_steps
    global batch_size, beta, include_l2_loss
    global use_dropout, in_dropout_rate, dropout_rate, current_adaptive_dropout
    global pool_size
    global start_eps,eps_decay,valid_acc_decay
    global n_tasks, prune_min_bound, prune_max_bound

    # Data specific parameters
    dataset_info = cnn_hyperparameters_getter.get_data_specific_hyperparameters()
    image_size = dataset_info['image_size']
    resize_to = dataset_info['resize_to']
    num_channels = dataset_info['n_channels']
    original_image_size = dataset_info['original_image_size']

    # interval parameters
    interval_parameters = cnn_hyperparameters_getter.get_interval_related_hyperparameters()

    # Research parameters
    research_parameters = cnn_hyperparameters_getter.get_research_hyperparameters(adapt_structure, use_rigid_pooling, logging_level)

    # Model Hyperparameters
    model_hyperparameters = cnn_hyperparameters_getter.get_model_specific_hyperparameters(adapt_structure, rigid_pooling, use_fse_capacity, dataset_info['n_labels'])

    n_epochs = model_hyperparameters['epochs']

    iterations_per_batch = model_hyperparameters['iterations_per_batch']

    num_labels = dataset_info['n_labels']

    cnn_string = model_hyperparameters['cnn_string']
    if adapt_structure:
        filter_vector = model_hyperparameters['filter_vector']

    start_lr = model_hyperparameters['start_lr']
    decay_learning_rate = model_hyperparameters['decay_learning_rate']
    decay_rate = model_hyperparameters['decay_rate']
    decay_steps = 1

    batch_size = model_hyperparameters['batch_size']
    beta = model_hyperparameters['beta']
    include_l2_loss = model_hyperparameters['include_l2_loss']

    # Dropout
    use_dropout = model_hyperparameters['use_dropout']
    in_dropout_rate = model_hyperparameters['in_dropout_rate']
    dropout_rate = model_hyperparameters['dropout_rate']
    current_adaptive_dropout = dropout_rate


    # pool parameters
    if adapt_structure or use_rigid_pooling:
        pool_size = model_hyperparameters['pool_size']

    if adapt_structure:
        start_eps = model_hyperparameters['start_eps']
        eps_decay = model_hyperparameters['eps_decay']
    valid_acc_decay = model_hyperparameters['validation_set_accumulation_decay']

    # Tasks
    n_tasks = model_hyperparameters['n_tasks']


cnn_ops, cnn_hyperparameters = None, None

state_action_history = []

cnn_ops, cnn_hyperparameters = None, None


num_gpus = -1

# Tensorflow Op / Variable related Python variables
# Optimizer Related
optimizer, custom_lr = None,None
tf_learning_rate = None
# Optimizer (Data) Related
tf_avg_grad_and_vars, apply_grads_op, concat_loss_vec_op, \
update_train_velocity_op, mean_loss_op = None,None,None,None,None

# Optimizer (Pool) Related
tf_pool_avg_gradvars, apply_pool_grads_op, update_pool_velocity_ops, mean_pool_loss = None, None, None, None

# Data related
tf_train_data_batch, tf_train_label_batch, tf_data_weights = [], [], []
tf_test_dataset, tf_test_labels = None, None
tf_valid_data_batch, tf_valid_label_batch = None, None

# Data (Pool) related
tf_pool_data_batch, tf_pool_label_batch = [], []
pool_pred, augmented_pool_data_batch, augmented_pool_label_batch = None, [],[]

# Logit related
tf_grads, tf_loss_vec, tf_loss, tf_pred = None, None, None, None
tf_pool_grad, tf_pool_loss = None, None
tf_logits = None
tf_dropout_rate = None

# Test/Valid related
valid_loss_op,valid_predictions_op, test_predicitons_op = None,None,None

# Adaptation related
tf_slice_optimize, tf_training_slice_optimize_top, tf_training_slice_vel_update_top, \
tf_training_slice_optimize_bottom, tf_training_slice_vel_update_bottom = {},{},{},{},{}
tf_slice_vel_update, tf_training_slice_vel_update = {},{}
tf_pool_slice_optimize_bottom, tf_pool_slice_vel_update_bottom = None, None
tf_pool_slice_optimize_top, tf_pool_slice_vel_update_top = None, None

tf_add_filters_ops, tf_rm_filters_ops, tf_replace_ind_ops = {}, {}, {}
tf_indices, tf_indices_size, tf_replicative_factor_vec = None,None,None
tf_update_hyp_ops = {}
tf_action_info = None
tf_weights_this,tf_bias_this = None, None
tf_weights_next,tf_wvelocity_this, tf_bvelocity_this, tf_wvelocity_next = None, None, None, None
tf_weight_shape,tf_in_size, tf_out_size = None, None, None
increment_global_step_op = None
tf_weight_mean_ops = None
tf_retain_id_placeholders = {}

tf_act_this = None

adapt_period = None

# Loggers
logger = None
perf_logger = None

# Reset CNN
tf_reset_cnn, tf_reset_cnn_custom = None, None


tf_scale_parameter = None

conv_dropout_placeholder_dict = {}


def define_conv_dropout_placeholder():
    global conv_dropout_placeholder_dict,cnn_ops

    for op in cnn_ops:
        if 'conv' in op:
            conv_dropout_placeholder_dict[op] = tf.placeholder(dtype=tf.float32,shape=[None],name='dropout_'+op)


def get_dropout_placeholder_dict():
    global conv_dropout_placeholder_dict,cnn_hyperparameters, dropout_rate

    placeholder_feed_dict = {}
    for scope in cnn_ops:
        if 'conv' in scope:
            binom_vec = np.random.binomial(1, 1.0 - dropout_rate,
                                   cnn_hyperparameters[scope]['weights'][-1])/(1.0 - dropout_rate)
            placeholder_feed_dict[conv_dropout_placeholder_dict[scope]] = binom_vec

    return placeholder_feed_dict


def inference(dataset, tf_cnn_hyperparameters, training):
    global logger,cnn_ops, act_decay

    first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'

    last_conv_id = ''
    for op in cnn_ops:
        if 'conv' in op:
            last_conv_id = op

    logger.debug('Defining the logit calculation ...')
    logger.debug('\tCurrent set of operations: %s' % cnn_ops)
    activation_ops = []

    x = dataset

    if training and use_dropout:
        x = tf.nn.dropout(x, keep_prob=1.0 - in_dropout_rate, name='input_dropped')

    logger.info('\tReceived data for X(%s)...' % x.get_shape().as_list())

    # need to calculate the output according to the layers we have
    for op in cnn_ops:
        if 'conv' in op:
            with tf.variable_scope(op, reuse=True) as scope:
                logger.info('\tConvolving (%s) With Weights:%s Stride:%s' % (
                    op, cnn_hyperparameters[op]['weights'], cnn_hyperparameters[op]['stride']))
                logger.debug('\t\tWeights: %s', tf.shape(tf.get_variable(TF_WEIGHTS)).eval())
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

                x = tf.nn.conv2d(x, w, cnn_hyperparameters[op]['stride'],
                                 padding=cnn_hyperparameters[op]['padding'])

                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                                     tf.assign(tf.get_variable(TF_ACTIVAIONS_STR),
                                               act_decay * tf.get_variable(TF_ACTIVAIONS_STR) + (1-act_decay)* tf.reduce_mean(tf.abs(w),axis=[0,1,2])))
                x = utils.lrelu(x + b, name=scope.name + '/top')

                if training and use_dropout:
                    x = x * tf.reshape(conv_dropout_placeholder_dict[op], [1, 1, 1, -1])

        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s' % (
                op, cnn_hyperparameters[op]['kernel'], cnn_hyperparameters[op]['stride']))
            if cnn_hyperparameters[op]['type'] is 'max':
                x = tf.nn.max_pool(x, ksize=cnn_hyperparameters[op]['kernel'],
                                   strides=cnn_hyperparameters[op]['stride'],
                                   padding=cnn_hyperparameters[op]['padding'])
            elif cnn_hyperparameters[op]['type'] is 'avg':
                x = tf.nn.avg_pool(x, ksize=cnn_hyperparameters[op]['kernel'],
                                   strides=cnn_hyperparameters[op]['stride'],
                                   padding=cnn_hyperparameters[op]['padding'])
            if training and use_dropout:
                x = tf.nn.dropout(x, keep_prob=1.0 - tf_dropout_rate, name='dropout')

        if 'fulcon' in op:
            with tf.variable_scope(op, reuse=True) as scope:

                if first_fc == op:
                    # we need to reshape the output of last subsampling layer to
                    # convert 4D output to a 2D input to the hidden layer
                    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]

                    logger.debug('Input size of fulcon_out : %d', cnn_hyperparameters[op]['in'])
                    # Transpose x (b,h,w,d) to (b,d,w,h)
                    # This help us to do adaptations more easily
                    x = tf.transpose(x, [0, 3, 1, 2])
                    x = tf.reshape(x, [batch_size, tf_cnn_hyperparameters[op][TF_FC_WEIGHT_IN_STR]])

                    h_list = {}
                    for di in ['left','straight','right']:
                        with tf.variable_scope(di,reuse=True):
                            w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
                            x = utils.lrelu(x, name=scope.name + '/top')

                            if training and use_dropout:
                                x = tf.nn.dropout(x, keep_prob=1.0 - tf_dropout_rate, name='dropout')

                            h_list[di] = tf.matmul(x, w) + b

                elif 'fulcon_out' == op:
                    h_out_list = []
                    for di in ['left', 'straight', 'right']:
                        with tf.variable_scope(di, reuse=True):
                            w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
                            h_out_list.append(tf.matmul(h_list[di], w) + b)

                    x = tf.squeeze(tf.stack(h_out_list, axis=1))

                else:
                    raise NotImplementedError

    return x


def get_weights_mean_for_pruning():
    weight_mean_ops = {}

    for op in cnn_ops:
        if 'conv' in op:
            with tf.variable_scope(op, reuse=True) as scope:
                w, b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)
                weight_mean_ops[op] = tf.assign(tf.get_variable(TF_ACTIVAIONS_STR), tf.reduce_mean(w, [0, 1, 2]),
                              validate_shape=False)

    return weight_mean_ops


def tower_loss(dataset, labels, weighted, tf_data_weights, tf_cnn_hyperparameters):
    global cnn_ops, batch_size

    rand_mask = tf.cast(tf.greater(tf.random_normal([batch_size, 3], dtype=tf.float32), 0.0), dtype=tf.float32)
    label_mask = labels + (tf.cast(tf.equal(labels, 0.0), dtype=tf.float32) * rand_mask)

    tf_logits = inference(dataset, tf_cnn_hyperparameters, True)

    loss = tf.reduce_mean(tf.reduce_sum((tf.nn.softmax(tf_logits) * label_mask - labels) ** 2, axis=[1]), axis=[0])

    if include_l2_loss:

        l2_weights = []
        for op in cnn_ops:

            if 'conv' in op:
                with tf.variable_scope(op):
                    l2_weights.append(tf.get_variable(TF_WEIGHTS))
            elif 'fulcon' in op:
                with tf.variable_scope(op):
                    for di in ['left','straight','right']:
                        with tf.variable_scope(di):
                            l2_weights.append(tf.get_variable(TF_WEIGHTS))

        loss = tf.reduce_sum([loss, beta * tf.reduce_sum([tf.nn.l2_loss(w) for w in l2_weights])])

    total_loss = loss

    return total_loss


def calc_loss_vector(dataset, labels, tf_cnn_hyperparameters):
    logits = inference(dataset, tf_cnn_hyperparameters, True)
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name=TF_LOSS_VEC_STR)


def average_gradients(tower_grads):
    # tower_grads => [((grads0gpu0,var0gpu0),...,(grads0gpuN,var0gpuN)),((grads1gpu0,var1gpu0),...,(grads1gpuN,var1gpuN))]
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, v in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, axis=0)
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def concat_loss_vector_towers(tower_loss_vectors):
    concat_loss_vec = None
    for loss_vec in tower_loss_vectors:
        if concat_loss_vec is None:
            concat_loss_vec = tf.identity(loss_vec)
        else:
            concat_loss_vec = tf.concat(axis=1, values=loss_vec)

    return concat_loss_vec


def mean_tower_activations(tower_activations):
    mean_activations = []
    if len(tower_activations)>1:
        for act_towers in zip(*tower_activations):
            stacked_activations = None
            for a in act_towers:
                if stacked_activations is None:
                    stacked_activations = tf.identity(a)
                else:
                    stacked_activations = tf.stack([stacked_activations, a], axis=0)

            mean_activations.append(tf.reduce_mean(stacked_activations, [0]))
    else:
        for act_single_tower in tower_activations[0]:
            mean_activations.append(act_single_tower)

    return mean_activations


def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction


def predict_with_dataset(dataset, tf_cnn_hyperparameters):
    logits = inference(dataset, tf_cnn_hyperparameters, False)
    prediction = tf.nn.softmax(logits)
    return prediction


def accuracy(predictions, labels):
    assert predictions.shape[0] == labels.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def setup_loggers(adapt_structure):
    '''
    Setting up loggers
    logger: Main Loggerrun
    perf_logger: Logging time
    hyp_logger: Log hyperparameters
    :param adapt_structure:
    :return:
    '''


    main_logger = logging.getLogger('main_ada_cnn_logger')
    main_logger.setLevel(logging_level)
    main_logger.propagate=False
    # File handler for writing to file
    main_file_handler = logging.FileHandler(output_dir + os.sep + 'ada_cnn_main.log', mode='w')
    main_file_handler.setLevel(logging.DEBUG)
    main_file_handler.setFormatter(logging.Formatter('%(message)s'))
    main_logger.addHandler(main_file_handler)
    # Console handler for writing to console
    main_console = logging.StreamHandler(sys.stdout)
    main_console.setFormatter(logging.Formatter(logging_format))
    #main_console.setLevel(logging_level)
    main_logger.addHandler(main_console)

    error_logger = logging.getLogger('error_logger')
    error_logger.propagate = False
    error_logger.setLevel(logging.INFO)
    errHandler = logging.FileHandler(output_dir + os.sep + 'Error.log', mode='w')
    errHandler.setFormatter(logging.Formatter('%(message)s'))
    error_logger.addHandler(errHandler)
    error_logger.info('#Batch_ID,Loss(Train),Train Accuracy, Valid(Unseen),Test Accuracy')

    perf_logger = logging.getLogger('time_logger')
    perf_logger.propagate = False
    perf_logger.setLevel(logging.INFO)
    perf_handler = logging.FileHandler(output_dir + os.sep + 'time.log', mode='w')
    perf_handler.setFormatter(logging.Formatter('%(message)s'))
    perf_logger.addHandler(perf_handler)
    perf_logger.info('#Batch_ID,Time(Full),Time(Train),Op count, Var count')

    hyp_logger = logging.getLogger('hyperparameter_logger')
    hyp_logger.propagate = False
    hyp_logger.setLevel(logging.INFO)
    hyp_handler = logging.FileHandler(output_dir + os.sep + 'Hyperparameter.log', mode='w')
    hyp_handler.setFormatter(logging.Formatter('%(message)s'))
    hyp_logger.addHandler(hyp_handler)

    cnn_structure_logger, q_logger, prune_logger = None, None, None

    if adapt_structure:
        cnn_structure_logger = logging.getLogger('cnn_structure_logger')
        main_logger.propagate = False
        cnn_structure_logger.setLevel(logging.INFO)
        structHandler = logging.FileHandler(output_dir + os.sep + 'cnn_structure.log', mode='w')
        structHandler.setFormatter(logging.Formatter('%(message)s'))
        cnn_structure_logger.addHandler(structHandler)
        cnn_structure_logger.info('#batch_id:state:action:reward:#layer_1_hyperparameters#layer_2_hyperparameters#...')

        q_logger = logging.getLogger('q_eval_rand_logger')
        main_logger.propagate = False
        q_logger.setLevel(logging.INFO)
        q_handler = logging.FileHandler(output_dir + os.sep + 'QMetric.log', mode='w')
        q_handler.setFormatter(logging.Formatter('%(message)s'))
        q_logger.addHandler(q_handler)
        q_logger.info('#batch_id,q_metric')

        prune_logger = logging.getLogger('prune_reward_logger')
        prune_logger.propagate = False
        prune_logger.setLevel(logging.INFO)
        prune_handler = logging.FileHandler(output_dir + os.sep + 'PruneRewardExperience.log', mode='w')
        prune_handler.setFormatter(logging.Formatter('%(message)s'))
        prune_logger.addHandler(prune_handler)
        prune_logger.info('#task_id,prune_factor,acc_after, acc_before, acc_gain,reward, infer_type')

    class_dist_logger = logging.getLogger('class_dist_logger')
    class_dist_logger.propagate = False
    class_dist_logger.setLevel(logging.INFO)
    class_dist_handler = logging.FileHandler(output_dir + os.sep + 'class_distribution.log', mode='w')
    class_dist_handler.setFormatter(logging.Formatter('%(message)s'))
    class_dist_logger.addHandler(class_dist_handler)

    pool_dist_logger = logging.getLogger('pool_distribution_logger')
    pool_dist_logger.propagate = False
    pool_dist_logger.setLevel(logging.INFO)
    pool_handler = logging.FileHandler(output_dir + os.sep + 'pool_valid_distribution.log', mode='w')
    pool_handler.setFormatter(logging.Formatter('%(message)s'))
    pool_dist_logger.addHandler(pool_handler)
    pool_dist_logger.info('#Class distribution')

    pool_ft_dist_logger = logging.getLogger('pool_distribution_logger')
    pool_ft_dist_logger.propagate = False
    pool_ft_dist_logger.setLevel(logging.INFO)
    pool_ft_handler = logging.FileHandler(output_dir + os.sep + 'pool_ft_distribution.log', mode='w')
    pool_ft_handler.setFormatter(logging.Formatter('%(message)s'))
    pool_ft_dist_logger.addHandler(pool_ft_handler)
    pool_ft_dist_logger.info('#Class distribution')

    return main_logger, perf_logger, \
           cnn_structure_logger, q_logger, class_dist_logger, \
           pool_dist_logger, pool_ft_dist_logger, hyp_logger, error_logger, prune_logger

def setup_activation_vector_logger():
    global cnn_ops
    act_vector_logger_dict = {}
    if not os.path.exists(output_dir + os.sep + 'activations'):
        os.mkdir(output_dir + os.sep + 'activations')

    for op in cnn_ops:
        act_vector_logger_dict[op] = logging.getLogger('activation_logger_' + op)
        act_vector_logger_dict[op].propagate = False
        act_vector_logger_dict[op].setLevel(logging.INFO)
        handler = logging.FileHandler(output_dir + os.sep + "activations" + os.sep + 'activations_ ' + op + '.log',
                                      mode='w')
        handler.setFormatter(logging.Formatter('%(message)s'))
        act_vector_logger_dict[op].addHandler(handler)

    return act_vector_logger_dict


def get_activation_dictionary(activation_list, cnn_ops, conv_op_ids):
    current_activations = {}
    for act_i, layer_act in enumerate(activation_list):
        current_activations[cnn_ops[conv_op_ids[act_i]]] = layer_act
    return current_activations


def define_tf_ops(global_step, tf_cnn_hyperparameters, init_cnn_hyperparameters):
    global optimizer
    global tf_train_data_batch, tf_train_label_batch, tf_data_weights, tf_reset_cnn, tf_reset_cnn_custom
    global tf_test_dataset,tf_test_labels
    global tf_pool_data_batch, tf_pool_label_batch
    global tf_grads, tf_loss_vec, tf_loss, tf_pred
    global tf_pool_grad, tf_pool_loss, tf_logits,tf_dropout_rate
    global tf_add_filters_ops, tf_rm_filters_ops, tf_replace_ind_ops
    global tf_slice_optimize, tf_slice_vel_update
    global tf_training_slice_optimize_top, tf_training_slice_vel_update_top, tf_training_slice_optimize_bottom, tf_training_slice_vel_update_bottom
    global tf_pool_slice_optimize_top, tf_pool_slice_vel_update_top, tf_pool_slice_optimize_bottom, tf_pool_slice_vel_update_bottom
    global tf_indices, tf_indices_size, tf_replicative_factor_vec, tf_weight_mean_ops, tf_retain_id_placeholders
    global tf_avg_grad_and_vars, apply_grads_op, concat_loss_vec_op, update_train_velocity_op, mean_loss_op
    global tf_pool_avg_gradvars, apply_pool_grads_op, update_pool_velocity_ops, mean_pool_loss
    global valid_loss_op,valid_predictions_op, test_predicitons_op
    global tf_valid_data_batch,tf_valid_label_batch
    global pool_pred, augmented_pool_data_batch, augmented_pool_label_batch
    global tf_update_hyp_ops, tf_action_info
    global tf_weights_this,tf_bias_this, tf_weights_next,tf_wvelocity_this, tf_bvelocity_this, tf_wvelocity_next
    global tf_weight_shape,tf_in_size, tf_out_size
    global increment_global_step_op,tf_learning_rate
    global tf_act_this
    global tf_scale_parameter
    global logger
    global train_data_gen, test_data_gen

    define_conv_dropout_placeholder()

    _, tf_train_data_batch, tf_train_label_batch = train_data_gen.tf_augment_data_with()
    _, tf_test_dataset, tf_test_labels = test_data_gen.tf_augment_data_with()

    # custom momentum optimizing we calculate momentum manually
    logger.info('Defining Optimizer')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
    increment_global_step_op = tf.assign(global_step, global_step + 1)
    tf_learning_rate = tf.train.exponential_decay(start_lr, global_step, decay_steps=decay_steps,
                               decay_rate=decay_rate, staircase=True)

    tf_dropout_rate = tf.Variable(dropout_rate,trainable=False,dtype=tf.float32,name='tf_dropout_rate')
    # Test data (Global)
    logger.info('Defining Test data placeholders')

    #tf_test_dataset = tf.placeholder(tf.float32, shape=[batch_size] + image_size + [num_channels],
    #                                     name='TestDataset')
    #tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='TestLabels')

    # Tower-Like Calculations
    # Calculate logits (train and pool),
    # Calculate gradients (train and pool)
    # Tower_grads will contain
    # [[(grad0gpu0,var0gpu0),...,(gradNgpu0,varNgpu0)],...,[(grad0gpuD,var0gpuD),...,(gradNgpuD,varNgpuD)]]

    logger.info('Defining TF operations for GPU')

    # Input train data
    logger.info('\tDefning Training Data placeholders and weights')

    #tf_train_data_batch = tf.placeholder(
    #    tf.float32, shape=[batch_size] + image_size + [num_channels],
    #    name='TrainDataset'
    #)

    #tf_train_label_batch =\
    #    tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='TrainLabels')

    tf_data_weights = tf.placeholder(tf.float32, shape=(batch_size), name='TrainWeights')

    # Training data opearations
    logger.info('\tDefining logit operations')
    tf_logits = inference(tf_train_data_batch, tf_cnn_hyperparameters, True)

    logger.info('\tDefine Loss for each tower')
    tf_loss = tower_loss(tf_train_data_batch, tf_train_label_batch, True,
                               tf_data_weights, tf_cnn_hyperparameters)

    tf_loss_vec = calc_loss_vector(tf_train_data_batch, tf_train_label_batch,
                                         tf_cnn_hyperparameters)

    logger.info('\tGradient calculation opeartions for tower')
    tf_grads = cnn_optimizer.gradients(
        optimizer, tf_loss, global_step,
        tf.constant(start_lr, dtype=tf.float32)
    )

    logger.info('\tPrediction operations for tower')
    tf_pred = predict_with_dataset(tf_train_data_batch, tf_cnn_hyperparameters)

    # Pooling data operations
    logger.info('\tPool related operations')

    tf_pool_data_batch = tf.placeholder(
        tf.float32, shape=[batch_size] + image_size + [num_channels],
        name='PoolDataset'
    )

    tf_pool_label_batch =\
        tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='PoolLabels')

    # Used for augmenting the pool data

    augmented_pool_data_batch = tf_augment_data_with(tf_pool_data_batch)
    augmented_pool_label_batch = tf_pool_label_batch

    with tf.name_scope('pool') as scope:
        #single_pool_logit_op = inference(tf_pool_data_batch[-1],tf_cnn_hyperparameters, True)

        tf_pool_loss = tower_loss(augmented_pool_data_batch, augmented_pool_label_batch, False, None,
                                      tf_cnn_hyperparameters)
        tf_pool_grad = cnn_optimizer.gradients(optimizer, tf_pool_loss, global_step, start_lr)

    logger.info('GLOBAL_VARIABLES (all)')
    logger.info('\t%s\n', [v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])

    logger.info('Tower averaging for Gradients for Training data')

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        apply_grads_op, update_train_velocity_op = cnn_optimizer.apply_gradient_with_rmsprop(optimizer, start_lr, global_step, tf_grads)

    logger.info('Tower averaging for Gradients for Pool data')
    # Pool data operations
    apply_pool_grads_op, update_pool_velocity_ops = cnn_optimizer.apply_pool_gradient_with_rmsprop(optimizer, start_lr, global_step, tf_pool_grad)

    # Weight mean calculation for pruning
    tf_weight_mean_ops = get_weights_mean_for_pruning()

    increment_global_step_op = tf.assign(global_step, global_step + 1)

    # GLOBAL: Tensorflow operations for hard_pool
    with tf.name_scope('pool') as scope:
        tf.get_variable_scope().reuse_variables()
        pool_pred = predict_with_dataset(tf_pool_data_batch, tf_cnn_hyperparameters)

    # GLOBAL: Tensorflow operations for test data
    # Valid data (Next train batch) Unseen
    logger.info('Validation data placeholders, losses and predictions')

    test_predicitons_op = predict_with_dataset(tf_test_dataset, tf_cnn_hyperparameters)

    # GLOBAL: Structure adaptation
    with tf.name_scope(TF_ADAPTATION_NAME_SCOPE):
        if research_parameters['adapt_structure']:
            # Tensorflow operations that are defined one for each convolution operation
            tf_indices = tf.placeholder(dtype=tf.int32, shape=(None,), name='optimize_indices')
            tf_indices_size = tf.placeholder(tf.int32)

            tf_action_info = tf.placeholder(shape=[3], dtype=tf.int32,
                                            name='tf_action')  # [op_id,action_id,amount] (action_id 0 - add, 1 -remove)

            logger.info('Defining placeholders for newly added parameters')
            # Adaptation placeholders for convolution layers
            tf_weights_this = tf.placeholder(shape=[None, None, None, None], dtype=tf.float32,
                                             name='new_weights_current')
            tf_bias_this = tf.placeholder(shape=(None,), dtype=tf.float32, name='new_bias_current')

            tf_act_this = tf.placeholder(shape=(None,), dtype=tf.float32, name='new_activations_current')

            tf_weights_next = tf.placeholder(shape=[None, None, None, None], dtype=tf.float32,
                                             name='new_weights_next')

            tf_wvelocity_this = tf.placeholder(shape=[None, None, None, None], dtype=tf.float32,
                                               name='new_weights_velocity_current')
            tf_bvelocity_this = tf.placeholder(shape=(None,), dtype=tf.float32,
                                               name='new_bias_velocity_current')
            tf_wvelocity_next = tf.placeholder(shape=[None, None, None, None], dtype=tf.float32,
                                               name='new_weights_velocity_next')

            logger.info('Defining weight shapes for setting shapes after adaptations')
            tf_weight_shape = tf.placeholder(shape=[4], dtype=tf.int32, name='weight_shape')
            tf_in_size = tf.placeholder(dtype=tf.int32, name='fulcon_input_size')
            tf_out_size = tf.placeholder(dtype=tf.int32, name='fulcon_output_size')
            tf_replicative_factor_vec = tf.placeholder(dtype=tf.float32, shape=[None], name='tf_replicative_factor')

            tf_scale_parameter = tf.placeholder(shape=[None], dtype=tf.float32, name='grad_scal_factor')

            tf_reset_cnn = cnn_intializer.reset_cnn(
                init_cnn_hyperparameters,cnn_ops
            )

            for op in cnn_ops:
                if 'pool' in op:
                    continue
                else:
                    tf_retain_id_placeholders[op] = {'in':None, 'out':None}
                    tf_retain_id_placeholders[op]['in'] = tf.placeholder(dtype=tf.int32,shape=[None],name='retain_id_placeholder_in_'+op)
                    tf_retain_id_placeholders[op]['out'] = tf.placeholder(dtype=tf.int32, shape=[None],
                                                                        name='retain_id_placeholder_out_' + op)

            #Partial Logits and Grads (TOP and BOTTOM)
            partial_bottom_train_loss = tower_loss(tf_train_data_batch, tf_train_label_batch, True,
                                                tf_data_weights, tf_cnn_hyperparameters)
            partial_bottom_grads = cnn_optimizer.gradients(optimizer, partial_bottom_train_loss, global_step, start_lr)

            tf_training_slice_optimize_bottom, tf_training_slice_vel_update_bottom = \
                cnn_optimizer.apply_gradient_with_rmsprop(optimizer, start_lr, global_step, partial_bottom_grads)

            partial_bottom_pool_loss = tower_loss(tf_pool_data_batch, tf_pool_label_batch, False,
                                                   None, tf_cnn_hyperparameters)
            partial_pool_bottom_grads = cnn_optimizer.gradients(optimizer, partial_bottom_pool_loss, global_step,
                                                           start_lr)

            tf_pool_slice_optimize_bottom, tf_pool_slice_vel_update_bottom = \
                cnn_optimizer.apply_pool_gradient_with_rmsprop(optimizer, start_lr, global_step, partial_pool_bottom_grads)

            for tmp_op in cnn_ops:
                # Convolution related adaptation operations
                if 'conv' in tmp_op:
                    tf_update_hyp_ops[tmp_op] = ada_cnn_adapter.update_tf_hyperparameters(tmp_op, tf_weight_shape, tf_in_size, tf_out_size)
                    tf_add_filters_ops[tmp_op] = ada_cnn_adapter.add_with_action(tmp_op, tf_action_info, tf_weights_this,
                                                                 tf_bias_this, tf_weights_next,
                                                                 tf_wvelocity_this, tf_bvelocity_this,
                                                                 tf_wvelocity_next,tf_replicative_factor_vec, tf_act_this)


def check_several_conditions_with_assert(num_gpus):
    batches_in_chunk = model_hyperparameters['chunk_size']//model_hyperparameters['batch_size']
    assert batches_in_chunk % num_gpus == 0
    assert num_gpus > 0


def distort_img(img):
    if np.random.random()<0.4:
        img = np.fliplr(img)
    if np.random.random()<0.4:
        brightness = np.random.random()*1.5 - 0.6
        img += brightness
    if np.random.random()<0.4:
        contrast = np.random.random()*0.8 + 0.4
        img *= contrast

    return img

def tf_augment_data_with(tf_pool_batch):


    tf_aug_pool_batch = tf.map_fn(lambda data: tf.image.random_flip_left_right(data), tf_pool_batch)

    tf_aug_pool_batch = tf.image.random_brightness(tf_aug_pool_batch,0.5)
    tf_aug_pool_batch = tf.image.random_contrast(tf_aug_pool_batch,0.5,1.5)

    # Not necessary they are already normalized
    #tf_image_batch = tf.map_fn(lambda img: tf.image.per_image_standardization(img), tf_image_batch)

    return tf_aug_pool_batch


def get_pool_valid_accuracy(hard_pool_valid):
    global pool_dataset, pool_labels, pool_pred
    tmp_pool_accuracy = []
    pool_dataset, pool_labels = hard_pool_valid.get_pool_data(False)
    for pool_id in range(0,(hard_pool_valid.get_size() // batch_size)):
        pbatch_data = pool_dataset[pool_id * batch_size:(pool_id + 1) * batch_size, :, :, :]
        pbatch_labels = pool_labels[pool_id * batch_size:(pool_id + 1) * batch_size, :]
        pool_feed_dict = {tf_pool_data_batch: pbatch_data,
                          tf_pool_label_batch: pbatch_labels}

        p_predictions = session.run(pool_pred, feed_dict=pool_feed_dict)
        tmp_pool_accuracy.append(accuracy(p_predictions, pbatch_labels))

    return np.mean(tmp_pool_accuracy)


def get_adaptive_dropout():
    global dropout_rate
    return dropout_rate

    global cnn_hyperparameters,cnn_ops, dropout_rate,filter_vector
    current_conv_depth_total, current_fulcon_depth_total = 0, 0
    conv_filter_vec, fulcon_filter_vec = [],[]
    is_fulcon_exist = False
    for op_i, op in enumerate(cnn_ops):
        if 'conv' in op:
            conv_filter_vec.append(filter_vector[op_i])
        elif 'fulcon_out'!= op and 'fulcon' in op:
            fulcon_filter_vec.append(filter_vector[op_i])
            is_fulcon_exist = True

    for scope in cnn_ops:
        if 'conv' in scope:
            current_conv_depth_total += cnn_hyperparameters[scope]['weights'][3]
        if scope!='fulcon_out' and 'fulcon' in scope:
            current_fulcon_depth_total += cnn_hyperparameters[scope]['out']

    if is_fulcon_exist:
        dropout_factor =  ((current_conv_depth_total*1.0/sum(conv_filter_vec)) + (current_fulcon_depth_total*1.0/sum(fulcon_filter_vec)))/2.0
    else:
        dropout_factor = (current_conv_depth_total * 1.0 / sum(conv_filter_vec))

    return dropout_rate*(dropout_factor**2)



def get_new_distorted_weights(new_curr_weights,curr_weight_shape):
    if np.random.random() < 0.25:
        new_curr_weights = np.flip(new_curr_weights, axis=np.random.choice([0, 1]))
    if np.random.random() < 0.25:
        new_curr_weights = np.swapaxes(new_curr_weights, 0, 1)
    if np.random.random() < 0.8:
        if np.random.random() < 0.5:
            translate_amout = np.random.choice([1, 2, 3])
            new_curr_weights = np.pad(new_curr_weights, ((translate_amout, 0), (0, 0), (0, 0), (0, 0)), 'mean')
            new_curr_weights = new_curr_weights[:curr_weight_shape[0], :, :, :]
        if np.random.random() < 0.5:
            translate_amout = np.random.choice([1, 2, 3])
            new_curr_weights = np.pad(new_curr_weights, ((0, translate_amout), (0, 0), (0, 0), (0, 0)), 'mean')
            new_curr_weights = new_curr_weights[translate_amout:, :, :, :]
        if np.random.random() < 0.5:
            translate_amout = np.random.choice([1, 2, 3])
            new_curr_weights = np.pad(new_curr_weights, ((0, 0), (translate_amout, 0), (0, 0), (0, 0)), 'mean')
            new_curr_weights = new_curr_weights[:, :curr_weight_shape[1], :, :]
        if np.random.random() < 0.5:
            translate_amout = np.random.choice([1, 2, 3])
            new_curr_weights = np.pad(new_curr_weights, ((0, 0), (0, translate_amout), (0, 0), (0, 0)), 'mean')
            new_curr_weights = new_curr_weights[:, translate_amout:, :, :]

    return new_curr_weights


def run_actual_add_operation(session, current_op, li, last_conv_id, hard_pool_ft, epoch):
    '''
    Run the add operation using the given Session
    :param session:
    :param current_op:
    :param li:
    :param last_conv_id:
    :param hard_pool_ft:
    :return:
    '''
    global current_adaptive_dropout
    amount_to_add = ai[1]
    scale_for_rand = 0.001

    min_rand_threshold = 0.01
    max_rand_threshold = 0.75

    #if epoch==-1:
    #rand_thresh_for_layer = max([min_rand_threshold, max_rand_threshold * (1.0 / (li+1.0)**0.5)])
    #else:
    rand_thresh_for_layer = 0.5

    print('Rand thresh for layer: ',rand_thresh_for_layer)
    if current_op != last_conv_id:
        next_conv_op = \
            [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][0]

    with tf.variable_scope(TF_GLOBAL_SCOPE, reuse=True):

        with tf.variable_scope(current_op,reuse=True):
            curr_weights = tf.get_variable(TF_WEIGHTS).eval()
            curr_bias = tf.get_variable(TF_BIAS).eval()
            curr_act = tf.get_variable(TF_ACTIVAIONS_STR).eval()

        if current_op != last_conv_id:
            with tf.variable_scope(next_conv_op, reuse=True):
                next_weights = tf.get_variable(TF_WEIGHTS).eval()
        else:
            with tf.variable_scope(first_fc, reuse=True):
                next_weights = tf.get_variable(TF_WEIGHTS).eval()

        #Net2Net type initialization
        if np.random.random()<rand_thresh_for_layer:
            print('Net2Net Initialization')

            curr_weight_shape = curr_weights.shape
            next_weights_shape = next_weights.shape

            #rand_indices_1 = np.random.choice(np.arange(curr_weights.shape[3]).tolist(),size=amount_to_add,replace=True)

            if np.random.random()<0.5:
                rand_indices_1 = np.argsort(curr_act).ravel()[-amount_to_add:]
                rand_indices_2 = np.argsort(curr_act).ravel()[:amount_to_add]
            else:
                rand_indices_1 = np.random.choice(np.arange(curr_weights.shape[3]).tolist(), size=amount_to_add,
                                                  replace=True)
                rand_indices_2 = np.random.choice(np.arange(curr_weights.shape[3]).tolist(), size=amount_to_add,
                                                  replace=True)

            #all_indices_plus_rand = np.concatenate([np.arange(0,curr_weights.shape[3]).ravel(), np.asarray(rand_indices_1).ravel()])
            #ind_counter = Counter(all_indices_plus_rand.tolist())
            #sorted_keys = sorted(ind_counter.keys())
            normalize_factor = 1.0*(curr_weight_shape[3] + amount_to_add) / curr_weight_shape[3]
            #print(normalize_factor)
            #count_vec = np.asarray([ind_counter[k] for k in sorted_keys ])
            #count_vec = np.concatenate([count_vec,count_vec[rand_indices_1]])#*normalize_factor
            count_vec = np.ones((curr_weight_shape[3] + amount_to_add), dtype=np.float32)  #* normalize_factor

            print('count vec',count_vec.shape)
            print(count_vec)
            new_curr_weights = (curr_weights[:,:,:,rand_indices_1] + curr_weights[:,:,:,rand_indices_2])/2.0
            new_curr_weights = get_new_distorted_weights(new_curr_weights,curr_weight_shape)
            new_act_this = (curr_act[rand_indices_1] + curr_act[rand_indices_2])/2.0

            new_curr_bias = np.random.normal(scale=scale_for_rand, size=(amount_to_add))

            if last_conv_id != current_op:
                new_next_weights = (next_weights[:,:,rand_indices_1,:] + next_weights[:,:,rand_indices_2,:])/2.0
                new_next_weights = get_new_distorted_weights(new_next_weights,next_weights_shape)

                #new_next_weights = next_weights[:, :, rand_indices, :]
            else:
                low_bound_1 = (rand_indices_1*final_2d_height*final_2d_width).tolist()
                upper_bound_1 = ((rand_indices_1+1) * final_2d_height * final_2d_width).tolist()
                low_up_bounds_1 = list(zip(low_bound_1,upper_bound_1))
                all_indices_1 = [np.arange(l,u) for (l,u) in low_up_bounds_1]
                all_indices_1 = np.stack(all_indices_1).ravel()

                low_bound_2 = (rand_indices_2 * final_2d_height * final_2d_width).tolist()
                upper_bound_2 = ((rand_indices_2 + 1) * final_2d_height * final_2d_width).tolist()
                low_up_bounds_2 = list(zip(low_bound_2, upper_bound_2))
                all_indices_2 = [np.arange(l, u) for (l, u) in low_up_bounds_2]
                all_indices_2 = np.stack(all_indices_2).ravel()

                new_next_weights = (next_weights[all_indices_1,:] + next_weights[all_indices_2,:])/2.0
                new_next_weights = np.expand_dims(np.expand_dims(new_next_weights,-1),-1)

        # Random initialization
        else:
            print('Random Initialization')
            curr_weight_shape = curr_weights.shape
            next_weights_shape = next_weights.shape

            new_curr_weights = np.random.normal(
                scale=np.sqrt(2.0 / (curr_weight_shape[0] * curr_weight_shape[1] * curr_weight_shape[2])),
                size=(curr_weight_shape[0], curr_weight_shape[1], curr_weight_shape[2], amount_to_add))
            new_curr_bias = np.random.normal(scale=scale_for_rand, size=(amount_to_add))

            new_act_this = np.zeros(shape=(amount_to_add),dtype=np.float32)

            if last_conv_id != current_op:
                new_next_weights = np.random.normal(scale=np.sqrt(0.1/(next_weights_shape[0]*next_weights_shape[1]*next_weights_shape[2])),
                                                     size=(next_weights_shape[0],next_weights_shape[1],amount_to_add, next_weights_shape[3]))
            else:
                new_next_weights = np.random.normal(scale=np.sqrt(0.1/next_weights_shape[0]),
                                                     size=(amount_to_add *final_2d_height *final_2d_width, next_weights_shape[1],1,1))

            normalize_factor = 1.0*(curr_weight_shape[3] + amount_to_add) / curr_weight_shape[3]
            count_vec = np.ones((curr_weight_shape[3] + amount_to_add), dtype=np.float32) #* normalize_factor

    _ = session.run(tf_add_filters_ops[current_op],
                    feed_dict={
                        tf_action_info: np.asarray([li, 1, ai[1]]),
                        tf_weights_this: new_curr_weights,
                        tf_bias_this: new_curr_bias,

                        tf_weights_next: new_next_weights,
                        tf_replicative_factor_vec: count_vec,
                        tf_wvelocity_this: np.zeros(shape=(
                            cnn_hyperparameters[current_op]['weights'][0],
                            cnn_hyperparameters[current_op]['weights'][1],
                            cnn_hyperparameters[current_op]['weights'][2], amount_to_add),dtype=np.float32),
                        tf_bvelocity_this: np.zeros(shape=(amount_to_add,),dtype=np.float32),
                        tf_wvelocity_next: np.zeros(shape=(
                            cnn_hyperparameters[next_conv_op]['weights'][0],
                            cnn_hyperparameters[next_conv_op]['weights'][1],
                            amount_to_add, cnn_hyperparameters[next_conv_op]['weights'][3]),dtype=np.float32) if last_conv_id != current_op else
                        np.zeros(shape=(final_2d_height * final_2d_width * amount_to_add,
                                        cnn_hyperparameters[first_fc]['out'], 1, 1),dtype=np.float32),
                        tf_act_this:new_act_this
                    })

    # change both weights and biase in the current op
    logger.debug('\tAdding %d new weights', amount_to_add)

    with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + current_op,
                           reuse=True) as scope:
        current_op_weights = tf.get_variable(TF_WEIGHTS)

    if research_parameters['debugging']:
        logger.debug('\tSummary of changes to weights of %s ...', current_op)
        logger.debug('\t\tNew Weights: %s', str(tf.shape(current_op_weights).eval()))

    # change out hyperparameter of op
    cnn_hyperparameters[current_op]['weights'][3] += amount_to_add
    if research_parameters['debugging']:
        assert cnn_hyperparameters[current_op]['weights'][2] == \
               tf.shape(current_op_weights).eval()[2]

    session.run(tf_update_hyp_ops[current_op], feed_dict={
        tf_weight_shape: cnn_hyperparameters[current_op]['weights']
    })

    if current_op == last_conv_id:

        with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + first_fc, reuse=True):
            first_fc_weights = tf.get_variable(TF_WEIGHTS)
        cnn_hyperparameters[first_fc]['in'] += final_2d_height * final_2d_width * amount_to_add

        if research_parameters['debugging']:
            logger.debug('\tNew %s in: %d', first_fc, cnn_hyperparameters[first_fc]['in'])
            logger.debug('\tSummary of changes to weights of %s', first_fc)
            logger.debug('\t\tNew Weights: %s', str(tf.shape(first_fc_weights).eval()))

        session.run(tf_update_hyp_ops[first_fc], feed_dict={
            tf_in_size: cnn_hyperparameters[first_fc]['in'],
            tf_out_size: cnn_hyperparameters[first_fc]['out']
        })

    else:

        next_conv_op = \
            [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][
                0]
        assert current_op != next_conv_op

        with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + next_conv_op, reuse=True):
            next_conv_op_weights = tf.get_variable(TF_WEIGHTS)

        if research_parameters['debugging']:
            logger.debug('\tSummary of changes to weights of %s', next_conv_op)
            logger.debug('\t\tCurrent Weights: %s', str(tf.shape(next_conv_op_weights).eval()))

        cnn_hyperparameters[next_conv_op]['weights'][2] += amount_to_add

        if research_parameters['debugging']:
            assert cnn_hyperparameters[next_conv_op]['weights'][2] == \
                   tf.shape(next_conv_op_weights).eval()[2]

        session.run(tf_update_hyp_ops[next_conv_op], feed_dict={
            tf_weight_shape: cnn_hyperparameters[next_conv_op]['weights']
        })

    # optimize the newly added fiterls only
    pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)

    logger.info('\t(Before) Size of Rolling mean vector for %s: %s', current_op,
                rolling_ativation_means[current_op].shape)

    rolling_ativation_means[current_op] = np.append(rolling_ativation_means[current_op],
                                                    np.zeros(ai[1]))

    logger.info('\tSize of Rolling mean vector for %s: %s', current_op,
                rolling_ativation_means[current_op].shape)


    current_adaptive_dropout = get_adaptive_dropout()
    # This is a pretty important step
    # Unless you run this onces, the sizes of weights do not change
    train_feed_dict.update({tf_dropout_rate:current_adaptive_dropout})
    _ = session.run([tf_logits], feed_dict=train_feed_dict)

    # Train only with half of the batch
    for pool_id in range(0, (hard_pool_ft.get_size() // batch_size) - 1):
        if np.random.random() < research_parameters['finetune_rate']:
            pbatch_data, pbatch_labels = [], []

            pool_feed_dict = {
                tf_indices: np.arange(cnn_hyperparameters[current_op]['weights'][3] - ai[1],
                                      cnn_hyperparameters[current_op]['weights'][3])}
            pool_feed_dict.update({tf_dropout_rate:0.0})

            pbatch_data.append(pool_dataset[pool_id * batch_size:(pool_id + 1) * batch_size,
                               :, :, :])
            pbatch_labels.append(pool_labels[(pool_id) * batch_size:(pool_id + 1) * batch_size,
                                 :])
            pool_feed_dict.update({tf_pool_data_batch: pbatch_data,
                                   tf_pool_label_batch: pbatch_labels})

            _,_ = session.run([tf_pool_slice_optimize_bottom, tf_pool_slice_vel_update_bottom],feed_dict=pool_feed_dict)

    '''train_feed_dict[tf_dropout_rate] = 0.0
    train_feed_dict.update({tf_indices: np.arange(cnn_hyperparameters[current_op]['weights'][3] - ai[1],
                                                  cnn_hyperparameters[current_op]['weights'][3])})

    _, _ = session.run([tf_training_slice_optimize[current_op], tf_training_slice_vel_update[current_op]],
                       feed_dict=train_feed_dict)'''

    run_actual_finetune_operation(hard_pool_ft)


def run_actual_add_operation_for_fulcon(session, current_op, li, last_conv_id, hard_pool_ft, epoch):
    '''
    Run the add operation using the given Session
    :param session:
    :param current_op:
    :param li:
    :param last_conv_id:
    :param hard_pool_ft:
    :return:
    '''
    global current_adaptive_dropout
    amount_to_add = ai[1]

    scale_for_rand = 0.001

    min_rand_threshold = 0.01
    max_rand_threshold = 0.75

    #if epoch==-1:
    #rand_thresh_for_layer = max([min_rand_threshold, max_rand_threshold * (1.0 / (li+1.0)**0.5)])
    #else:
    rand_thresh_for_layer = 0.5

    print('Rand thresh for layer: ', rand_thresh_for_layer)

    if current_op != last_conv_id:
        next_fulcon_op = \
            [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:]][0]


    with tf.variable_scope(TF_GLOBAL_SCOPE, reuse=True):

        with tf.variable_scope(current_op,reuse=True):
            curr_weights = tf.get_variable(TF_WEIGHTS).eval()
            curr_bias = tf.get_variable(TF_BIAS).eval()
            curr_act = tf.get_variable(TF_ACTIVAIONS_STR).eval()

        with tf.variable_scope(next_fulcon_op, reuse=True):
            next_weights = tf.get_variable(TF_WEIGHTS).eval()

        # Net2Net Initialization
        if np.random.random()<rand_thresh_for_layer:

            print('Net2Net Initialization')
            curr_weight_shape = curr_weights.shape
            next_weights_shape = next_weights.shape

            #rand_indices_1 = np.random.choice(np.arange(curr_weights.shape[1]).tolist(),size=amount_to_add,replace=True)
            #rand_indices_2 = np.random.choice(np.arange(curr_weights.shape[1]).tolist(), size=amount_to_add, replace=True)
            if np.random.random()<0.5:
                rand_indices_1 = np.random.choice(np.arange(curr_weights.shape[1]).tolist(), size=amount_to_add,
                                                  replace=True)
                rand_indices_2 = np.random.choice(np.arange(curr_weights.shape[1]).tolist(), size=amount_to_add,
                                                  replace=True)
            else:
                rand_indices_1 = np.argsort(curr_act).ravel()[-amount_to_add:]
                rand_indices_2 = np.argsort(curr_act).ravel()[:amount_to_add]

            #all_indices_plus_rand = np.concatenate([np.arange(0,curr_weights.shape[1]).ravel(), np.asarray(rand_indices_1).ravel()])
            #print('allindices plus rand',all_indices_plus_rand.shape)
            #ind_counter = Counter(all_indices_plus_rand.tolist())
            #sorted_keys = np.asarray(sorted(ind_counter.keys()))
            #count_vec = np.asarray([ind_counter[k] for k in sorted_keys])

            normalize_factor = 1.0*(curr_weight_shape[1] + amount_to_add) / curr_weight_shape[1]
            print(normalize_factor)
            #count_vec = np.concatenate([count_vec,count_vec[rand_indices_1]])#*normalize_factor
            count_vec = np.ones((curr_weight_shape[1] + amount_to_add), dtype=np.float32)  #*normalize_factor
            print('count vec',count_vec.shape)
            print(count_vec)

            new_curr_weights = (curr_weights[:, rand_indices_1] + curr_weights[:, rand_indices_2])/2.0
            new_curr_weights = np.expand_dims(np.expand_dims(new_curr_weights,-1),-1)
            new_curr_act = (curr_act[rand_indices_1] + curr_act[rand_indices_2])/2.0

            new_curr_bias = np.random.normal(scale=scale_for_rand, size=(amount_to_add))

            new_next_weights = (next_weights[rand_indices_1,:] + next_weights[rand_indices_2,:])/2.0
            new_next_weights = np.expand_dims(np.expand_dims(new_next_weights,-1),-1)

        else:
            print('Random Initialization')
            curr_weight_shape = curr_weights.shape
            next_weights_shape = next_weights.shape
            new_curr_weights = np.random.normal(scale=np.sqrt(0.1/curr_weight_shape[0]),size=(curr_weight_shape[0],amount_to_add,1,1))
            new_curr_bias = np.random.normal(scale=scale_for_rand, size=(amount_to_add))
            new_curr_act = np.zeros(shape=(amount_to_add),dtype=np.float32)

            new_next_weights = np.random.normal(scale=np.sqrt(0.1/next_weights_shape[0]),size=(amount_to_add,next_weights_shape[1],1,1))

            normalize_factor = (curr_weight_shape[1] + amount_to_add) / curr_weight_shape[1]
            count_vec = np.ones((curr_weight_shape[1]+amount_to_add),dtype=np.float32) #* normalize_factor

    _ = session.run(tf_add_filters_ops[current_op],
                    feed_dict={
                        tf_action_info: np.asarray([li, 1, ai[1]]),
                        tf_weights_this: new_curr_weights,
                        tf_bias_this: new_curr_bias,
                        tf_replicative_factor_vec: count_vec,
                        tf_weights_next: new_next_weights,
                        tf_wvelocity_this: np.zeros(shape=(
                            cnn_hyperparameters[current_op]['in'],
                            amount_to_add,1,1),dtype=np.float32),
                        tf_bvelocity_this: np.zeros(shape=(amount_to_add),dtype=np.float32),
                        tf_wvelocity_next: np.zeros(shape=(
                            amount_to_add,
                            cnn_hyperparameters[next_fulcon_op]['out'],1,1),dtype=np.float32),
                        tf_act_this:new_curr_act
                    })

    # change both weights and biase in the current op
    logger.debug('\tAdding %d new weights', amount_to_add)

    with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + current_op,
                           reuse=True) as scope:
        current_op_weights = tf.get_variable(TF_WEIGHTS)

    if research_parameters['debugging']:
        logger.debug('\tSummary of changes to weights of %s ...', current_op)
        logger.debug('\t\tNew Weights: %s', str(tf.shape(current_op_weights).eval()))

    # change out hyperparameter of op
    cnn_hyperparameters[current_op]['out'] += amount_to_add
    if research_parameters['debugging']:
        assert cnn_hyperparameters[current_op]['in'] == \
               tf.shape(current_op_weights).eval()[1]

    session.run(tf_update_hyp_ops[current_op], feed_dict={
        tf_in_size: cnn_hyperparameters[current_op]['in'],
        tf_out_size: cnn_hyperparameters[current_op]['out']
    })

    next_fulcon_op = cnn_ops[cnn_ops.index(current_op) + 1]
    assert current_op != next_fulcon_op

    with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + next_fulcon_op, reuse=True):
        next_fulcon_op_weights = tf.get_variable(TF_WEIGHTS)

    if research_parameters['debugging']:
        logger.debug('\tSummary of changes to weights of %s', next_fulcon_op)
        logger.debug('\t\tCurrent Weights: %s', str(tf.shape(next_fulcon_op_weights).eval()))

    cnn_hyperparameters[next_fulcon_op]['in'] += amount_to_add

    if research_parameters['debugging']:
        assert cnn_hyperparameters[next_fulcon_op]['in'] == \
               tf.shape(next_fulcon_op_weights).eval()[0]

    session.run(tf_update_hyp_ops[next_fulcon_op], feed_dict={
        tf_in_size: cnn_hyperparameters[next_fulcon_op]['in'],
        tf_out_size: cnn_hyperparameters[next_fulcon_op]['out']
    })

    # optimize the newly added fiterls only
    pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)

    current_adaptive_dropout = get_adaptive_dropout()
    # This is a pretty important step
    # Unless you run this onces, the sizes of weights do not change
    train_feed_dict.update({tf_dropout_rate:current_adaptive_dropout})
    _ = session.run([tf_logits], feed_dict=train_feed_dict)
    pbatch_train_count = 0

    # Train only newly added parameters
    for pool_id in range(0, (hard_pool_ft.get_size() // batch_size) - 1):
        if np.random.random() < research_parameters['finetune_rate']:
            pbatch_data, pbatch_labels = [], []

            pool_feed_dict = {
                tf_indices: np.arange(cnn_hyperparameters[current_op]['out'] - ai[1],
                                      cnn_hyperparameters[current_op]['out'])}
            pool_feed_dict.update({tf_dropout_rate:0.0})


            pbatch_data.append(pool_dataset[(pool_id) * batch_size:(pool_id + 1) * batch_size,
                               :, :, :])
            pbatch_labels.append(pool_labels[(pool_id) * batch_size:(pool_id + 1) * batch_size,
                                 :])
            pool_feed_dict.update({tf_pool_data_batch: pbatch_data,
                                   tf_pool_label_batch: pbatch_labels})


            _,_ = session.run([tf_pool_slice_optimize_bottom,tf_pool_slice_vel_update_bottom],feed_dict=pool_feed_dict)


    '''train_feed_dict[tf_dropout_rate] = 0.0
    train_feed_dict.update({tf_indices: np.arange(cnn_hyperparameters[current_op]['out'] - ai[1],
                                                  cnn_hyperparameters[current_op]['out'])})

    _, _ = session.run([tf_training_slice_optimize[current_op], tf_training_slice_vel_update[current_op]], feed_dict=train_feed_dict)'''

    # Optimize full network
    run_actual_finetune_operation(hard_pool_ft)


def run_actual_remove_operation(session, current_op, li, last_conv_id, hard_pool_ft):
    global current_adaptive_dropout
    _, rm_indices = session.run(tf_rm_filters_ops[current_op],
                                feed_dict={
                                    tf_action_info: np.asarray([li, 0, ai[1]]),
                                    tf_running_activations: rolling_ativation_means[current_op]
                                })
    rm_indices = rm_indices.flatten()
    amount_to_rmv = ai[1]

    with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + current_op, reuse=True):
        current_op_weights = tf.get_variable(TF_WEIGHTS)

    if research_parameters['remove_filters_by'] == 'Activation':
        logger.debug('\tRemoving filters for op %s', current_op)
        logger.debug('\t\t\tIndices: %s', rm_indices[:10])

    elif research_parameters['remove_filters_by'] == 'Distance':
        logger.debug('\tRemoving filters for op %s', current_op)

        logger.debug('\t\tSimilarity summary')
        logger.debug('\t\t\tIndices: %s', rm_indices[:10])

        logger.debug('\t\tSize of indices to remove: %s/%d', rm_indices.size,
                     cnn_hyperparameters[current_op]['weights'][3])
        indices_of_filters_keep = list(
            set(np.arange(cnn_hyperparameters[current_op]['weights'][3])) - set(
                rm_indices.tolist()))
        logger.debug('\t\tSize of indices to keep: %s/%d', len(indices_of_filters_keep),
                     cnn_hyperparameters[current_op]['weights'][3])

    cnn_hyperparameters[current_op]['weights'][3] -= amount_to_rmv
    if research_parameters['debugging']:
        logger.debug('\tSize after feature map reduction: %s,%s', current_op,
                     tf.shape(current_op_weights).eval())
        assert tf.shape(current_op_weights).eval()[3] == \
               cnn_hyperparameters[current_op]['weights'][3]

    session.run(tf_update_hyp_ops[current_op], feed_dict={
        tf_weight_shape: cnn_hyperparameters[current_op]['weights']
    })

    if current_op == last_conv_id:
        with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + first_fc, reuse=True):
            first_fc_weights = tf.get_variable(TF_WEIGHTS)

        cnn_hyperparameters[first_fc]['in'] -= final_2d_height * final_2d_width * amount_to_rmv
        if research_parameters['debugging']:
            logger.debug('\tSize after feature map reduction: %s,%s',
                         first_fc, str(tf.shape(first_fc_weights).eval()))

        session.run(tf_update_hyp_ops[first_fc], feed_dict={
            tf_in_size: cnn_hyperparameters[first_fc]['in'],
            tf_out_size: cnn_hyperparameters[first_fc]['out']
        })

    else:
        next_conv_op = \
            [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][0]
        assert current_op != next_conv_op

        with tf.variable_scope(TF_GLOBAL_SCOPE + TF_SCOPE_DIVIDER + next_conv_op, reuse=True):
            next_conv_op_weights = tf.get_variable(TF_WEIGHTS)

        cnn_hyperparameters[next_conv_op]['weights'][2] -= amount_to_rmv

        if research_parameters['debugging']:
            logger.debug('\tSize after feature map reduction: %s,%s', next_conv_op,
                         str(tf.shape(next_conv_op_weights).eval()))
            assert tf.shape(next_conv_op_weights).eval()[2] == \
                   cnn_hyperparameters[next_conv_op]['weights'][2]

        session.run(tf_update_hyp_ops[next_conv_op], feed_dict={
            tf_weight_shape: cnn_hyperparameters[next_conv_op]['weights']
        })

    logger.info('\t(Before) Size of Rolling mean vector for %s: %s', current_op,
                rolling_ativation_means[current_op].shape)
    rolling_ativation_means[current_op] = np.delete(rolling_ativation_means[current_op],
                                                    rm_indices)
    logger.info('\tSize of Rolling mean vector for %s: %s', current_op,
                rolling_ativation_means[current_op].shape)

    current_adaptive_dropout = get_adaptive_dropout()
    # This is a pretty important step
    # Unless you run this onces, the sizes of weights do not change
    _ = session.run([tf_logits], feed_dict=train_feed_dict)

    if hard_pool_ft.get_size() > batch_size:
        pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)

        # Train with latter half of the data
        for pool_id in range(0, (hard_pool_ft.get_size() // batch_size) - 1):
            if np.random.random() < research_parameters['finetune_rate']:
                pool_feed_dict = {}
                pool_feed_dict.update({tf_dropout_rate:current_adaptive_dropout})


                pbatch_data = pool_dataset[(pool_id) * batch_size:(pool_id + 1) * batch_size,
                              :, :, :]
                pbatch_labels = pool_labels[(pool_id) * batch_size:(pool_id + 1) * batch_size,
                                :]
                pool_feed_dict.update({tf_pool_data_batch: pbatch_data,
                                       tf_pool_label_batch: pbatch_labels})

                _,_ = session.run([apply_pool_grads_op,update_pool_velocity_ops],
                                   feed_dict=pool_feed_dict)


def run_actual_finetune_operation(hard_pool_ft,overried_finetune_rate=False):
    '''
    Run the finetune opeartion in the default session
    :param hard_pool_ft:
    :return:
    '''
    global current_adaptive_dropout

    pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)

    # without if can give problems in exploratory stage because of no data in the pool
    if hard_pool_ft.get_size() > batch_size:
        # Train with latter half of the data

        for pool_id in range(0, (hard_pool_ft.get_size() // batch_size) - 1):
            if np.random.random() < research_parameters['finetune_rate'] or overried_finetune_rate:
                #print('fintuning network (pool_id: ',pool_id,')')
                pool_feed_dict = {}
                pool_feed_dict.update({tf_dropout_rate:current_adaptive_dropout})

                pbatch_data = pool_dataset[(pool_id) * batch_size:(pool_id + 1) * batch_size,
                              :, :, :]
                pbatch_labels = pool_labels[(pool_id) * batch_size:(pool_id + 1) * batch_size,
                                :]
                pool_feed_dict.update({tf_pool_data_batch: pbatch_data,
                                       tf_pool_label_batch: pbatch_labels})

                _ = session.run([apply_pool_grads_op,update_pool_velocity_ops],
                                   feed_dict=pool_feed_dict)


def top_n_accuracy(predictions,labels,n):
    '''
    Gives the top-n accuracy instead of top-1 accuracy
    Useful for large datasets
    :param predictions:
    :param labels:
    :param n:
    :return:
    '''
    assert predictions.shape[0] == labels.shape[0]
    correct_total = 0
    for pred_item, lbl_item in zip(predictions,labels):
        lbl_idx = int(np.argmax(lbl_item))
        top_preds = list(np.argsort(pred_item).flatten()[-n:])
        if lbl_idx in top_preds:
            correct_total += 1
    return (100.0 * correct_total)/predictions.shape[0]


def logging_hyperparameters(hyp_logger, cnn_hyperparameters, research_hyperparameters,
                            model_hyperparameters, interval_hyperparameters, dataset_info):

    hyp_logger.info('#Various hyperparameters')
    hyp_logger.info('# Initial CNN architecture related hyperparameters')
    hyp_logger.info(cnn_hyperparameters)
    hyp_logger.info('# Dataset info')
    hyp_logger.info(dataset_info)
    hyp_logger.info('# Research parameters')
    hyp_logger.info(research_hyperparameters)
    hyp_logger.info('# Interval parameters')
    hyp_logger.info(interval_hyperparameters)
    hyp_logger.info('# Model parameters')
    hyp_logger.info(model_hyperparameters)


def get_explore_action_probs(epoch, trial_phase, n_conv, n_fulcon):
    '''
    Explration action probabilities. We manually specify a probabilities of a stochastic policy
    used to explore the state space adequately
    :param epoch:
    :param trial_phase: use the global_trial_phase because at this time we only care about exploring actions
    :param n_conv:
    :return:
    '''

    n_all = n_conv + n_fulcon
    trial_phase_split = 0.4
    if epoch == 0 and trial_phase<trial_phase_split:
        logger.info('Finetune phase')
        trial_action_probs = [0.2 / (1.0 * n_all) for _ in range(n_all)]  # add
        trial_action_probs.extend([0.1, .35, 0.35])

    elif epoch == 0 and trial_phase >= trial_phase_split and trial_phase < 1.0:
        logger.info('Growth phase')
        # There is 0.1 amount probability to be divided between all the remove actions
        # We give 1/10 th as for other remove actions for the last remove action
        trial_action_probs = []
        trial_action_probs.extend([0.6 / (1.0 * (n_all)) for _ in range(n_all)])  # add
        trial_action_probs.extend([0.1, 0.15, 0.15])

    elif epoch == 2 and trial_phase >= 2.0 and trial_phase < 2.7:
        logger.info('Shrink phase')
        # There is 0.6 amount probability to be divided between all the remove actions
        # We give 1/10 th as for other remove actions for the last remove action
        raise NotImplementedError

    elif epoch == 2 and trial_phase >= 2.7 and trial_phase < 3.0:
        logger.info('Finetune phase')
        trial_action_probs = []
        trial_action_probs.extend([0.2 / (1.0 * n_all) for _ in range(n_all)])  # add
        trial_action_probs.extend([0.26, 0.26, 0.28])

    return trial_action_probs

# Continuous adaptation
def get_continuous_adaptation_action_in_different_epochs(
        q_learner, data, epoch, global_trial_phase,
        local_trial_phase, n_conv, n_fulcon, eps, adaptation_period):
    '''
    Continuously adapting the structure
    :param q_learner:
    :param data:
    :param epoch:
    :param trial_phase (global and local):
    :param n_conv:
    :param eps:
    :param adaptation_period:
    :return:
    '''

    adapting_now = None
    if epoch == 0:
        # Grow the network mostly (Exploration)
        logger.info('Explorative Growth Stage')
        state, action, invalid_actions = q_learner.output_action_with_type(
            data, 'Explore', p_action=get_explore_action_probs(epoch, global_trial_phase, n_conv,n_fulcon)
        )
        adapting_now = True
    else:
        logger.info('Epsilon: %.3f',eps)

        logger.info('Greedy Adapting period of epoch (both)')

        non_adapting_threshold = 0.1

        if local_trial_phase> non_adapting_threshold:
            if np.random.random() >= eps:
                state, action, invalid_actions = q_learner.output_action_with_type(
                    data, 'Greedy')

            else:
                state, action, invalid_actions = q_learner.output_action_with_type(
                    data, 'Stochastic'
                )
        else:

            # This is a hacky way to get only non-adaptive actions for sometime at the beginning of every task
            # As this might be cause some problems in convergence
            logger.info('\tGetting non adaptive actions for a bit')
            state, action, invalid_actions = q_learner.output_action_with_type(
                data, 'Explore', p_action=get_explore_action_probs(0, 0.01, n_conv, n_fulcon)
            )

        adapting_now = True

    return state, action, invalid_actions, adapting_now

def get_continuous_adaptation_action_randomly(q_learner, data, epoch, global_trial_phase, local_trial_phase, n_conv, n_fulcon, eps, adaptation_period):
    '''
    Continuously adapting the structure
    :param q_learner:
    :param data:
    :param epoch:
    :param trial_phase (global and local):
    :param n_conv:
    :param eps:
    :param adaptation_period:
    :return:
    '''

    adapting_now = None

    logger.info('Epsilon: %.3f',eps)
    if adaptation_period=='first':
        if local_trial_phase<=0.5:
            logger.info('Greedy Adapting period of epoch (first)')

            state, action, invalid_actions = q_learner.output_action_with_type(
                data, 'Stochastic'
            )
            adapting_now = True

        else:
            logger.info('Greedy Not adapting period of epoch (first)')
            state, action, invalid_actions = q_learner.get_finetune_action(data)

            adapting_now = False

    elif adaptation_period == 'last':
        if local_trial_phase > 0.5:
            logger.info('Greedy Adapting period of epoch (last)')
            state, action, invalid_actions = q_learner.output_action_with_type(
                data, 'Stochastic'
            )
            adapting_now=True
        else:
            logger.info('Not adapting period of epoch (last). Randomly outputting (Donothing, Naive Triain, Finetune')
            state, action, invalid_actions = q_learner.get_finetune_action(data)

            adapting_now = False

    elif adaptation_period =='both':

        logger.info('Greedy Adapting period of epoch (both)')
        state, action, invalid_actions = q_learner.output_action_with_type(
            data, 'Stochastic'
        )
        adapting_now = True

    elif adaptation_period =='none':

        logger.info('Greedy Adapting period of epoch (both)')
        logger.info('Not adapting period of epoch. Randomly outputting (Donothing, Naive Triain, Finetune')
        if np.random.random() < 0.3:
            state, action, invalid_actions = q_learner.get_naivetrain_action(data)
        else:
            state, action, invalid_actions = q_learner.get_finetune_action(data)

        adapting_now = False

    else:
        raise NotImplementedError

    return state, action, invalid_actions, adapting_now


def calculate_pool_accuracy(hard_pool):
    '''
    Calculates the mini-batch wise accuracy for all the data points in the pool
    :param hard_pool:
    :return:
    '''
    global batch_size, tf_pool_data_batch, tf_pool_label_batch
    pool_accuracy = []
    pool_dataset, pool_labels = hard_pool.get_pool_data(False)
    for pool_id in range(hard_pool.get_size() // batch_size):
        pbatch_data = pool_dataset[pool_id * batch_size:(pool_id + 1) * batch_size, :, :, :]
        pbatch_labels = pool_labels[pool_id * batch_size:(pool_id + 1) * batch_size, :]
        pool_feed_dict = {tf_pool_data_batch: pbatch_data,
                          tf_pool_label_batch: pbatch_labels}
        p_predictions = session.run(pool_pred, feed_dict=pool_feed_dict)
        if num_labels <= 100:
            pool_accuracy.append(accuracy(p_predictions, pbatch_labels))
        else:
            pool_accuracy.append(top_n_accuracy(p_predictions, pbatch_labels, 5))

    return np.mean(pool_accuracy) if len(pool_accuracy) > 0 else 0


if __name__ == '__main__':

    # Various run-time arguments specified
    #
    num_gpus=1
    allow_growth = False
    fake_tasks = False
    noise_label_rate = None
    noise_image_rate = None
    adapt_randomly = False
    use_fse_capacity = False
    inter_op_threads, intra_op_threads = 0,0
    data_overlap = False
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "", ["output_dir=", "num_gpus=", "memory=", "intra_op_threads=", "inter_op_threads=",'allow_growth=',
                               'adapt_structure=', 'rigid_pooling=', 'adapt_randomly='])
    except getopt.GetoptError as err:
        print(err.with_traceback())
        print('<filename>.py --output_dir= --num_gpus= --memory=')

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '--output_dir':
                output_dir = arg
            if opt == '--num_gpus':
                num_gpus = int(arg)
            if opt == '--memory':
                mem_frac = float(arg)
            if opt == '--allow_growth':
                allow_growth = bool(int(arg))
            if opt == '--adapt_structure':
                adapt_structure = bool(int(arg))
            if opt == '--rigid_pooling':
                rigid_pooling = bool(int(arg))
            if opt == '--adapt_randomly':
                adapt_randomly = bool(int(arg))
            if opt == '--intra_op_threads':
                intra_op_threads = int(arg)
            if opt=='--inter_op_threads':
                inter_op_threads = int(arg)
            if opt=='--class_overlap':
                data_overlap = bool(int(arg))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not (adapt_structure or rigid_pooling):
        rigid_naive = True

    # Setting up loggers
    logger, perf_logger, cnn_structure_logger, \
    q_logger, class_dist_logger, pool_dist_logger, pool_dist_ft_logger, \
    hyp_logger, error_logger, prune_logger = setup_loggers(adapt_structure)

    logger.info('Created loggers')

    logger.info('Created Output directory: %s', output_dir)
    logger.info('Received all the required user arguments at Runtime')
    logger.debug('Output DIR: %s', output_dir)
    logger.info('Number of GPUs: %d', num_gpus)
    logger.info('Memory fraction per GPU: %.3f', mem_frac)
    logger.info('Use AdaCNN: %d', adapt_structure)
    logger.info('Use rigid pooling: %d', rigid_pooling)
    logger.info('Use rigid naive: %d',rigid_naive)
    logger.info('Adapt Randomly: %d',adapt_randomly)

    # =====================================================================
    # VARIOS SETTING UPS
    # SET FROM MAIN FUNCTIONS OF OTHER CLASSES

    set_varialbes_with_input_arguments(adapt_structure,rigid_pooling)
    cnn_intializer.set_from_main(research_parameters, logging_level, logging_format)

    logger.info('Creating CNN hyperparameters and operations in the correct format')
    # Getting hyperparameters
    cnn_ops, cnn_hyperparameters, final_2d_width, final_2d_height = utils.get_ops_hyps_from_string(dataset_info, cnn_string)

    act_vector_logger_dict = setup_activation_vector_logger()
    activation_vec_ops = []
    for op in cnn_ops:
        if ('conv' in op or 'fulcon' in op) and op!='fulcon_out':
            activation_vec_ops.append(op)

    init_cnn_ops, init_cnn_hyperparameters, final_2d_width, final_2d_height  = utils.get_ops_hyps_from_string(dataset_info, cnn_string)

    logger.info('Created CNN hyperparameters and operations in the correct format successfully\n')

    ada_cnn_adapter.set_from_main(research_parameters, final_2d_height, final_2d_width,cnn_ops, cnn_hyperparameters, logging_level, logging_format)
    cnn_optimizer.set_from_main(research_parameters,model_hyperparameters,logging_level,logging_format,cnn_ops, final_2d_height, final_2d_width)

    logger.info('Creating loggers\n')
    logger.info('=' * 80 + '\n')
    logger.info('Recognized following convolution operations')
    logger.info(cnn_ops)
    logger.info('With following Hyperparameters')
    logger.info(cnn_hyperparameters)
    logger.info(('=' * 80) + '\n')
    logging_hyperparameters(
        hyp_logger,cnn_hyperparameters,research_parameters,
        model_hyperparameters,interval_parameters,dataset_info
    )

    logging.info('Reading data from HDF5 File sucessful.\n')

    # Setting up graph parameters
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.intra_op_parallelism_threads = intra_op_threads
    config.inter_op_parallelism_threads = inter_op_threads
    if mem_frac is not None:
        config.gpu_options.per_process_gpu_memory_fraction = mem_frac
    else:
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

    session = tf.InteractiveSession(config=config)

    # Defining pool
    if adapt_structure or rigid_pooling:
        logger.info('Defining pools of data (validation and finetuning)')
        hardness = 0.5

        hard_pool_valid = Pool(size=pool_size//2, batch_size=batch_size, image_size=image_size,
                               num_channels=num_channels, num_labels=num_labels, assert_test=False)
        hard_pool_ft = Pool(size=pool_size//2, batch_size=batch_size, image_size=image_size,
                               num_channels=num_channels, num_labels=num_labels, assert_test=False)

        logger.info('Defined pools of data successfully\n')

    first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'
    # -1 is because we don't want to count fulcon_out
    layer_count = len(cnn_ops) - 1

    # ids of the convolution ops
    convolution_op_ids = []
    fulcon_op_ids = []
    if adapt_structure:
        add_amout_filter_vec = []
        for op_i, op in enumerate(cnn_ops):
            if 'conv' in op:
                convolution_op_ids.append(op_i)
                add_amout_filter_vec.append(model_hyperparameters['add_amount'])
            # Do not put the fulcon_out in the fulcon_ids we do not adapt the output layer
            # it already has the total number  of classes
            if 'fulcon' in op and op!='fulcon_out':
                fulcon_op_ids.append(op_i)
                add_amout_filter_vec.append(model_hyperparameters['add_fulcon_amount'])

            if 'pool' in op:
                add_amout_filter_vec.append(1)

        add_amout_filter_vec = np.asarray(add_amout_filter_vec)

    logger.info('Found all convolution opeartion ids')

    # Defining initial hyperparameters as TF variables (so we can change them later during adaptation)
    # Define global set (used for learning rate decay)
    # Define CNN variables
    # Define CNN variable momentums
    # Define all the required operations
    with tf.variable_scope(TF_GLOBAL_SCOPE,reuse=False) as scope:
        global_step = tf.get_variable(initializer=0, dtype=tf.int32, trainable=False, name='global_step')
        logger.info('Defining TF Hyperparameters')
        tf_cnn_hyperparameters = cnn_intializer.init_tf_hyperparameters(cnn_ops, cnn_hyperparameters)
        logger.info('Defining Weights and Bias for CNN operations')
        _ = cnn_intializer.initialize_cnn_with_ops(cnn_ops, cnn_hyperparameters)
        logger.info('Following parameters defined')
        logger.info([v.name for v in tf.trainable_variables()])
        logger.info('='*80)
        logger.info('Defining Velocities for Weights and Bias for CNN operations')
        _ = cnn_intializer.define_velocity_vectors(scope, cnn_ops, cnn_hyperparameters)
        logger.info('Following parameters defined')
        logger.info([v.name for v in tf.global_variables()])
        logger.info('=' * 80)
    logger.info(('=' * 80) + '\n')


    if research_parameters['adapt_structure']:

        # Pruning hyperparameter initialization

        # Adapting Policy Learner
        current_adaptive_dropout = get_adaptive_dropout()
        state_history_length = 2
        growth_adapter = ada_cnn_qlearner.AdaCNNAdaptingQLearner(
            qlearner_type='growth', discount_rate=0.5, fit_interval=1,
            exploratory_tries_factor=5, exploratory_interval=10000, stop_exploring_after=10,
            filter_vector=filter_vector,
            conv_ids=convolution_op_ids, fulcon_ids=fulcon_op_ids, net_depth=layer_count,
            n_conv=len(convolution_op_ids), n_fulcon=len(fulcon_op_ids),
            epsilon=0.5, target_update_rate=20,
            batch_size=32, persist_dir=output_dir,
            session=session, random_mode=False,
            state_history_length=state_history_length,
            hidden_layers=[128, 64, 32], momentum=0.9, learning_rate=0.01,
            rand_state_length=32, add_amount=model_hyperparameters['add_amount'],
            remove_amount=model_hyperparameters['remove_amount'], add_fulcon_amount=model_hyperparameters['add_fulcon_amount'],
            num_classes=num_labels, filter_min_threshold=model_hyperparameters['filter_min_threshold'],
            trial_phase_threshold=1.0, binned_data_dist_length=model_hyperparameters['binned_data_dist_length'],
            top_k_accuracy=model_hyperparameters['top_k_accuracy']
        )


    # Running initialization opeartion
    logger.info('Running global variable initializer')
    init_op = tf.global_variables_initializer()
    _ = session.run(init_op)
    logger.info('Variable initialization successful\n')

    # Defining all Tensorflow ops required for
    # calculating logits, loss, predictions
    logger.info('Defining all the required Tensorflow operations')

    dataset_filenames, dataset_sizes = dataset_name_factory.new_get_noncol_train_data_sorted_by_direction_noncol_test_data()

    train_data_gen = data_generator.DataGenerator(
        batch_size, num_labels, dataset_sizes['train_dataset'],
        original_image_size, session, dataset_filenames['train_dataset'],
        image_size + [num_channels], False
    )

    test_data_gen = data_generator.DataGenerator(
        batch_size, num_labels, dataset_sizes['test_dataset'],
        original_image_size, session, dataset_filenames['test_dataset'],
        image_size + [num_channels], True
    )

    with tf.variable_scope(TF_GLOBAL_SCOPE, reuse=True) as scope:
        define_tf_ops(global_step,tf_cnn_hyperparameters,init_cnn_hyperparameters)
    logger.info('Defined all TF operations successfully\n')

    logger.debug('CNN_HYPERPARAMETERS')
    logger.debug('\t%s\n', tf_cnn_hyperparameters)

    logger.debug('TRAINABLE_VARIABLES')
    logger.debug('\t%s\n', [v.name for v in tf.trainable_variables()])

    logger.info('Variables initialized...')

    train_losses = []
    mean_train_loss = 0
    prev_test_accuracy = 0  # used to calculate test accuracy drop

    rolling_ativation_means = {}
    for op in cnn_ops:
        if 'conv' in op:
            logger.debug('\tDefining rolling activation mean for %s', op)
            rolling_ativation_means[op] = np.zeros([cnn_hyperparameters[op]['weights'][3]])

    current_state, current_action,curr_adaptation_status = None, None,None
    curr_layer_sizes = model_hyperparameters['start_filter_vector'] if adapt_structure else None
    pool_acc_queue = []

    current_q_learn_op_id = 0
    logger.info('Convolutional Op IDs: %s', convolution_op_ids)

    logger.info('Starting Training Phase')

    # Check if loss is stabilized (for starting adaptations)
    previous_loss = 1e5  # used for the check to start adapting

    # Reward for Q-Learner
    prev_pool_accuracy = 0
    max_pool_accuracy = 0

    # Stop and start adaptations when necessary
    start_adapting = False
    stop_adapting = False
    adapt_period = 'both'
    # need to have a starting value because if the algorithm choose to add the data to validation set very first step
    train_accuracy = 0

    running_binned_data_dist_vector = np.zeros((model_hyperparameters['binned_data_dist_length']),dtype=np.float32)
    binned_data_dist_decay = 0.5

    current_action_type = growth_adapter.get_naivetrain_action_type() if adapt_structure else None

    current_train_acc, prev_train_acc = 0,0

    if adapt_structure:
        adapter = growth_adapter

    research_parameters['momentum'] = 0.9
    research_parameters['pool_momentum'] = 0.1  # 0.9**(epoch+1)

    global_batch_id = -1
    for epoch in range(n_epochs):

        for train_env_idx in range(3):

            tr_img_id, tr_images, tr_labels = train_data_gen.sample_a_batch_from_data(train_env_idx, shuffle=True)
            cnn_optimizer.update_hyperparameters(research_parameters)

            # we stop 'num_gpus' items before the ideal number of training batches
            # because we always keep num_gpus items for valid data in memory
            n_iterations = dataset_sizes['train_dataset'][train_env_idx]//batch_size
            all_dataset_sizes = sum([dataset_sizes['train_dataset'][tr_e]//batch_size for tr_e in range(3)])

            for batch_id in range(n_iterations):

                global_batch_id += 1
                global_trial_phase = (global_batch_id * 1.0 / all_dataset_sizes)
                local_trial_phase = batch_id * 1.0 / n_iterations

                print('n_iter: ',n_iterations,'batch id: ',batch_id,' globa: ',global_trial_phase, ' local: ',local_trial_phase)
                t0 = time.clock()  # starting time for a batch

                # We load 1 extra batch (chunk_size+1) because we always make the valid batch the batch_id+1

                # Feed dicitonary with placeholders for each tower
                batch_data, batch_labels, batch_weights = [], [], []
                train_feed_dict = {}
                train_feed_dict.update({tf_dropout_rate:current_adaptive_dropout})

                # ========================================================
                # Creating data batchs for the towers

                tr_img_id, tr_images, tr_labels = train_data_gen.sample_a_batch_from_data(train_env_idx,
                                                                                          shuffle=True)
                tr_ohe_labels = np.zeros(shape=(batch_size,num_labels),dtype=np.float32)
                tr_ohe_labels[:,tr_labels.astype(np.int32)] = 1.0

                logger.debug('Got label sequence (for batch %d)', global_batch_id)

                # Get current batch of data nd labels (training)
                batch_data =tr_images
                batch_labels = tr_ohe_labels

                label_seq = tr_labels

                if batch_id == 0:
                    logger.info('\tDataset shape: %s', tr_images.shape)
                    logger.info('\tLabels shape: %s', tr_labels.shape)

                # calculate binned data distribution and running average of that for RL state
                batch_weights = np.ones((batch_size,))

                train_feed_dict.update(get_dropout_placeholder_dict())
                train_feed_dict.update({
                    train_data_gen.tf_image_ph: batch_data, train_data_gen.tf_label_ph: tr_labels,
                    tf_data_weights: batch_weights
                })

                # =========================================================

                t0_train = time.clock()

                # =========================================================
                # Training Phase (Calculate loss and predictions)

                l, super_loss_vec, train_predictions = session.run(
                    [tf_loss, tf_loss_vec,
                     tf_pred], feed_dict=train_feed_dict
                )

                current_train_acc = accuracy(train_predictions, batch_labels) / 100.0
                # =========================================================

                # ==========================================================
                # Updating Pools of data

                # This if condition get triggered stochastically for AdaCNN
                # Never for Rigid-Pooling or Naive-Training
                if adapt_structure:
                        #print('Only collecting valid data')
                        # Concatenate current 'num_gpus' batches to a single matrix

                        single_iteration_batch_data, single_iteration_batch_labels = \
                            session.run(
                                [tf_train_data_batch, tf_train_label_batch],
                                feed_dict=train_feed_dict)


                        hard_pool_valid.add_hard_examples(single_iteration_batch_data, single_iteration_batch_labels,
                                                          super_loss_vec, max(0.05,prev_train_acc - current_train_acc))

                        logger.debug('Pooling data summary')
                        logger.debug('\tData batch size %d', single_iteration_batch_data.shape[0])
                        logger.debug('\tAccuracy %.3f', train_accuracy)
                        logger.debug('\tPool size (Valid): %d', hard_pool_valid.get_size())

                else:

                    # Concatenate current 'num_gpus' batches to a single matrix

                    single_iteration_batch_data, single_iteration_batch_labels = batch_data, \
                                                                                 batch_labels

                    # Higer rates of accumulating data causes the pool to lose uniformity
                    if rigid_pooling:
                        #print('add examples to rigid pooling naive')
                        hard_pool_ft.add_hard_examples(single_iteration_batch_data, single_iteration_batch_labels,
                                                          super_loss_vec,1.0)
                        logger.debug('\tPool size (FT): %d', hard_pool_ft.get_size())

                    # =========================================================
                    # # Training Phase (Optimization)

                    feed_dict = get_dropout_placeholder_dict()
                    feed_dict.update(train_feed_dict)
                    for _ in range(iterations_per_batch):
                        #print('training on current batch (action type: ', current_action_type, ')')
                        _ = session.run(
                            [apply_grads_op,update_train_velocity_op], feed_dict=feed_dict
                        )

                t1_train = time.clock()

                if np.isnan(l):
                    logger.critical('Diverged (NaN detected) (batchID) %d (last Cost) %.3f', batch_id,
                                    train_losses[-1])
                assert not np.isnan(l)

                train_losses.append(l)
                prev_train_acc = current_train_acc

                # =============================================================
                # log hard_pool distribution over time
                if (adapt_structure or rigid_pooling) and \
                        (batch_id > 0 and batch_id % interval_parameters['history_dump_interval'] == 0):

                    # with open(output_dir + os.sep + 'Q_' + str(epoch) + "_" + str(batch_id)+'.pickle', 'wb') as f:
                    #    pickle.dump(adapter.get_Q(), f, pickle.HIGHEST_PROTOCOL)

                    pool_dist_string = ''
                    for val in hard_pool_valid.get_class_distribution():
                        pool_dist_string += str(val) + ','

                    pool_dist_logger.info('%s%d', pool_dist_string, hard_pool_valid.get_size())

                    pool_dist_string = ''
                    for val in hard_pool_ft.get_class_distribution():
                        pool_dist_string += str(val) + ','

                    pool_dist_ft_logger.info('%s%d', pool_dist_string, hard_pool_ft.get_size())
                # ==============================================================


                # ================================================================
                # Things done if one of below scenarios
                # For AdaCNN if adaptations stopped
                # For rigid pool CNNs from the beginning
                if ((research_parameters['pooling_for_nonadapt'] and
                         not research_parameters['adapt_structure']) or stop_adapting) and \
                        (batch_id > 0 and batch_id % interval_parameters['finetune_interval'] == 0):

                    logger.info('Pooling for non-adaptive CNN')
                    logger.info('Using dropout rate of: %.3f', dropout_rate)

                    # ===============================================================
                    # Finetune with data in hard_pool_ft (AdaCNN)
                    run_actual_finetune_operation(hard_pool_ft)

                    # =================================================================
                    # Calculate pool accuracy (hard_pool_valid)
                    if adapt_structure:
                        mean_pool_accuracy = get_pool_valid_accuracy(hard_pool_valid)
                    elif rigid_pooling:
                        mean_pool_accuracy = get_pool_valid_accuracy(hard_pool_ft)

                    logger.info('\tPool accuracy (hard_pool_valid): %.5f', mean_pool_accuracy)
                    # ==================================================================

                # ==============================================================
                # Testing Phase
                for test_env_idx in range(3):
                    if batch_id % interval_parameters['test_interval'] == 0:
                        mean_train_loss = np.mean(train_losses)
                        logger.info('=' * 60)
                        logger.info('\t (Test Environment: %d)Batch ID: %d' % (test_env_idx,batch_id))
                        if decay_learning_rate:
                            logger.info('\tLearning rate: %.5f' % session.run(tf_learning_rate))
                        else:
                            logger.info('\tLearning rate: %.5f' % start_lr)

                        logger.info('\tMinibatch Mean Loss: %.3f' % mean_train_loss)
                        logger.info('\tTrain Accuracy: %.3f'%(train_accuracy*100))
                        logger.info('\tValidation accumulation rate %.3f', 0.5*(valid_acc_decay**epoch))
                        logger.info('\tCurrent adaptive Dropout: %.3f', current_adaptive_dropout)
                        logger.info('\tTrial phase: %.3f (Local) %.3f (Global)', local_trial_phase, global_trial_phase)

                        ts_img_id, ts_images, ts_labels = test_data_gen.sample_a_batch_from_data(test_env_idx, shuffle=False)

                        ts_ohe_labels = np.zeros(shape=(batch_size,num_labels),dtype=np.float32)
                        ts_ohe_labels[:,ts_labels.astype(np.int32)] = 1.0

                        test_accuracies = []
                        for test_batch_id in range(dataset_sizes['test_dataset'][test_env_idx] // batch_size):

                            feed_test_dict = {test_data_gen.tf_image_ph: ts_images, test_data_gen.tf_label_ph: ts_labels}
                            test_predictions = session.run(test_predicitons_op, feed_dict=feed_test_dict)
                            test_accuracies.append(accuracy(test_predictions, ts_ohe_labels))

                        current_test_accuracy = np.mean(test_accuracies)

                        logger.info('\tTest Accuracy: %.3f' % current_test_accuracy)

                        logger.info('=' * 60)
                        logger.info('')

                        # Logging error
                        prev_test_accuracy = current_test_accuracy
                        error_logger.info('%d,%.3f,%.3f,%.3f',
                                          global_batch_id, mean_train_loss, train_accuracy*100.0,
                                          np.mean(test_accuracies)
                                          )

                # ====================================================================

                    if research_parameters['adapt_structure'] and \
                            not start_adapting and \
                            (previous_loss - mean_train_loss < research_parameters['loss_diff_threshold'] or batch_id >
                                research_parameters['start_adapting_after']):
                        start_adapting = True
                        logger.info('=' * 80)
                        logger.info('Loss Stabilized: Starting structural adaptations...')
                        logger.info('Hardpool acceptance rate: %.2f', research_parameters['hard_pool_acceptance_rate'])
                        logger.info('=' * 80)

                    previous_loss = mean_train_loss

                    # reset variables
                    mean_train_loss = 0.0
                    train_losses = []

                # =======================================================================
                # Adaptations Phase of AdaCNN
                if research_parameters['adapt_structure']:

                    # ==============================================================
                    # Before starting the adaptations
                    if not start_adapting and batch_id > 0 and batch_id % (interval_parameters['finetune_interval']) == 0:

                        logger.info('Finetuning before starting adaptations. (To gain a reasonable accuracy to start with)')

                        pool_dataset, pool_labels = hard_pool_ft.get_pool_data(True)

                        # without if can give problems in exploratory stage because of no data in the pool
                        if hard_pool_ft.get_size() > batch_size:
                            # Train with latter half of the data
                            run_actual_finetune_operation(hard_pool_ft)
                    # ==================================================================

                    # ==================================================================
                    # Actual Adaptations
                    if (start_adapting and not stop_adapting) and batch_id > 0 and \
                                            batch_id % interval_parameters['policy_interval'] == 4:

                        # ==================================================================
                        # Policy Update (Update policy only when we take actions actually using the qlearner)
                        # (Not just outputting finetune action)
                        # ==================================================================
                        if (not adapt_randomly) and curr_adaptation_status and current_state:

                            # ==================================================================
                            # Calculating pool accuracy
                            p_accuracy = calculate_pool_accuracy(hard_pool_ft)
                            pool_acc_queue.append(p_accuracy)
                            if len(pool_acc_queue) > 20:
                                del pool_acc_queue[0]
                            # ===============================================================================

                            # don't use current state as the next state, current state is for a different layer
                            next_state = []
                            affected_layer_index = 0
                            for li, la in enumerate(current_action):
                                if la is None:
                                    assert li not in convolution_op_ids
                                    next_state.append(0)
                                    continue
                                elif la[0] == 'add':
                                    next_state.append(current_state[li] + la[1])
                                    affected_layer_index = li
                                elif la[0] == 'remove':
                                    next_state.append(current_state[li] - la[1])
                                    affected_layer_index = li
                                else:
                                    next_state.append(current_state[li])

                            next_state = tuple(next_state)
                            curr_layer_sizes = np.asarray(next_state)

                            logger.info(('=' * 40) + ' Update Summary ' + ('=' * 40))
                            logger.info('\tState (prev): %s', str(current_state))
                            logger.info('\tAction (prev): %s', str(current_action))
                            logger.info('\tState (next): %s', str(next_state))
                            logger.info('\tPool Accuracy: %.3f', p_accuracy)
                            logger.info('\tPrev pool Accuracy: %.3f\n', prev_pool_accuracy)
                            logger.info(('=' * 80) + '\n')
                            assert not np.isnan(p_accuracy)

                            # ================================================================================
                            # Actual updating of the policy
                            adapter.update_policy({'prev_state': current_state, 'prev_action': current_action,
                                                   'curr_state': next_state,
                                                   'next_accuracy': None,
                                                   'prev_accuracy': None,
                                                   'pool_accuracy': pool_acc_queue[-1],
                                                   'prev_pool_accuracy': pool_acc_queue[0],
                                                   'max_pool_accuracy': max_pool_accuracy,
                                                   'invalid_actions': curr_invalid_actions,
                                                   'batch_id': global_batch_id,
                                                   'layer_index': affected_layer_index}, True)
                            # ===================================================================================

                            cnn_structure_logger.info(
                                '%d:%s:%s:%.5f:%s', global_batch_id, current_state,
                                current_action, p_accuracy,
                                utils.get_cnn_string_from_ops(cnn_ops, cnn_hyperparameters)
                            )

                            # The growth adapter has the random states for Q_eval^rand
                            q_logger.info('%d,%.5f', global_batch_id, growth_adapter.get_average_Q())

                            logger.debug('Resetting both data distribution means')

                            max_pool_accuracy = max(max(pool_acc_queue), p_accuracy)
                            prev_pool_accuracy = p_accuracy

                        # ===================================================================================
                        # Execute action according to the policy
                        # ==================================================================================
                        filter_dict, filter_list = {}, []
                        for op_i, op in enumerate(cnn_ops):
                            if 'conv' in op:
                                filter_dict[op_i] = cnn_hyperparameters[op]['weights'][3]
                                filter_list.append(cnn_hyperparameters[op]['weights'][3])
                            elif 'pool' in op:
                                filter_dict[op_i] = 0
                                filter_list.append(0)
                            elif 'fulcon' in op and op!='fulcon_out':
                                filter_dict[op_i] = cnn_hyperparameters[op]['out']
                                filter_list.append(cnn_hyperparameters[op]['out'])

                        # For epoch 0 and 1
                        # Epoch 0: Randomly grow the network
                        # Epoch 1: Deterministically grow the network
                        if not adapt_randomly:
                            current_state, current_action, curr_invalid_actions,curr_adaptation_status = get_continuous_adaptation_action_in_different_epochs(
                                adapter, data = {'filter_counts': filter_dict, 'filter_counts_list': filter_list, 'binned_data_dist':running_binned_data_dist_vector.tolist()}, epoch=epoch,
                                global_trial_phase=global_trial_phase, local_trial_phase=local_trial_phase, n_conv=len(convolution_op_ids), n_fulcon=len(fulcon_op_ids),
                                eps=start_eps, adaptation_period=adapt_period)
                        else:
                            current_state, current_action, curr_invalid_actions, curr_adaptation_status = get_continuous_adaptation_action_randomly(
                                adapter, data={'filter_counts': filter_dict, 'filter_counts_list': filter_list,
                                               'binned_data_dist': running_binned_data_dist_vector.tolist()},
                                epoch=epoch,
                                global_trial_phase=global_trial_phase, local_trial_phase=local_trial_phase,
                                n_conv=len(convolution_op_ids), n_fulcon=len(fulcon_op_ids),
                                eps=start_eps, adaptation_period=adapt_period)

                        current_action_type = adapter.get_action_type_with_action_list(current_action)
                        # reset the binned data distribution
                        running_binned_data_dist_vector = np.zeros(
                            (model_hyperparameters['binned_data_dist_length']), dtype=np.float32)

                        for li, la in enumerate(current_action):
                            # pooling and fulcon layers
                            if la is None or la[0] == 'do_nothing':
                                continue

                            # where all magic happens (adding and removing filters)
                            si, ai = current_state, la
                            current_op = cnn_ops[li]

                            for tmp_op in reversed(cnn_ops):
                                if 'conv' in tmp_op:
                                    last_conv_id = tmp_op
                                    break

                            if ai[0] == 'add':
                                if 'conv' in current_op:
                                    run_actual_add_operation(session,current_op,li,last_conv_id,hard_pool_ft, epoch)
                                elif 'fulcon' in current_op:
                                    run_actual_add_operation_for_fulcon(session,current_op,li, last_conv_id, hard_pool_ft, epoch)

                            elif 'conv' in current_op and ai[0] == 'remove':
                                if 'conv' in current_op:
                                    run_actual_remove_operation(session,current_op,li,last_conv_id,hard_pool_ft)
                                elif 'fulcon' in current_op:
                                    raise NotImplementedError

                            elif 'conv' in current_op and ai[0] == 'finetune':
                                # pooling takes place here
                                run_actual_finetune_operation(hard_pool_ft)
                                break

                        # =============================================================

                    t1 = time.clock()
                    op_count = len(tf.get_default_graph().get_operations())
                    var_count = len(tf.global_variables()) + len(tf.local_variables()) + len(tf.model_variables())
                    perf_logger.info('%d,%.5f,%.5f,%d,%d', global_batch_id, t1 - t0,
                                     (t1_train - t0_train) / num_gpus, op_count, var_count)

        # =======================================================
        # Decay learning rate (if set) Every 2 epochs
        if decay_learning_rate and epoch>0 and epoch%2==1:
            session.run(increment_global_step_op)
        # ======================================================

        # AdaCNN Algorithm
        if research_parameters['adapt_structure']:
            if epoch > 0:
                start_eps = max([start_eps*eps_decay,0.1])
                adapt_period = np.random.choice(['first','last','both'],p=[0.0,0.0,1.0])
                # At the moment not stopping adaptations for any reason
                # stop_adapting = adapter.check_if_should_stop_adapting()

        #cnn_model_saver.save_cnn_hyperparameters(cnn_ops,final_2d_width,cnn_hyperparameters,'cnn-hyperparameters-%d.pickle'%epoch)
        #cnn_model_saver.save_cnn_weights(cnn_ops,session,'cnn-model-%d.ckpt'%epoch)