import cnn_learner_detached
import numpy as np
import config
import models_utils
from math import ceil
import dataset_name_factory
import tensorflow as tf
import cnn_variable_initializer
import logging
import sys
import os
import getopt

scope_list = config.TF_ANG_SCOPES
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

def sample_set_of_hyperparameters():

    # TF_ANG_VAR_SHAPES_DETACHED = {'conv1': [4, 8, 3, 32], 'pool1': [1, 4, 8, 1], 'conv2': [5, 5, 32, 64],
    #                              'pool2': [1, 3, 3, 1],
    #                              'conv3': [3, 3, 64, 128], 'pool3': [1, 3, 3, 1], 'conv4': [3, 3, 128, 128],
    #                              'fc1': [fc_h * fc_w * 128, FC1_WEIGHTS_DETACHED],
    #                              'out': [FC1_WEIGHTS_DETACHED, 1]}

    if np.random.random()<0.5:
        config.BATCH_SIZE = np.random.choice([10,25,50])

    if np.random.random()<0.5:
        config.START_LR = np.random.choice([0.001,0.0001,0.00001,0.0005,0.00005])

    if np.random.random()<0.5:
        config.IN_DROPOUT = np.random.choice([0.0, 0.1, 0.2, 0.5])

    if np.random.random()<0.5:
        config.LAYER_DROPOUT = np.random.choice([0.0, 0.1, 0.34, 0.5])

    if np.random.random()<0.5:
        config.L2_BETA = np.random.choice([0.0, 0.01, 0.001,0.0001,0.005, 0.0005])

    fc_h, fc_w = config.TF_INPUT_AFTER_RESIZE[0],config.TF_INPUT_AFTER_RESIZE[1]
    prev_fc_h, prev_fc_w = -1, -1
    previous_selected_depth = 3
    curr_fc = -1

    for op in config.TF_ANG_SCOPES:
        if 'conv' in op:
            # Convolution shape sampling
            if np.random.random()<0.5:
                conv_shape_choices = [
                    [2, 4, previous_selected_depth, 32], [2, 4, previous_selected_depth, 64],
                    [2, 4, previous_selected_depth, 128],
                    [4, 8, previous_selected_depth, 32], [4, 8, previous_selected_depth, 64],
                    [4, 8, previous_selected_depth, 128],
                    [2, 2, previous_selected_depth, 32], [2, 2, previous_selected_depth, 64],
                    [2, 2, previous_selected_depth, 128],
                    [4, 4, previous_selected_depth, 32], [4, 4, previous_selected_depth, 64],
                    [4, 4, previous_selected_depth, 128]]

                choice_idx = np.random.choice(list(range(len(conv_shape_choices))))
                curr_conv_shape = conv_shape_choices[choice_idx]

                config.TF_ANG_VAR_SHAPES_DETACHED[op] = curr_conv_shape
            else:
                #no matter we choose random or just previous, we always need to update the in channel depth
                config.TF_ANG_VAR_SHAPES_DETACHED[op][2] = previous_selected_depth

            previous_selected_depth = config.TF_ANG_VAR_SHAPES_DETACHED[op][-1]
            print('prev depth: %d'%previous_selected_depth)

            # Convolution Stride Sampling
            if np.random.random()<0.5:
                conv_stride_choices = [[1,1,1,1], [1,2, 4,1], [1,2, 2,1],[1,4, 4,1]]
                choice_idx = np.random.choice(list(range(len(conv_stride_choices))))
                curr_conv_stride = conv_stride_choices[choice_idx]

            else:
                curr_conv_stride = config.TF_ANG_STRIDES[op]

            fc_h = ceil(fc_h * 1.0 / curr_conv_stride[1])
            fc_w = ceil(fc_w * 1.0 / curr_conv_stride[2])

            # if current output width and height is less than 2, we use a stride of 1
            if fc_h <= 2 or fc_w <= 2:
                curr_conv_stride = [1, 1, 1, 1]
                # if the fc_h or fc_w goes below 2 we need to reset fc_h and fc_w to
                # previously found whatever value. Because we are using a stride of 1 here
                fc_h = prev_fc_h
                fc_w = prev_fc_w
            else:
                prev_fc_h = fc_h
                prev_fc_w = fc_w
            config.TF_ANG_STRIDES[op] = curr_conv_stride

        if 'pool' in op:

            if np.random.random()<0.5:
                curr_pool_choices = [
                    [1, 2, 4, 1],
                    [1, 4, 8, 1],
                    [1, 2, 2, 1],
                    [1, 4, 4, 1]
                ]
                choice_idx = np.random.choice(list(range(len(curr_pool_choices))))
                curr_pool_shape = curr_pool_choices[choice_idx]
                config.TF_ANG_VAR_SHAPES_DETACHED[op] = curr_pool_shape

            if np.random.random() < 0.5:
                pool_stride_choices = [[1, 1, 1, 1], [1, 2, 4, 1], [1, 2, 2, 1], [1, 4, 4, 1]]
                choice_idx = np.random.choice(list(range(len(pool_stride_choices))))
                curr_pool_stride = pool_stride_choices[choice_idx]

            else:
                curr_pool_stride = config.TF_ANG_STRIDES[op]

            fc_h = ceil(fc_h * 1.0 / curr_pool_stride[1])
            fc_w = ceil(fc_w * 1.0 / curr_pool_stride[2])

            # if current output width and height is less than 2, we use a stride of 1
            if fc_h <= 2 or fc_w <= 2:
                curr_pool_stride = [1, 1, 1, 1]
                # if the fc_h or fc_w goes below 2 we need to reset fc_h and fc_w to
                # previously found whatever value. Because we are using a stride of 1 here
                fc_h = prev_fc_h
                fc_w = prev_fc_w
            else:
                prev_fc_h = fc_h
                prev_fc_w = fc_w
            config.TF_ANG_STRIDES[op] = curr_pool_stride

        if 'fc' in op:

            if 'fc1'==op:
                print('fcH: ',fc_h,' fcW: ',fc_w)
                if np.random.random()<0.5:
                    curr_fc = np.random.choice([100,200,300])
                else:
                    curr_fc = config.TF_ANG_VAR_SHAPES_DETACHED[op][-1]

                config.TF_ANG_VAR_SHAPES_DETACHED[op] = [fc_h*fc_w*previous_selected_depth,curr_fc]
                print('FC1 in size: %d'%(fc_h*fc_w*previous_selected_depth))
            else:
                raise NotImplementedError

        if 'out' in op:

            config.TF_ANG_VAR_SHAPES_DETACHED[op] = [curr_fc,1]


def make_dict_with_hyperparameters():
    hyps = {}
    hyps['batch_size'] = config.BATCH_SIZE
    hyps['start_lr'] = config.START_LR
    hyps['in_dropout'] = config.IN_DROPOUT
    hyps['layer_dropout'] = config.LAYER_DROPOUT
    hyps['l2_beta'] = config.L2_BETA
    hyps['var_shapes'] = dict(config.TF_ANG_VAR_SHAPES_DETACHED)
    hyps['strides'] = dict(config.TF_ANG_STRIDES)
    return hyps


def pretty_print_hyperparameters(logger):

    logger.info('='*40)
    logger.info('Batch Size: %d',config.BATCH_SIZE)
    logger.info('L2 Beta: %.5f',config.L2_BETA)
    logger.info('Input dropout: %.5f',config.IN_DROPOUT)
    logger.info('Layer dropout: %.5f',config.LAYER_DROPOUT)
    logger.info('Learning rate: %.5f',config.START_LR)
    logger.info('='*20)
    logger.info(config.TF_ANG_VAR_SHAPES_DETACHED)
    logger.info('=' * 20)
    logger.info(config.TF_ANG_SCOPES)
    logger.info('=' * 20)
    logger.info(config.TF_ANG_STRIDES)
    logger.info('=' * 20)
    logger.info('='*40)


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

    if not os.path.exists(IMG_DIR + os.sep + 'random_search'):
        os.mkdir(IMG_DIR + os.sep + 'random_search')

    random_search_epochs = 50
    epochs_per_search = 100

    logger = logging.getLogger('RandSearchLogger')
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(IMG_DIR + os.sep + 'random_search' + os.sep + 'main.log', mode='w')
    fileHandler.setFormatter(logging.Formatter(logging_format))
    fileHandler.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(fileHandler)

    dataset_filenames, dataset_sizes = dataset_name_factory.new_get_noncol_train_data_sorted_by_direction_noncol_test_data()

    config.setup_user_dependent_hyperparameters(no_pooling=True,square_input=False)

    all_hyps, all_tr_accuracies = [],[]

    graph = tf.Graph()
    configp = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(graph=graph, config=configp)

    cnn_learner_detached.activation = config.ACTIVATION
    cnn_learner_detached.kernel_size_dict = config.TF_ANG_VAR_SHAPES_DETACHED
    cnn_learner_detached.stride_dict = config.TF_ANG_STRIDES
    cnn_learner_detached.scope_list = config.TF_ANG_SCOPES
    cnn_learner_detached.IMG_DIR = IMG_DIR

    cnn_learner_detached.setup_loggers()

    print(config.TF_ANG_STRIDES)
    for r_ep in range(random_search_epochs):
        logger.info('='*80)
        # first run we use the default
        if r_ep>0:
            sample_set_of_hyperparameters()

            activation = config.ACTIVATION
            kernel_size_dict = config.TF_ANG_VAR_SHAPES_DETACHED
            stride_dict = config.TF_ANG_STRIDES
            scope_list = config.TF_ANG_SCOPES

            tf.reset_default_graph()
            graph = tf.Graph()
            configp = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.InteractiveSession(graph=graph, config=configp)

        ep_hyp_dict = make_dict_with_hyperparameters()
        pretty_print_hyperparameters(logger)

        with sess.as_default() and graph.as_default():
            cnn_variable_initializer.set_from_main(sess)
            cnn_variable_initializer.build_tensorflw_variables_detached()
            models_utils.set_from_main(sess, logger)

            tf_train_img_ids, tf_train_images, tf_train_labels = cnn_learner_detached.define_input_pipeline(dataset_filenames)
            tf_test_img_ids, tf_test_images, tf_test_labels = models_utils.build_input_pipeline(
                dataset_filenames['test_dataset'], config.BATCH_SIZE, shuffle=True,
                training_data=False, use_opposite_label=False, inputs_for_sdae=False)

            cnn_learner_detached.define_tf_ops(tf_train_images, tf_train_labels, tf_test_images, tf_test_labels)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            tf.global_variables_initializer().run(session=sess)

            dataset_filenames, dataset_sizes = dataset_name_factory.new_get_noncol_train_data_sorted_by_direction_noncol_test_data()

        for loc_ep in range(epochs_per_search):

            for step in range(dataset_sizes['train_dataset']//config.BATCH_SIZE):
                l1, _ = sess.run([cnn_learner_detached.tf_loss, cnn_learner_detached.tf_optimize],
                                 feed_dict=cnn_learner_detached.get_dropout_placeholder_dict())

            avg_train_accuracy = []
            for step in range(dataset_sizes['train_dataset'] // config.BATCH_SIZE):
                train_predictions,train_actuals = sess.run([cnn_learner_detached.tf_train_predictions,tf_train_labels])
                avg_train_accuracy.append(models_utils.soft_accuracy(
                    train_predictions, train_actuals, use_argmin=False,
                    max_thresh=cnn_learner_detached.max_thresh,min_thresh=cnn_learner_detached.min_thresh)
                )

            logger.info('\t\t Training accuracy: %.3f' % np.mean(avg_train_accuracy))

        sess.close()

        all_hyps.append(ep_hyp_dict)
        all_tr_accuracies.append(np.mean(avg_train_accuracy))

    logger.info('='*100)
    logger.info('WINNER')
    logger.info(all_hyps[np.argmax(all_tr_accuracies)])
    logger.info(all_tr_accuracies[np.argmax(all_tr_accuracies)])
    logger.info('='*100)