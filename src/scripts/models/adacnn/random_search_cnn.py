import cnn_learner_detached
import numpy as np
import config
import models_utils
from math import ceil
import dataset_name_factory
import tensorflow as tf
import cnn_variable_initializer

scope_list = config.TF_ANG_SCOPES


def sample_set_of_hyperparameters():
    # TF_ANG_VAR_SHAPES_DETACHED = {'conv1': [4, 8, 3, 32], 'pool1': [1, 4, 8, 1], 'conv2': [5, 5, 32, 64],
    #                              'pool2': [1, 3, 3, 1],
    #                              'conv3': [3, 3, 64, 128], 'pool3': [1, 3, 3, 1], 'conv4': [3, 3, 128, 128],
    #                              'fc1': [fc_h * fc_w * 128, FC1_WEIGHTS_DETACHED],
    #                              'out': [FC1_WEIGHTS_DETACHED, 1]}


    if np.random.random()<0.5:
        config.START_LR = np.random.choice([0.001,0.0001,0.00001,0.0005,0.00005])

    if np.random.random()<0.5:
        config.IN_DROPOUT = np.random.choice([0.0, 0.1, 0.2, 0.5])

    if np.random.random()<0.5:
        config.LAYER_DROPOUT = np.random.choice([0.0, 0.1, 0.34, 0.5])

    if np.random.random()<0.5:
        config.L2_BETA = np.random.choice([0.0, 0.01, 0.001,0.0001,0.005, 0.0005])



    fc_h, fc_w = config.TF_INPUT_AFTER_RESIZE[0],config.TF_INPUT_AFTER_RESIZE[1]
    previous_selected_depth = 3
    curr_fc = -1

    for op in scope_list:
        if 'conv' in op:
            if np.random.random()<0.5:
                curr_conv_shape = np.random.choice([
                    [2, 4, previous_selected_depth, 32], [2, 4, previous_selected_depth, 64],
                    [2, 4, previous_selected_depth, 128],
                    [4, 8, previous_selected_depth, 32], [4, 8, previous_selected_depth, 64],
                    [4, 8, previous_selected_depth, 128],
                    [2, 2, previous_selected_depth, 32], [2, 2, previous_selected_depth, 64],
                    [2, 2, previous_selected_depth, 128],
                    [4, 4, previous_selected_depth, 32], [4, 4, previous_selected_depth, 64],
                    [4, 4, previous_selected_depth, 128]])

                config.TF_ANG_VAR_SHAPES_DETACHED[op] = curr_conv_shape

            if np.random.random()<0.5:
                curr_conv_stride = np.random.choice([[1,1,1,1], [1,2, 4,1], [1,2, 2,1],[1,4, 4,1]])

            else:
                curr_conv_stride = config.TF_ANG_STRIDES[op]

            fc_h = ceil(fc_h * 1.0 / curr_conv_stride[1])
            fc_w = ceil(fc_w * 1.0 / curr_conv_stride[2])

            # if current output width and height is less than 2, we use a stride of 1
            if fc_h < 2 or fc_w < 2:
                curr_conv_stride = [1, 1, 1, 1]

            config.TF_ANG_STRIDES[op] = curr_conv_stride

        if 'pool' in op:

            if np.random.random()<0.5:
                curr_pool_shape = np.random.choice([
                    [1,2, 4,1],
                    [1, 4, 8, 1],
                    [1, 2, 2, 1],
                    [1, 4, 4, 1]
                ])
                config.TF_ANG_VAR_SHAPES_DETACHED[op] = curr_pool_shape

            if np.random.random() < 0.5:
                curr_pool_stride = np.random.choice([[1, 1, 1, 1], [1, 2, 4, 1], [1, 2, 2, 1], [1, 4, 4, 1]])
            else:
                curr_pool_stride = config.TF_ANG_STRIDES[op]

            fc_h = ceil(fc_h * 1.0 / curr_pool_stride[1])
            fc_w = ceil(fc_w * 1.0 / curr_pool_stride[2])

            # if current output width and height is less than 2, we use a stride of 1
            if fc_h < 2 or fc_w < 2:
                curr_pool_stride = [1, 1, 1, 1]

            config.TF_ANG_STRIDES[op] = curr_pool_stride

        if 'fc' in op:

            if 'fc1'==op:

                if np.random.random()<0.5:
                    curr_fc = np.random.choice([100,200,300])
                else:
                    curr_fc = config.TF_ANG_VAR_SHAPES_DETACHED[op]

                config.TF_ANG_VAR_SHAPES_DETACHED[op] = [fc_h*fc_w*curr_fc]

        if 'out' in op:

            config.TF_ANG_VAR_SHAPES_DETACHED[op] = [curr_fc,1]


if __name__ == '__main__':

    random_search_epochs = 25
    epochs_per_search = 10

    for r_ep in range(random_search_epochs):
        # first run we use the default
        if r_ep>0:
            sample_set_of_hyperparameters()

            graph = tf.Graph()
            configp = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.InteractiveSession(graph=graph, config=configp)

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

            activation = config.ACTIVATION
            kernel_size_dict = config.TF_ANG_VAR_SHAPES_DETACHED
            stride_dict = config.TF_ANG_STRIDES
            scope_list = config.TF_ANG_SCOPES

            avg_train_accuracy = []
            for step in range(dataset_sizes['train_dataset'] // config.BATCH_SIZE):
                train_predictions,train_actuals = sess.run([cnn_learner_detached.tf_train_predictions,tf_train_labels])
                avg_train_accuracy.append(models_utils.soft_accuracy(
                    train_predictions, train_actuals, use_argmin=False,
                    max_thresh=cnn_learner_detached.max_thresh,min_thresh=cnn_learner_detached.min_thresh)
                )

                if step < 2:
                    logger.debug('Predictions for Non-Collided data')
                    for pred, lbl in zip(train_predictions, train_actuals):
                        logger.debug('\t%s;%s', pred, lbl)

            logger.info('\t\t Training accuracy: %.3f' % np.mean(avg_train_accuracy))

