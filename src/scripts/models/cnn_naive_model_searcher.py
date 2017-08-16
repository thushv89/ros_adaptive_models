import cnn_learner_naive
import tensorflow as tf
import numpy as np
import config
import os
import models_utils
import logging
import sys

logger = logging.getLogger('ModelSearcherLogger')
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter('[%(name)s] [%(funcName)s] %(message)s'))
console.setLevel(logging.DEBUG)
logger.addHandler(console)


speclogger = logging.getLogger('CNNSpecificationLogger')
speclogger.setLevel(logging.INFO)
specfileHandler = logging.FileHandler('model_search.log', mode='w')
specfileHandler.setFormatter(logging.Formatter('%(message)s'))
specfileHandler.setLevel(logging.INFO)
speclogger.addHandler(specfileHandler)


filter_h_and_w_space = [[2,4],[4,8],[4,4],[6,6]]
strides_space = [[1,1,1,1],[1,1,2,1],[1,2,2,1],[1,2,4,1]]
depths_space = [32,48,64]
fulcon_space = [64,128,256]

'''filter_h_and_w_space = [[4,8],[4,4]]
strides_space = [[1,1,2,1],[1,2,2,1]]
depths_space = [32,48]
fulcon_space = [64,128]'''

num_epochs = 50
training_fraction = 3

min_last_2d = [6,6]

to_try_scope_list = ['conv1','pool1','conv2','pool2','conv3','pool3','conv4','pool4','conv5','pool5']
max_2d_layer_count = len(to_try_scope_list)


sel_scope_list = []
sel_filter_hw_dict = {}
sel_strid_dict = {}

col_accuracy_dict = {}
noncol_accuracy_dict = {}


def build_tensorflw_variables_naive(scope_list,k_dict):
    '''
    Build the required tensorflow variables to populate the VGG-16 model
    All are initialized with zeros (initialization doesn't matter in this case as we assign exact values later)
    :param variable_shapes: Shapes of the variables
    :return:
    '''
    global logger,sess,graph

    logger.info("Building Tensorflow Variables (Tensorflow)...")

    for si,scope in enumerate(scope_list):
        with tf.variable_scope(scope) as sc:

            # Try Except because if you try get_variable with an intializer and
            # the variable exists, you will get a ValueError saying the variable exists

            try:
                if 'pool' not in scope:
                    tf.get_variable(config.TF_WEIGHTS_STR,shape=k_dict[scope],
                                              initializer=tf.contrib.layers.xavier_initializer())
                    tf.get_variable(config.TF_BIAS_STR, k_dict[scope][-1],
                                           initializer = tf.constant_initializer(0.001,dtype=tf.float32))

            except ValueError as e:
                logger.critical(e)
                logger.debug('Variables in scope %s already initialized\n'%scope)

        print([v.name for v in tf.global_variables()])


tf_loss, tf_bump_loss = None,None
tf_optimize, tf_bump_optimize, tf_test_predictions, tf_bump_test_predictions = None,None,None,None
inc_noncol_gstep, inc_col_gstep = None, None
tf_test_labels, tf_bump_test_labels = None, None

dataset_filenames = {
        'train_dataset': ['..' + os.sep + 'data_indoor_1_1000' + os.sep + 'image-direction-shuffled.tfrecords'],
        'train_bump_dataset': [
            '..' + os.sep + 'data_indoor_1_bump_200' + os.sep + 'image-direction-shuffled.tfrecords'],
        'test_dataset': ['..' + os.sep + 'data_grande_salle_1000' + os.sep + 'image-direction-shuffled.tfrecords'],
        'test_bump_dataset': [
            '..' + os.sep + 'data_grande_salle_bump_200' + os.sep + 'image-direction-shuffled.tfrecords']
    }

dataset_sizes = {'train_dataset': 1000 + 1000,
                 'train_bump_dataset': 400,
                 'test_dataset': 1000,
                 'test_bump_dataset': 200}


def define_cnn_ops():
    global inc_noncol_gstep,inc_col_gstep
    global tf_optimize, tf_bump_optimize, tf_test_predictions, tf_bump_test_predictions
    global tf_loss,tf_bump_loss
    global tf_test_labels, tf_bump_test_labels

    noncol_global_step = tf.Variable(0, trainable=False)
    col_global_step = tf.Variable(0, trainable=False)

    inc_noncol_gstep = cnn_learner_naive.inc_gstep(noncol_global_step)
    inc_col_gstep = cnn_learner_naive.inc_gstep(col_global_step)

    tf_img_ids, tf_images, tf_labels = models_utils.build_input_pipeline(
        dataset_filenames['train_dataset'], config.BATCH_SIZE, shuffle=True,
        training_data=True, use_opposite_label=False, inputs_for_sdae=False)

    tf_bump_img_ids, tf_bump_images, tf_bump_labels = models_utils.build_input_pipeline(
        dataset_filenames['train_bump_dataset'], config.BATCH_SIZE, shuffle=True,
        training_data=True, use_opposite_label=True, inputs_for_sdae=False)

    tf_logits = cnn_learner_naive.logits(tf_images)
    tf_loss = cnn_learner_naive.calculate_loss(tf_logits, tf_labels)

    tf_bump_logits = cnn_learner_naive.logits(tf_bump_images)
    tf_bump_loss = cnn_learner_naive.calculate_loss(tf_bump_logits, tf_bump_labels)

    tf_optimize, tf_grads_and_vars = cnn_learner_naive.cnn_optimizer.optimize_model_naive_no_momentum(tf_loss, noncol_global_step,
                                                                                    tf.global_variables())
    tf_bump_optimize, _ = cnn_learner_naive.cnn_optimizer.optimize_model_naive_no_momentum(tf_bump_loss, col_global_step,
                                                                         tf.global_variables())

    tf_test_img_ids, tf_test_images, tf_test_labels = models_utils.build_input_pipeline(dataset_filenames['test_dataset'],
                                                                                        config.BATCH_SIZE,
                                                                                        shuffle=False,
                                                                                        training_data=False,
                                                                                        use_opposite_label=False,
                                                                                        inputs_for_sdae=False)
    tf_bump_test_img_ids, tf_bump_test_images, tf_bump_test_labels = models_utils.build_input_pipeline(
        dataset_filenames['test_bump_dataset'], config.BATCH_SIZE, shuffle=False,
        training_data=False, use_opposite_label=True, inputs_for_sdae=False)

    tf_test_predictions = cnn_learner_naive.predictions_with_inputs(tf_test_images)
    tf_bump_test_predictions = cnn_learner_naive.predictions_with_inputs(tf_bump_test_images)


sess = None
def train_cnn():
    global sess

    configp = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.InteractiveSession(config=configp)

    with sess.as_default():

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf.global_variables_initializer().run(session=sess)

        max_noncol_accuracy, max_col_accuracy = 0,0
        noncol_acc_drop_count, col_acc_drop_count = 0,0
        acc_drop_threshold = 3
        min_col_loss, min_noncol_loss = 10000,10000
        col_exceed_min_count, noncol_exceed_min_count = 0,0

        for epoch in range(num_epochs):
            logger.info('\n')
            logger.info('\tRunning epoch %d',epoch)
            avg_loss,avg_bump_loss = [],[]
            test_accuracy, test_bump_accuracy = [],[]

            # ------------------------------------------------------------
            # train the net
            for step in range(dataset_sizes['train_dataset'] // config.BATCH_SIZE//training_fraction):
                if np.random.random() < 0.7:
                    l1, _, = sess.run([tf_loss, tf_optimize])
                    avg_loss.append(l1)
                else:
                    bump_l1, _, = sess.run([tf_bump_loss, tf_bump_optimize])
                    avg_bump_loss.append(bump_l1)
            # -----------------------------------------------------------

            # ------------------------------------------------------------
            # learning rate decay
            if min_noncol_loss > np.mean(avg_loss):
                min_noncol_loss = np.mean(avg_loss)
            else:
                noncol_exceed_min_count += 1
                logger.info('Increase noncol_exceed to %d',noncol_exceed_min_count)

            # if count exceeds threshold drop the learning rate
            if noncol_exceed_min_count >=acc_drop_threshold:
                logger.info('Stepping down collision learning rate')
                sess.run(inc_col_gstep)
                noncol_exceed_min_count = 0

            if min_col_loss > np.mean(avg_bump_loss):
                min_col_loss = np.mean(avg_bump_loss)
            else:
                col_exceed_min_count += 1
                logger.info('Increase col_exceed to %d',col_exceed_min_count)

            # if count exceeds threshold drop the learning rate
            if col_exceed_min_count >= acc_drop_threshold:
                logger.info('Stepping down non-collision learning rate')
                sess.run(inc_noncol_gstep)
                col_exceed_min_count = 0
            # ----------------------------------------------------------

            # -----------------------------------------------------------
            # Testing phase
            if (epoch+1) % 5 == 0:
                for step in range(dataset_sizes['test_dataset'] // config.BATCH_SIZE):
                    predicted_labels, actual_labels = sess.run([tf_test_predictions, tf_test_labels])
                    test_accuracy.append(models_utils.accuracy(predicted_labels, actual_labels, use_argmin=False))

                for step in range(dataset_sizes['test_bump_dataset'] // config.BATCH_SIZE):
                    bump_predicted_labels, bump_actual_labels = sess.run([tf_bump_test_predictions, tf_bump_test_labels])
                    test_bump_accuracy.append(models_utils.accuracy(bump_predicted_labels, bump_actual_labels, use_argmin=True))

                avg_test_accuracy = np.mean(test_accuracy)
                avg_bump_test_accuracy = np.mean(test_bump_accuracy)
                logger.info('\tAverage accuracys: %.3f (non-collision) %.3f (collision)', avg_test_accuracy, avg_bump_test_accuracy)
            # ------------------------------------------------------------

            # ------------------------------------------------------------
            # Checking test accuracy if it is not increasing break the loop
                if avg_bump_test_accuracy > max_col_accuracy:
                    max_col_accuracy = avg_bump_test_accuracy
                else:
                    col_acc_drop_count += 1

                if avg_test_accuracy > max_noncol_accuracy:
                    max_noncol_accuracy = avg_test_accuracy
                else:
                    noncol_acc_drop_count += 1

                if col_acc_drop_count >= acc_drop_threshold and noncol_acc_drop_count >= acc_drop_threshold:
                    break
            # ----------------------------------------------------------------

        logger.debug('\t Variable count: %d', len(tf.global_variables()))
        logger.debug('\t Operation count: %d', len(sess.graph.get_operations()))

        coord.request_stop()
        coord.join(threads)
        sess.close()

        return max_noncol_accuracy, max_col_accuracy


def choose_best_layer_choice(lyr_choice):
    '''
    Choose the best specification from a bunch of choices for a layer
    :param lyr_choice: choice dict with corresponding accuracies
    :return:
    '''
    max_accuracy = 0
    sel_spec = None
    for spec,accuracy in lyr_choice:
        if accuracy > max_accuracy:
            sel_spec = spec

    return sel_spec


if __name__ == '__main__':
    global sess

    max_accuracy_sofar = 0
    in_depth = 0
    last_conv_layer = None # used for fc calculation and pool layer doesn't have depth

    # make cnns of size 1,2,..., max_2d_layer_count
    for lyr_idx,scope in enumerate(to_try_scope_list):
        logger.info('Trying out CNNs of %d layers',lyr_idx)
        logger.debug('\t Previous scope: %s', sel_scope_list)
        logger.debug('\t Previous kernel: %s', sel_filter_hw_dict)
        logger.debug('\t Previous stride: %s', sel_strid_dict)
        if lyr_idx == 0:
            sel_scope_list.append(scope)
        else:
            del sel_scope_list[-1]
            del sel_scope_list[-1]
            sel_scope_list.append(scope)

        # add fc layer in the 1st conv exploration
        sel_scope_list.extend(['fc1','out'])

        logger.info('\t Current scope: %s',sel_scope_list)
        lyr_choices = []
        lyr_choices_0 = [] # use only for 0th layer
        for filter_hw in filter_h_and_w_space:
            for stride in strides_space:
                for depth in depths_space:

                    if lyr_idx==0:
                        in_depth = 3

                    if 'conv' in scope:
                        kernel_size = filter_hw + [in_depth,depth]
                        last_conv_layer = scope
                    elif 'pool' in scope:
                        kernel_size = [1] + filter_hw + [1]

                    logger.info('\n')
                    logger.info('\t\t Trying specs: %s (kernel), %s (stride)',kernel_size,stride)

                    # Temporarily add current kernel and strid to dicts
                    sel_filter_hw_dict[scope] = kernel_size
                    sel_strid_dict[scope] = stride

                    # deciding the fulcon size first and freeze that for layers above
                    if lyr_idx == 0:
                        logger.info('Jointly seaching conv1 and fulcon spaces ...')
                        for fc_size in fulcon_space:

                            # updating fc_h and fc_w as it changes with the stride of convolution layers below
                            fc_h,fc_w = models_utils.get_fc_height_width(config.TF_INPUT_AFTER_RESIZE, sel_scope_list, sel_strid_dict)
                            logger.debug('\t\t Obtained height and with for fulcon : %d, %d',fc_h,fc_w)
                            logger.debug('\t\t selelcted scope %s and strides: %s',sel_scope_list,sel_strid_dict)
                            sel_filter_hw_dict['fc1'] = [fc_h*fc_w*depth,fc_size]
                            sel_filter_hw_dict['out'] = [fc_size, config.TF_NUM_CLASSES]

                            logger.info('\n')
                            logger.info('\t\t Trying specs: %d (fc_size), %s (fc_kernel), %s (out kernel)', fc_size, sel_filter_hw_dict['fc1'],sel_filter_hw_dict['out'])

                            tf.reset_default_graph()
                            build_tensorflw_variables_naive(sel_scope_list, sel_filter_hw_dict)
                            logger.info('\t\t Variable creation successful')

                            cnn_learner_naive.set_scope_kernel_stride_dicts(sel_scope_list, sel_filter_hw_dict,
                                                                            sel_strid_dict)

                            define_cnn_ops()
                            logger.info('\t\t Defining tensorflow ops successful')
                            max_noncol_acc, max_col_acc = train_cnn()
                            logger.info('\t\t Accuracy measures: %.3f (noncol), %.3f (col)', max_noncol_acc,
                                        max_col_acc)
                            logger.info('\n')

                            avg_accuracy = (max_noncol_acc + max_col_acc) / 2.0

                            if avg_accuracy > max_accuracy_sofar:
                                max_accuracy_sofar = avg_accuracy
                                lyr_choices_0.append(
                                    [{'fc_size': fc_size}, (max_col_acc + max_noncol_acc) / 2.0])
                                lyr_choices.append(
                                    [{'kernel': kernel_size, 'stride': stride}, (max_col_acc + max_noncol_acc) / 2.0])
                                speclogger.info('Best choice (FC)')

                            speclogger.info('Layer index %d',lyr_idx)
                            speclogger.info(sel_scope_list)
                            speclogger.info(sel_filter_hw_dict)
                            speclogger.info(sel_strid_dict)
                            speclogger.info('Accuracy measures: %.3f (noncol), %.3f (col)', max_noncol_acc,
                                        max_col_acc)
                            speclogger.info('\n')

                        # if we have a better choice of FC
                        if len(lyr_choices_0) > 0:
                            logger.info('Choosing the best specs for conv1 + fulcon')
                            logger.info(lyr_choices_0)
                            selected_spec = choose_best_layer_choice(lyr_choices_0)
                            logger.info('Best selected spec: %s',selected_spec)
                            sel_filter_hw_dict['fc1'] = [fc_h*fc_w*depth,selected_spec['fc_size']]
                            sel_filter_hw_dict['out'] = [selected_spec['fc_size'],config.TF_NUM_CLASSES]
                            logger.info('Best FC selected %s',sel_filter_hw_dict['fc1'])

                            speclogger.info('(BEST) Layer index %d', lyr_idx)
                            speclogger.info(sel_scope_list)
                            speclogger.info(sel_filter_hw_dict)
                            speclogger.info(sel_strid_dict)
                            speclogger.info('Accuracy measures: %.3f (noncol), %.3f (col)', max_noncol_acc,
                                            max_col_acc)
                            speclogger.info('\n')
                        else:
                            # this should not happen
                            raise NotImplementedError
                    else:

                        logger.info('Calculating FC size...\n')
                        logger.debug('scope: %s',sel_scope_list)
                        logger.debug('filter: %s', sel_filter_hw_dict)
                        logger.debug('stride: %s',sel_strid_dict)
                        logger.debug('last_depth: %d',sel_filter_hw_dict[last_conv_layer][-1])

                        # everytime we try a new configuration, we need to modify the fc layer input size
                        fc_h, fc_w = models_utils.get_fc_height_width(config.TF_INPUT_AFTER_RESIZE, sel_scope_list,
                                                                      sel_strid_dict)
                        sel_filter_hw_dict['fc1'][0] = fc_h * fc_w * sel_filter_hw_dict[last_conv_layer][-1]
                        logger.debug('fc_h: %d, fc_w: %d', fc_h, fc_w)
                        logger.debug('New fulcon layer input size: %d', sel_filter_hw_dict['fc1'][0])

                        tf.reset_default_graph()
                        build_tensorflw_variables_naive(sel_scope_list,sel_filter_hw_dict)
                        logger.info('\t\t Variable creation successful')

                        cnn_learner_naive.set_scope_kernel_stride_dicts(sel_scope_list,sel_filter_hw_dict,sel_strid_dict)

                        define_cnn_ops()
                        logger.info('\t\t Defining tensorflow ops successful')
                        max_noncol_acc, max_col_acc = train_cnn()
                        logger.info('\t\t\t Accuracy measures: %.3f (noncol), %.3f (col)',max_noncol_acc, max_col_acc)

                        avg_accuracy = (max_noncol_acc+max_col_acc)/2.0

                        if avg_accuracy > max_accuracy_sofar:
                            max_accuracy_sofar = avg_accuracy
                            lyr_choices.append([{'kernel':kernel_size,'stride':stride},(max_col_acc+max_noncol_acc)/2.0])
                            logger.debug('\t\t\t Found best layer choice. Current layer choices')
                            logger.debug('\t\t\t %s',lyr_choices)
                            logger.info('\n')
                            speclogger.info('Current Best')

                        speclogger.info('Layer index %d', lyr_idx)
                        speclogger.info(sel_scope_list)
                        speclogger.info(sel_filter_hw_dict)
                        speclogger.info(sel_strid_dict)
                        speclogger.info('Accuracy measures: %.3f (noncol), %.3f (col)', max_noncol_acc,
                                        max_col_acc)
                        speclogger.info('Avg Accuracy measures: %.3f\n', (max_noncol_acc + max_col_acc) / 2.0)

                    logger.info('Removing temoporarily added scope')
                    # Remove current kernel and strid from dicts
                    del sel_filter_hw_dict[scope]
                    del sel_strid_dict[scope]

                    # no point in traversing in the most inner loop as the pooling layer doesn't change with depth
                    if 'pool' in scope:
                        break

        # if we have some choice that acctually increase our maximum performance
        if len(lyr_choices)>0:
            selected_spec = choose_best_layer_choice(lyr_choices)
            sel_filter_hw_dict[scope] = selected_spec['kernel']
            sel_strid_dict[scope] = selected_spec['stride']
            if 'conv' in scope:
                in_depth = selected_spec['kernel'][-1]

            speclogger.info('(BEST) Layer index %d', lyr_idx)
            speclogger.info(sel_scope_list)
            speclogger.info(sel_filter_hw_dict)
            speclogger.info(sel_strid_dict)
            speclogger.info('Accuracy measures: %.3f (noncol), %.3f (col)', max_noncol_acc,
                            max_col_acc)
            speclogger.info('Avg Accuracy measures: %.3f\n', (max_noncol_acc + max_col_acc)/2.0)


        else:
            del sel_scope_list[-1] # out
            del sel_scope_list[-1] # fc1
            del sel_scope_list[-1] # last layer

            # if last layer did not improve performance update last_conv_layer
            for tmp_scope in reversed(sel_scope_list):
                if 'conv' in tmp_scope:
                    last_conv_layer = tmp_scope
                    break

            sel_scope_list.extend(['fc1','out'])
            logger.info('Adding scope %s did not help',scope)
            speclogger.info('Adding scope %s did not help\n', scope)

        logger.info('\n')
        logger.info('Current Summary')
        logger.info(sel_scope_list)
        logger.info(sel_filter_hw_dict)
        logger.info(sel_strid_dict)
        logger.info(lyr_choices)

        logger.info('\n')

        # to check if the last 2d size is too small
        fc_h, fc_w = models_utils.get_fc_height_width(config.TF_INPUT_AFTER_RESIZE, sel_scope_list, sel_strid_dict)
        if fc_h < min_last_2d[0] or fc_w < min_last_2d[1]:
            break

    logger.info('\n')
    logger.info('THE BEST MODEL')
    logger.info(sel_scope_list)
    logger.info(sel_filter_hw_dict)
    logger.info(sel_strid_dict)