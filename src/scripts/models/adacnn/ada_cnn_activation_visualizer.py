import pickle
import tensorflow as tf
import models_utils
import ada_cnn_model_saver
import os
import config
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import dataset_name_factory
import data_generator
import numpy as np
from math import ceil
import ada_cnn_constants as constants
import getopt

logger = logging.getLogger('CNNVisualizationLogger')
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter('[%(name)s] [%(funcName)s] %(message)s'))
console.setLevel(logging.INFO)
logger.addHandler(console)

def cnn_visualize_activations(hyperparam_dict, activation_image,batch_size):

    scope_list = hyperparam_dict['layers']
    print(scope_list)
    #activation = hyperparam_dict['activations']
    activation = config.ACTIVATION
    activation_dict = {}

    for scope in scope_list:

        mod_weight_string = config.TF_WEIGHTS_STR + ':0'
        mod_bias_string = config.TF_BIAS_STR + ':0'
        with tf.variable_scope(scope,reuse=True):
            if 'conv' in scope:
                with tf.variable_scope('best', reuse=True):
                    weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)
                    weight.set_shape(hyperparam_dict[scope]['weights'])
                    bias.set_shape([hyperparam_dict[scope]['weights'][-1]])

                    if scope=='conv_0':
                        logger.info('\t\tConvolution with %s activation for %s', activation, scope)
                        h = models_utils.activate(
                            tf.nn.conv2d(activation_image, weight, strides=hyperparam_dict[scope]['stride'], padding='SAME') + bias,
                            activation, name='hidden')
                    else:
                        logger.info('\t\tConvolution with %s activation for %s', activation, scope)
                        h = models_utils.activate(
                            tf.nn.conv2d(h, weight, strides=hyperparam_dict[scope]['stride'], padding='SAME') + bias,
                            activation, name='hidden')

                activation_dict[scope] = h

            elif 'pool' in scope:
                logger.info('\t\tMax pooling for %s', scope)
                h = tf.nn.max_pool(h, hyperparam_dict[scope]['weights'], hyperparam_dict[scope]['stride'],
                                   padding='SAME', name='pool_hidden')

            elif 'fulcon' in scope:

                # Reshaping required for the first fulcon layer
                #if scope == 'fulcon_out':
                #    logger.info('\t\tFully-connected with output Logits for %s', scope)
                #    h = tf.matmul(h, weight) + bias

                if scope == 'fulcon_0':
                    h_shape = h.get_shape().as_list()
                    logger.info('\t\t\tReshaping the input (of size %s) before feeding to %s', scope, h_shape)
                    h = tf.reshape(h, [batch_size, h_shape[1] * h_shape[2] * h_shape[3]])

                    for di in ['left','straight','right']:
                        with tf.variable_scope(di,reuse=True):
                            with tf.variable_scope('best', reuse = True):
                                weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(
                                    config.TF_BIAS_STR)

                                h_tmp = models_utils.activate(tf.matmul(h, weight) + bias, activation)
                                activation_dict[scope+'-'+di] = h_tmp


    return activation_dict


def save_activation_maps(main_dir,epoch):

    dataset_filenames, dataset_sizes = dataset_name_factory.new_get_noncol_train_data_sorted_by_direction_noncol_test_data()

    configp = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(config=configp)


    hyperparam_filepath = main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + 'cnn-hyperparameters-'+str(epoch)+'.pickle'
    weights_filepath = main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + 'cnn-model-'+str(epoch)+'.ckpt'

    with open(hyperparam_filepath,'rb') as f:
        hyperparam_dict = pickle.load(f)

    print(hyperparam_dict)

    # saver = tf.train.import_meta_graph(weights_filepath+'.meta')

    if config.ACTIVATION_MAP_DIR and not os.path.exists(main_dir + os.sep + config.ACTIVATION_MAP_DIR):
        os.mkdir(main_dir + os.sep + config.ACTIVATION_MAP_DIR)

    with sess.as_default():

        ada_cnn_model_saver.create_and_restore_cnn_weights(sess, hyperparam_filepath, weights_filepath, True)

        with tf.variable_scope(constants.TF_GLOBAL_SCOPE):

            train_data_gen = data_generator.DataGenerator(
                1, 3, dataset_sizes['train_dataset'],
                config.TF_INPUT_SIZE, sess, dataset_filenames['train_dataset'],
                config.TF_INPUT_AFTER_RESIZE, False
            )

            test_data_gen = data_generator.DataGenerator(
                1, 3, dataset_sizes['test_dataset'],
                config.TF_INPUT_SIZE, sess, dataset_filenames['test_dataset'],
                config.TF_INPUT_AFTER_RESIZE, True
            )

            print([v.name for v in tf.global_variables()])
            v_list = sess.run(tf.global_variables())

            tf_train_img_ids, tf_train_data_batch, tf_train_label_batch = train_data_gen.tf_augment_data_with(adjust_brightness_contrast=False)
            tf_test_img_ids , tf_test_dataset, tf_test_labels = test_data_gen.tf_augment_data_with(adjust_brightness_contrast=False)

            tf_train_activation_dict = cnn_visualize_activations(hyperparam_dict, tf_train_data_batch,1)
            tf_test_activation_dict = cnn_visualize_activations(hyperparam_dict,tf_test_dataset,1)

            for train_env_idx in range(3):
                for step in range(20):
                    tr_img_id, tr_images, tr_labels = train_data_gen.sample_a_batch_from_data(train_env_idx, shuffle=True)
                    ts_img_id, ts_images, ts_labels = test_data_gen.sample_a_batch_from_data(train_env_idx, shuffle=True)
                    print('Train env idx: %d, Step %d' %(train_env_idx,step))
                    cropped_train_images, train_activation_dict = sess.run([tf_train_data_batch,tf_train_activation_dict],feed_dict={train_data_gen.tf_image_ph:tr_images})
                    cropped_test_images, test_activation_dict = sess.run([tf_test_dataset,tf_test_activation_dict], feed_dict={test_data_gen.tf_image_ph: ts_images})

                    cnn_store_activations_as_image(train_env_idx,train_activation_dict, cropped_train_images, tr_img_id,
                                                                        main_dir + os.sep + config.ACTIVATION_MAP_DIR + os.sep + 'train_cnn_model')
                    cnn_store_activations_as_image(train_env_idx,test_activation_dict, cropped_test_images, ts_img_id,
                                                                        main_dir + os.sep + config.ACTIVATION_MAP_DIR + os.sep + 'test_cnn_model')


def cnn_store_activations_as_image(train_env_idx, activation_dict,orig_image,img_id, filename_prefix):

    number_of_cols = 10

    for k,v in activation_dict.items():
        tensor_depth = v.shape[-1]
        print(k)
        # depth+1 for the original image
        if 'conv' in k:
            fig, ax = plt.subplots(nrows=ceil((tensor_depth+1)*1.0/number_of_cols), ncols=number_of_cols)
        elif 'fulcon' in k:
            fig, ax = plt.subplots(nrows=1, ncols=2)

        if 'conv' in k:
            for ri in range(ceil((tensor_depth+1)*1.0/number_of_cols)):
                for ci in range(number_of_cols):
                    ax[ri, ci].axis('off')
                    if ri==0 and ci==0:
                        norm_img = orig_image[0,:,:,:] - np.min(orig_image[0,:,:,:])
                        norm_img = (norm_img / np.max(norm_img))
                        ax[ri, ci].imshow(norm_img)

                    else:
                        index = ri*number_of_cols + ci - 1
                        if index >= tensor_depth:
                            break

                        ax[ri,ci].imshow(v[0,:,:,index],aspect=None)

        if 'fulcon' in k:
            norm_img = orig_image[0, :, :, :] - np.min(orig_image[0, :, :, :])
            norm_img = (norm_img / np.max(norm_img))
            ax[0].imshow(norm_img)
            ax[1].imshow(np.tile(v,(2,1)),aspect=None,vmin=0.0,vmax=0.5)

        plt.subplots_adjust(wspace=0.05,hspace=0.05)
        fig.savefig(filename_prefix + '_activation_%d_%d_%s.jpg'%(train_env_idx,img_id,k))
        plt.cla()
        plt.close(fig)


if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "", ["output_dir=","epoch="])
    except getopt.GetoptError as err:
        print(err.with_traceback())
        print('<filename>.py --output_dir= --num_gpus= --memory=')

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '--output_dir':
                main_dir = arg
            if opt =='--epoch':
                epoch = int(arg)

    save_activation_maps(main_dir,epoch)