import numpy as np
import tensorflow as tf
import cnn_model_visualizer
import models_utils
import os
import config
from PIL import Image
import pickle

def save_activation_maps_for_naive():
    main_dir = 'test-naive-full-recommended-architecture' + os.sep + 'apartment-my3-2000'

    non_col_image_filenames = ['%d.png' % i for i in range(1, 25)]
    col_image_filenames = ['%d.png' % i for i in range(1, 22)]
    collision_dir = 'selected_col_images'
    noncol_dir = 'selected_noncol_images'

    image_mat_list = []
    for img_fname in non_col_image_filenames:
        img_mat = Image.open(main_dir + os.sep + noncol_dir + os.sep + img_fname)
        image_mat_list.append(img_mat)

    for img_fname in col_image_filenames:
        img_mat = Image.open(main_dir + os.sep + collision_dir + os.sep + img_fname)
        image_mat_list.append(img_mat)

    configp = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(config=configp)

    tf_image_placeholder = tf.placeholder(dtype=tf.float32, shape=config.TF_INPUT_SIZE, name='image')
    image = tf.image.crop_to_bounding_box(tf_image_placeholder, 16, 0, 64, 128)
    image = tf.image.resize_images(image, [config.TF_INPUT_AFTER_RESIZE[0], config.TF_INPUT_AFTER_RESIZE[1]])

    image = tf.image.per_image_standardization(image)

    image_batched = tf.expand_dims(image, axis=0)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    weights_filepath = main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + 'cnn_model-final.ckpt'
    hyperparam_filepath = main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + 'hyperparams-final.pickle'

    print(pickle.load(open(hyperparam_filepath,'rb')))
    # saver = tf.train.import_meta_graph(weights_filepath+'.meta')

    if config.ACTIVATION_MAP_DIR and not os.path.exists(main_dir + os.sep + config.ACTIVATION_MAP_DIR):
        os.mkdir(main_dir + os.sep + config.ACTIVATION_MAP_DIR)

    with sess.as_default():

        cnn_model_visualizer.cnn_create_variables_naive_with_scope_size_stride(hyperparam_filepath)
        saver = tf.train.Saver()
        saver.restore(sess, weights_filepath)

        v_list = sess.run(tf.global_variables())
        print('Restored following variables')
        print('='*80)
        print([v.name for v in tf.global_variables()])
        print('='*80)
        tf_activation_dict = cnn_model_visualizer.cnn_visualize_activations_naive(main_dir, sess, weights_filepath,
                                                                                  hyperparam_filepath, image_batched,
                                                                                  1)

        img_id = 0
        for step,img_mat in enumerate(image_mat_list):
            print('Processing image %d' % step)
            activation_dict = sess.run(tf_activation_dict,feed_dict={tf_image_placeholder:img_mat})

            print(activation_dict.keys())
            cnn_model_visualizer.cnn_store_activations_as_image(
                activation_dict,['conv1','conv2','conv3','conv4'],np.expand_dims(img_mat,0),
                img_id, main_dir + os.sep + config.ACTIVATION_MAP_DIR + os.sep + 'cnn_activations','highest')
            img_id += 1

        coord.request_stop()
        coord.join(threads)


def save_activation_maps_for_multiple():
    main_dir = 'test-multiple-more-env-recommended' + os.sep + 'apartment-my3-2000'

    non_col_image_filenames = ['%d.png'%i for i in range(1,25)]
    col_image_filenames = ['%d.png'%i for i in range(1,22)]
    collision_dir = 'selected_col_images'
    noncol_dir = 'selected_noncol_images'

    image_mat_list = []
    for img_fname in non_col_image_filenames:
        img_mat = Image.open(main_dir + os.sep + noncol_dir + os.sep + img_fname)
        image_mat_list.append(img_mat)

    for img_fname in col_image_filenames:
        img_mat = Image.open(main_dir + os.sep + collision_dir + os.sep + img_fname)
        image_mat_list.append(img_mat)

    configp = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(config=configp)

    tf_image_placeholder = tf.placeholder(dtype=tf.float32,shape=config.TF_INPUT_SIZE,name='image')
    image = tf.image.crop_to_bounding_box(tf_image_placeholder, 16, 0, 64, 128)
    image = tf.image.resize_images(image, [config.TF_INPUT_AFTER_RESIZE[0], config.TF_INPUT_AFTER_RESIZE[1]])

    image = tf.image.per_image_standardization(image)

    image_batched = tf.expand_dims(image,axis=0)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    weights_filepath = main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + 'cnn-model-final.ckpt'
    hyperparam_filepath = main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + 'hyperparams-final.pickle'

    # saver = tf.train.import_meta_graph(weights_filepath+'.meta')

    if config.ACTIVATION_MAP_DIR and not os.path.exists(main_dir + os.sep + config.ACTIVATION_MAP_DIR):
        os.mkdir(main_dir + os.sep + config.ACTIVATION_MAP_DIR)

    with sess.as_default():

        cnn_model_visualizer.cnn_create_variables_multiple_with_scope_size_stride(hyperparam_filepath)
        saver = tf.train.Saver()
        print('Restoring weights from file: ', weights_filepath)
        saver.restore(sess, weights_filepath)

        v_list = sess.run(tf.global_variables())
        print('Restored following variables')
        print('='*80)
        print([v.name for v in tf.global_variables()])
        print('='*80)
        tf_activation_dict = cnn_model_visualizer.cnn_visualize_activations_multiple(main_dir, sess, weights_filepath,
                                                                                  hyperparam_filepath, image_batched,
                                                                                  1)

        img_id = 0
        for step,img_mat in enumerate(image_mat_list):
            print('Processing image %d' % step)
            activation_dict = sess.run(tf_activation_dict,feed_dict={tf_image_placeholder:img_mat})

            print(activation_dict.keys())
            cnn_model_visualizer.cnn_store_activations_as_image_heirarchy_for_cerberus(
                activation_dict,['conv1','conv2','conv3','conv4'],np.expand_dims(img_mat,0),
                img_id, main_dir + os.sep + config.ACTIVATION_MAP_DIR + os.sep + 'cnn_activations','highest')
            img_id += 1

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

    save_activation_maps_for_naive()
    #save_activation_maps_for_multiple()