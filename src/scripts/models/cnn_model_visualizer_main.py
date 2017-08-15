import tensorflow as tf
import cnn_model_visualizer
import models_utils
import os
import config

if __name__ == '__main__':

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


    configp = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.InteractiveSession(config=configp)

    tf_test_img_ids, tf_test_images, tf_test_labels = models_utils.build_input_pipeline(
        dataset_filenames['test_dataset'], 1, shuffle=False, training_data=False, use_opposite_label=False,
        inputs_for_sdae=False)
    tf_bump_test_img_ids, tf_bump_test_images, tf_bump_test_labels = models_utils.build_input_pipeline(
        dataset_filenames['test_bump_dataset'], 1, shuffle=False,
        training_data=False, use_opposite_label=True, inputs_for_sdae=False)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    main_dir = 'test-single-naive-visualize'
    weights_filepath = main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + 'cnn_model_149.ckpt'
    hyperparam_filepath = main_dir + os.sep + config.WEIGHT_SAVE_DIR + os.sep + 'hyperparams_149.pickle'

    #saver = tf.train.import_meta_graph(weights_filepath+'.meta')

    if config.ACTIVATION_MAP_DIR and not os.path.exists(main_dir + os.sep + config.ACTIVATION_MAP_DIR):
        os.mkdir(main_dir + os.sep + config.ACTIVATION_MAP_DIR)

    with sess.as_default():

        cnn_model_visualizer.cnn_create_variables_naive_with_scope_size_stride(hyperparam_filepath)
        saver = tf.train.Saver()
        saver.restore(sess, weights_filepath)

        print([v.name for v in tf.global_variables()])
        v_list = sess.run(tf.global_variables())

        tf_activation_dict = cnn_model_visualizer.cnn_visualize_activations_naive(main_dir, sess, weights_filepath, hyperparam_filepath, tf_test_images,1)
        tf_bump_activation_dict = cnn_model_visualizer.cnn_visualize_activations_naive(main_dir, sess, weights_filepath,
                                                                                  hyperparam_filepath, tf_bump_test_images,
                                                                                  1)

        for step in range(10):
            print('Step %d'%step)
            test_img_id,test_image,activation_dict = sess.run([tf_test_img_ids,tf_test_images,tf_activation_dict])
            test_bump_img_id, test_bump_image, bump_activation_dict = sess.run([tf_bump_test_img_ids, tf_bump_test_images, tf_bump_activation_dict])
            cnn_model_visualizer.cnn_store_activations_as_image(activation_dict,test_image,test_img_id, main_dir + os.sep + config.ACTIVATION_MAP_DIR + os.sep + 'cnn_model_149')
            cnn_model_visualizer.cnn_store_activations_as_image(bump_activation_dict, test_bump_image, test_bump_img_id,
                                                                main_dir + os.sep + config.ACTIVATION_MAP_DIR + os.sep + 'bump_cnn_model_149')

        coord.request_stop()
        coord.join(threads)
