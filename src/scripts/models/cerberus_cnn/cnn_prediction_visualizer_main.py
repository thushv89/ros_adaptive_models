import cnn_model_visualizer
import tensorflow as tf
import config
import os
import models_utils
import cnn_learner_multiple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def get_images_predictions_labels_3_way(main_dir,env):


    test_sub_dir = 'data-equal'
    test_dataset_filename = 'data-chunk-0.tfrecords'

    dataset_filenames_dict = {
        'test_dataset': ['..' + os.sep + env + os.sep + test_sub_dir + os.sep + test_dataset_filename],
        'test_bump_dataset': ['..' + os.sep + env + os.sep + test_sub_dir + os.sep + test_dataset_filename]}
    batch_size = 10

    configp = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(config=configp)

    tf_test_img_ids, tf_test_images, tf_test_labels = \
        models_utils.build_input_pipeline(dataset_filenames_dict['test_dataset'], batch_size,
                                          shuffle=False, training_data=False, use_opposite_label=False,
                                          inputs_for_sdae=False, rand_valid_direction_for_bump=False)
    tf_bump_test_img_ids, tf_bump_test_images, tf_bump_test_labels = models_utils.build_input_pipeline(
        dataset_filenames_dict['test_bump_dataset'], batch_size, shuffle=False,
        training_data=False, use_opposite_label=True, inputs_for_sdae=False, rand_valid_direction_for_bump=False)

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
        print('=' * 80)
        print([v.name for v in tf.global_variables()])
        print('=' * 80)

        tf_predictions = cnn_learner_multiple.predictions_with_inputs(tf_test_images)

        all_test_images, all_predictoins, all_labels = None,None,None
        for step in range(50):
            np_test_images, predictions, actuals = sess.run([tf_test_images,tf_predictions,tf_test_labels])
            if all_test_images is None and all_predictoins is None:
                all_test_images = np.asarray(normalize_image_batch(np_test_images))
                all_predictoins = np.asarray(predictions)
                all_labels = np.asarray(actuals)
            else:
                all_test_images = np.append(all_test_images,normalize_image_batch(np_test_images),axis=0)
                all_predictoins = np.append(all_predictoins,predictions,axis=0)
                all_labels = np.append(all_labels, actuals,axis=0)

    print('Got data of following sizes')
    print(all_test_images.shape)
    print(all_predictoins.shape)
    print(all_labels.shape)
    return all_test_images,all_predictoins,all_labels

def normalize_image_batch(img_batch):

    norm_img_batch = img_batch - np.reshape(np.min(img_batch,axis=(1,2,3)),(-1,1,1,1))
    norm_img_batch = norm_img_batch / np.reshape(np.max(norm_img_batch,axis=(1,2,3)),(-1,1,1,1))
    return norm_img_batch

def select_correct_classification_results_from_all(images, predictions, labels, max_thresh,select_rand):
    sorted_data=[]
    for di,direction in enumerate(config.TF_DIRECTION_LABELS):
        print('Current labels')
        print(labels[:10])
        dir_indices = np.where(np.argmax(labels,axis=1)==di)[0]
        print('found indices (%s): %s'%(direction,dir_indices))
        dir_images = images[dir_indices,:,:,:]
        dir_predictions = predictions[dir_indices,:]
        dir_labels = labels[dir_indices,:]

        print('Data size for direction %s',direction)
        print(dir_images.shape)

        correct_ids_hard = np.where(np.argmax(dir_predictions,axis=1)==np.argmax(dir_labels,axis=1))[0]
        reduced_predictions_to_non_zero_labels = dir_predictions[np.arange(dir_labels.shape[0]),np.argmax(dir_labels,axis=1)]
        correct_ids_soft = np.where(reduced_predictions_to_non_zero_labels>max_thresh)[0]

        print(correct_ids_hard)
        print(correct_ids_soft)

        correct_ids = np.intersect1d(correct_ids_soft,correct_ids_hard)
        np.random.shuffle(correct_ids)
        selected_correct_ids = correct_ids[:select_rand]

        sorted_data.append({'images':dir_images[selected_correct_ids,:,:,:],
                                  'predictions':dir_predictions[selected_correct_ids,:],
                                  'labels':dir_labels[selected_correct_ids,:]
                                  })

    return sorted_data

def select_wrong_classification_results_from_all(images, predictions, labels, max_thresh,select_rand):
    sorted_data = []
    for di, direction in enumerate(config.TF_DIRECTION_LABELS):

        dir_indices = np.where(np.argmax(labels,axis=1)==di)[0]
        dir_images = images[dir_indices,:,:,:]
        dir_predictions = predictions[dir_indices,:]
        dir_labels = labels[dir_indices,:]

        wrong_ids_hard = np.where(np.argmax(dir_predictions,axis=1) != np.argmax(dir_labels,axis=1))[0]
        reduced_predictions_to_non_zero_labels = dir_predictions[np.arange(dir_labels.shape[0]), np.argmax(dir_labels, axis=1)]
        wrong_ids_soft = np.where(reduced_predictions_to_non_zero_labels < max_thresh)[0]

        print(wrong_ids_hard)
        print(wrong_ids_soft)

        wrong_ids = np.intersect1d(wrong_ids_soft, wrong_ids_hard)
        np.random.shuffle(wrong_ids)
        selected_wrong_ids = wrong_ids[:select_rand]
        sorted_data.append({'images':dir_images[selected_wrong_ids, :, :, :],
                                  'predictions':dir_predictions[selected_wrong_ids, :],
                                  'labels':dir_labels[selected_wrong_ids, :]
                                  })
    return sorted_data


def save_to_fig_classification_results(correct_images_dict, incorrect_images,main_dir):

    nrows_correct,nrows_wrong, ncols = 3, 2, 3
    nrows=nrows_correct+nrows_wrong
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    fig = plt.figure(1)
    padding = 0.0
    fig_w = ncols * 1.0 + (ncols + 1) * padding
    fig_h = nrows * 1.0 + (nrows + 1) * padding
    fig.set_size_inches(fig_w, fig_h)

    gs0 = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.25,height_ratios=[0.8,0.5])

    gs00 = gridspec.GridSpecFromSubplotSpec(nrows_correct, ncols, wspace=0.1, hspace=0.05, subplot_spec=gs0[0])
    gs01 = gridspec.GridSpecFromSubplotSpec(nrows_wrong, ncols, wspace=0.1, hspace=0.05, subplot_spec=gs0[1])


    #ax = [[None for _ in range(ncols)] for _ in range(nrows)]
    for ci in range(ncols):
        for ri in range(nrows_correct):
            ax = plt.subplot(gs00[ri * ncols + ci])
            ax.axis('off')
            ax.imshow(correct_images_dict[ci][ri, :, :, :], aspect='auto')

    for ci in range(ncols):
        for ri in range(nrows_wrong):
            ax = plt.subplot(gs01[ri * ncols + ci])
            ax.axis('off')
            ax.imshow(incorrect_images[ci][ri, :, :, :], aspect='auto')

    # Setting up title for correctly classified
    ax = plt.subplot(gs00[1])
    ax.set_title('Correctly Classified (Ind-2)\nGS')
    ax = plt.subplot(gs00[0])
    ax.set_title('TL')
    ax = plt.subplot(gs00[2])
    ax.set_title('TR')

    # Setting up title for correctly classified
    ax = plt.subplot(gs01[1])
    ax.set_title('Mis-Classified (Ind-2)')

    fig.subplots_adjust(bottom=0.01, top=0.9, right=0.99, left=0.01)
    fig.savefig(main_dir+os.sep+'classification_results.jpg')
    plt.cla()
    plt.close(fig)


if __name__=='__main__':

    #env = 'apartment-my3-2000'
    env = 'indoor-1-my1-2000'
    main_dir = 'test-multiple-more-env-recommended' + os.sep + env

    all_images,all_preds,all_labels = get_images_predictions_labels_3_way(main_dir,env)
    correct_data = select_correct_classification_results_from_all(all_images,all_preds,all_labels,0.6,3)
    incorrect_data = select_wrong_classification_results_from_all(all_images,all_preds,all_labels,0.6,2)
    correct_images = []
    for item in correct_data:
        correct_images.append(item['images'])
    incorrect_images = []
    for item in incorrect_data:
        incorrect_images.append(item['images'])
    save_to_fig_classification_results(correct_images,incorrect_images,main_dir)