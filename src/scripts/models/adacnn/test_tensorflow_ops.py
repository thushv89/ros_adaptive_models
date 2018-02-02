import config
import data_generator
import dataset_name_factory
import tensorflow as tf
import numpy as np
from PIL import Image
import os

def test_if_images_after_preprocessing_are_fine():
    graph = tf.Graph()
    config  = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(graph=graph, config=config)

    dataset_filenames, dataset_sizes = dataset_name_factory.new_get_noncol_train_data_sorted_by_direction_noncol_test_data()

    train_data_gen = data_generator.DataGenerator(
        config.BATCH_SIZE, config.TF_NUM_CLASSES, dataset_sizes['train_dataset'],
        config.TF_INPUT_SIZE, sess, dataset_filenames['train_dataset'], config.TF_INPUT_AFTER_RESIZE, False
    )

    test_data_gen = data_generator.DataGenerator(
        config.BATCH_SIZE, config.TF_NUM_CLASSES, dataset_sizes['test_dataset'],
        config.TF_INPUT_SIZE, sess, dataset_filenames['test_dataset'], config.TF_INPUT_AFTER_RESIZE, True
    )

    tf_train_img_ids, tf_train_images, tf_train_labels = train_data_gen.tf_augment_data_with()
    tf_test_img_ids, tf_test_images, tf_test_labels = test_data_gen.tf_augment_data_with()

    for env_idx in range(4):

        tr_img_id, tr_images, tr_labels = train_data_gen.sample_a_batch_from_data(env_idx, shuffle=True)
        ts_img_id, ts_images, ts_labels = test_data_gen.sample_a_batch_from_data(env_idx, shuffle=False)

        save_batch_of_data('train_env_%d'%env_idx,tr_images, tr_labels)
        save_batch_of_data('test_env_%d' % env_idx, ts_images, ts_labels)


img_save_id = 0

def save_batch_of_data(dir_name, images,labels):
    global img_save_id

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for img, lbl in zip(images, labels):
        print(img.shape)

        print(np.min(np.min(img,axis=0),axis=0))
        img -= np.min(np.min(img,axis=0),axis=0).reshape(1,1,-1)
        img /= np.max(np.max(img,axis=0),axis=0).reshape(1,1,-1)
        img *= img*128.0
        img = img.astype(np.uint8)

        im = Image.fromarray(img)
        im.save(dir_name + os.sep + '%d_img_%d.png'%(int(lbl),img_save_id))
        img_save_id += 1


if __name__=='__main__':

    test_if_images_after_preprocessing_are_fine()