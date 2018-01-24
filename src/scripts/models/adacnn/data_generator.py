import tensorflow as tf
import h5py
import numpy as np
import config
import sys

class DataGenerator(object):

    def __init__(self, _batch_size, _n_labels, _data_sizes, _input_size, _session, _dataset_fnames, _input_size_resized, _is_testing_data):

        self.batch_size = _batch_size
        self.n_labels = _n_labels
        self.data_sizes = _data_sizes
        self.input_size = _input_size
        self.input_size_after_resize = _input_size_resized
        self.tf_img_id_ph = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)
        self.tf_image_ph = tf.placeholder(shape=[self.batch_size] + self.input_size, dtype=tf.float32)
        self.tf_label_ph = tf.placeholder(shape=[self.batch_size], dtype=tf.int32)

        self.session = _session

        self.dataset_idx = [0 for _ in range(len(_dataset_fnames))]
        self.is_testing_data = _is_testing_data

        self.img_ids, self.images, self.labels = [],[],[]

        for f in _dataset_fnames:
            dataset_file = h5py.File(f,"r")
            self.img_ids.append(dataset_file['/image_ids'])
            self.images.append(dataset_file['/images'])
            self.labels.append(dataset_file['/labels'])


    def tf_augment_data_with(self):

        tf_image_batch = self.tf_image_ph
        label_batch = tf.one_hot(self.tf_label_ph, config.TF_NUM_CLASSES, dtype=tf.float32, on_value=1.0, off_value=0.0)

        if not self.is_testing_data:
            tf_image_batch = tf.map_fn(lambda img: tf.random_crop(
                img, self.input_size_after_resize, seed=tf.set_random_seed(2334543)
            ), tf_image_batch)

            #Adjust contrast/brightness randomly
            tf_image_batch = tf.image.random_brightness(tf_image_batch, 0.2)
            tf_image_batch = tf.image.random_contrast(tf_image_batch, 0.5, 1.2)
            #tf_image_batch = tf.random_crop(tf_image_batch,[self.batch_size] + self.input_size_after_resize,seed=13423905832)
        else:
            tf_image_batch = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 16, 0, 64, 128), tf_image_batch)
            tf_image_batch = tf.image.adjust_contrast(tf_image_batch, 0.25)

        # standardize image
        tf_image_batch = tf.map_fn(lambda img: tf.image.per_image_standardization(img), tf_image_batch)

        return self.tf_img_id_ph, tf_image_batch, label_batch


    def sample_a_batch_from_data(self, env_id, shuffle):

        if shuffle:
            ind_to_sample = np.random.choice(np.arange(0,self.data_sizes[env_id]),size=(self.batch_size),replace=False).tolist()
        else:
            if (self.dataset_idx[env_id]+self.batch_size)<self.data_sizes[env_id]:
                ind_to_sample = np.arange(self.dataset_idx[env_id], self.dataset_idx[env_id]+self.batch_size).tolist()
                self.dataset_idx[env_id] = self.dataset_idx[env_id]+self.batch_size
            else:
                ind_to_sample = np.concatenate(
                    [np.arange(self.dataset_idx[env_id],self.data_sizes[env_id]),
                     np.arange(0,(self.dataset_idx[env_id]+self.batch_size)%self.data_sizes[env_id])]
                ).tolist()

                self.dataset_idx[env_id] = (self.dataset_idx[env_id] + self.batch_size)%self.data_sizes[env_id]
        ind_to_sample = sorted(ind_to_sample)

        return self.img_ids[env_id][ind_to_sample], self.images[env_id][ind_to_sample, :, :, :], self.labels[env_id][ind_to_sample, 0]