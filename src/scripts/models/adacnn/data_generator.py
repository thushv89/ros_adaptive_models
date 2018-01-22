import tensorflow as tf
import h5py
import numpy as np
import config


class DataGenerator(object):

    def __init__(self, _batch_size, _n_labels, _data_sizes, _input_size, _session, _dataset_fnames):

        self.batch_size = _batch_size
        self.n_labels = _n_labels
        self.data_sizes = _data_sizes
        self.input_size = _input_size
        self.tf_img_id_ph = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)
        self.tf_image_ph = tf.placeholder(shape=[self.batch_size] + self.input_size, dtype=tf.float32)
        self.tf_label_ph = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)

        self.session = _session

        self.dataset_idx = [0 for _ in range(len(_dataset_fnames))]

        self.img_ids, self.images, self.labels = [],[],[]

        for f in _dataset_fnames:
            dataset_file = h5py.File(f,"r")
            self.img_ids.append(dataset_file['/image_ids'])
            self.images.append(dataset_file['/images'])
            self.labels.append(dataset_file['/labels'])


    def tf_augment_data_with(self):

        tf_image_batch = self.tf_image_ph

        tf_image_batch = tf.random_crop(tf_image_batch,[self.batch_size,self.resize_to,self.resize_to,self.n_channels],seed=13423905832)

        tf_image_batch = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), tf_image_batch)

        tf_image_batch = tf.image.random_brightness(tf_image_batch,0.5)
        tf_image_batch = tf.image.random_contrast(tf_image_batch,0.5,1.5)

        label_batch = tf.one_hot(self.tf_label_ph, config.TF_NUM_CLASSES, dtype=tf.float32, on_value=1.0, off_value=0.0)

        # standardize image
        tf_image_batch = tf.image.per_image_standardization(tf_image_batch)

        flip_rand = tf.random_uniform(shape=[1], minval=0, maxval=1.0, seed=154324654)
        image_batch = tf.cond(flip_rand[0] > 0.5, lambda: tf.reverse(image_batch, axis=[2]), lambda: image_batch)
        label_batch = tf.cond(flip_rand[0] > 0.5, lambda: tf.reverse(label_batch, axis=[1]), lambda: label_batch)

        return self.tf_img_id_ph, tf_image_batch, label_batch


    def sample_a_batch_from_data(self, env_id, shuffle):

        if shuffle:
            ind_to_sample = np.random.randint(0,self.data_sizes[env_id],size=(self.batch_size))
        else:
            if (self.dataset_idx[env_id]+self.batch_size)<self.data_sizes[env_id]:
                ind_to_sample = np.arange(self.dataset_idx[env_id], self.dataset_idx[env_id]+self.batch_size)
                self.dataset_idx[env_id] = self.dataset_idx[env_id]+self.batch_size
            else:
                ind_to_sample = np.concatenate(
                    [np.arange(self.dataset_idx[env_id],self.data_sizes[env_id]),
                     np.arange(0,(self.dataset_idx[env_id]+self.batch_size)%self.data_sizes[env_id])]
                )
                self.dataset_idx[env_id] = (self.dataset_idx[env_id] + self.batch_size)%self.data_size[env_id]

        return self.img_ids[env_id][ind_to_sample], self.images[env_id][ind_to_sample, :, :, :], self.labels[env_id][ind_to_sample, 0]