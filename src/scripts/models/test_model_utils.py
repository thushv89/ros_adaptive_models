import models_utils
import tensorflow as tf
import numpy as np
import os
from scipy.misc import imsave
import logging

def test_multiclass_precision_should_return_approximately_0_dot_7_for_each():
    test_labels = np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],
                   [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                   [0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.float32)

    test_predictions = np.array([[0.5,0,0],[0.5,0,0],[0.5,0,0],[0,0.5,0],[0,0,0.5],
                        [0,0.5,0],[0,0.5,0],[0,0.5,0],[0,0.5,0],[0,0.5,0],
                        [0,0,0.5],[0,0,0.5],[0,0,0.5],[0.5,0,0],[0,0.5,0]],dtype=np.float32)

    prec = models_utils.precision_multiclass(test_predictions,test_labels, False)
    print(prec)


def test_if_tensorflow_use_two_batches_to_calculate_two_things_in_same_session_dot_run():
    session = tf.InteractiveSession()

    tf_images, tf_labels = models_utils.build_input_pipeline(
        ['..' + os.sep + 'data_indoor_1_1000' + os.sep + 'image-direction-0-0.tfrecords','..' + os.sep + 'data_indoor_1_1000' + os.sep + 'image-direction-0-2.tfrecords'], 10, shuffle=True,
        training_data=True, use_opposite_label=False, inputs_for_sdae=False)

    def get_reduce_mean(tf_labels):
        return tf.reduce_mean(tf_labels,axis=[0])

    def get_identity(tf_labels):
        return tf_labels

    tf_red_mean = get_reduce_mean(tf_labels)
    tf_identity = get_identity(tf_labels)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)

    for _ in range(10):
        red_mean, labels = session.run([tf_red_mean,tf_identity])

    coord.request_stop()
    coord.join(threads)

    print(red_mean)
    print(labels)


def test_if_input_pipeline_images_makes_sense_in_terms_of_direction_to_turn():
    # with random_flipping on

    session = tf.InteractiveSession()

    dir_to_save = 'test_if_inputs_correct'
    if dir_to_save and not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)

    tf_images, tf_labels = {},{}
    tf_bump_images, tf_bump_labels = {},{}

    for di,direct in enumerate(['left','straight','right']):
        _, tf_images[direct], tf_labels[direct] = models_utils.build_input_pipeline(
            ['..' + os.sep + 'data_indoor_1_1000' + os.sep + 'image-direction-0-%d.tfrecords'%di], 5, shuffle=True,
            training_data=True, use_opposite_label=False, inputs_for_sdae=False)

        _, tf_bump_images[direct], tf_bump_labels[direct] = models_utils.build_input_pipeline(
            ['..' + os.sep + 'data_indoor_1_bump_200' + os.sep + 'image-direction-0-%d.tfrecords' % di], 5, shuffle=True,
            training_data=True, use_opposite_label=True, inputs_for_sdae=False)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)

    img_indices = [0 for _ in range(3)]
    loggers = []
    for di, direct in enumerate(['left', 'straight', 'right']):
        dir_to_save_direct = 'test_if_inputs_correct' + os.sep + direct
        if dir_to_save_direct and not os.path.exists(dir_to_save_direct):
            os.mkdir(dir_to_save_direct)

        testLogger = logging.getLogger('Logger')
        testLogger.setLevel(logging.INFO)
        testFH = logging.FileHandler(dir_to_save_direct + os.sep + 'test_labels.log', mode='w')
        testFH.setFormatter(logging.Formatter(logging.INFO))
        testFH.setLevel(logging.INFO)
        testLogger.addHandler(testFH)
        loggers.append(testLogger)

        for _ in range(10):
            imgs, labels = session.run([tf_images[direct], tf_labels[direct]])
            bump_imgs, bump_labels = session.run([tf_bump_images[direct], tf_bump_labels[direct]])

            for img, lbl in zip(imgs,labels):
                filename = dir_to_save_direct + os.sep + 'test_img_%d.jpg'%img_indices[di]
                img = (img - np.min(img))
                img /= np.max(img)
                imsave(filename,img)

                #loggers[di].info('%d , %s'%(img_indices[di],lbl.tolist()))
                img_indices[di] += 1

    coord.request_stop()
    coord.join(threads)

if __name__=='__main__':
    #test_multiclass_precision_should_return_approximately_0_dot_7_for_each()
    #test_if_tensorflow_use_two_batches_to_calculate_two_things_in_same_session_dot_run()
    test_if_input_pipeline_images_makes_sense_in_terms_of_direction_to_turn()