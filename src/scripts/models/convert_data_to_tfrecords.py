from PIL import Image
import numpy as np
import tensorflow as tf
import os
import config
'''
======================== convert_data_to_tfrecords ===============================
This script will take the data in a given folder (.png images, log file) and create tf records
'''

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def dump_to_tfrecord(data_folder, drive_direct_dict, image_ids,image_fname_prefix, max_instances_per_file):
    """
    Converts a dataset to tfrecords.
    :param data_folder:
    :param name:
    :return:
    """
    #print(os.getcwd()) # use to get the current working directory

    tfrecords_filenames = []
    writers = []
    items_for_direction = [0 for _ in range(3)]
    file_indices = [0 for _ in range(3)]

    for di in range(3):
        tfrecords_filenames.append(data_folder + os.sep + 'image-direction-%d-%d.tfrecords'%(file_indices[di],di))
        writers.append(tf.python_io.TFRecordWriter(tfrecords_filenames[di]))

    for img_id in image_ids:
        direction = int(drive_direct_dict[img_id])
        im = Image.open(data_folder+os.sep+ image_fname_prefix + '_%d.png'%img_id)
        im_mat = np.array(im,dtype=np.float32)
        (rows,cols,ch) = im_mat.shape
        im_raw = im_mat.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            config.FEAT_IMG_HEIGHT: _int64_feature(rows),
            config.FEAT_IMG_WIDTH: _int64_feature(cols),
            config.FEAT_IMG_CH: _int64_feature(ch),
            config.FEAT_IMG_RAW: _bytes_feature(im_raw),
            config.FEAT_LABEL: _int64_feature(drive_direct_dict[img_id])
        }))
        writers[direction].write(example.SerializeToString())
        items_for_direction[direction] += 1

        if items_for_direction[direction]>=max_instances_per_file:
            print('Items fore %d direction exceeds max'%direction)
            items_for_direction[direction] = 0
            file_indices[direction] += 1
            writers[direction].close()
            tfrecords_filenames[direction] = (data_folder + os.sep + 'image-direction-%d-%d.tfrecords' % (file_indices[direction], direction))
            writers[direction]=tf.python_io.TFRecordWriter(tfrecords_filenames[direction])

    for di in range(3):
        writers[di].close()

if __name__ == '__main__':
    is_bump_data = False
    data_folder = '.'+os.sep+ '..'+ os.sep+'sample-with-dir-1'

    angle_dict = {}
    direction_dict = {}
    img_indices = []

    drive_angle_filename = config.DRIVE_ANGLE_LOG if not is_bump_data else config.BUMP_DRIVE_ANGLE_LOG
    try:
        with open(data_folder+os.sep+drive_angle_filename) as f:
            f = f.readlines()
            for line in f:
                txt_tokens = line.split(':')
                angle_dict[int(txt_tokens[0])] = float(txt_tokens[1])
                img_indices.append(int(txt_tokens[0]))
                direction_dict[int(txt_tokens[0])] = int(txt_tokens[2])
    except FileNotFoundError as e:
        print(e)

    if not is_bump_data:
        data_indices = range(0, 500)
        image_fname_prefix = 'img'
    else:
        data_indices = img_indices
        image_fname_prefix = 'bump_img'

    #dump_to_tfrecord(data_folder,direction_dict,data_indices,image_fname_prefix,100)

    record_iterator = tf.python_io.tf_record_iterator(path=data_folder + os.sep + 'image-direction-0-0.tfrecords')

    record_count = 0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = example.features.feature['height']
        width = example.features.feature['width']
        img = np.fromstring(example.features.feature['image_raw'].bytes_list.value[0],dtype=np.float32)
        #print(len(example.features.feature['image_raw'].bytes_list.value[0]))
        print(example.features.feature[config.FEAT_LABEL].int64_list.value[0])

        record_count += 1
    print(record_count)
