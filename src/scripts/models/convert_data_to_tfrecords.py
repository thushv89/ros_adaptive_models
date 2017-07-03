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


file_index = 0
def dump_to_tfrecord(data_folder, drive_direct_dict, data_range,image_fname_prefix):
    """
    Converts a dataset to tfrecords.
    :param data_folder:
    :param name:
    :return:
    """
    global file_index
    #print(os.getcwd()) # use to get the current working directory
    tfrecords_filename = data_folder + os.sep + 'image-direction-%d.tfrecords'%file_index
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for di in data_range:
        im = Image.open(data_folder+os.sep+ image_fname_prefix + '_%d.png'%di)
        im_mat = np.array(im,dtype=np.float32)
        (rows,cols,ch) = im_mat.shape
        im_raw = im_mat.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            config.FEAT_IMG_HEIGHT: _int64_feature(rows),
            config.FEAT_IMG_WIDTH: _int64_feature(cols),
            config.FEAT_IMG_CH: _int64_feature(ch),
            config.FEAT_IMG_RAW: _bytes_feature(im_raw),
            config.FEAT_LABEL: _int64_feature(drive_direct_dict[di])
        }))
        writer.write(example.SerializeToString())
    file_index += 1
    writer.close()

if __name__ == '__main__':
    is_bump_data = True
    data_folder = '.'+os.sep+ '..'+ os.sep+'sample-with-dir-3-bump'

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
        data_indices = [range(0, 100), range(100, 200), range(200, 300), range(300, 400), range(400, 500)]
        image_fname_prefix = 'img'
    else:
        data_indices = [img_indices]
        image_fname_prefix = 'bump_img'
    for di in data_indices:
        dump_to_tfrecord(data_folder,direction_dict,di,image_fname_prefix)

    '''record_iterator = tf.python_io.tf_record_iterator(path=data_folder + os.sep + 'image_angle-0.tfrecords')

    record_count = 0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = example.features.feature['height']
        width = example.features.feature['width']
        img = np.fromstring(example.features.feature['image_raw'].bytes_list.value[0],dtype=np.float32)
        print(len(example.features.feature['image_raw'].bytes_list.value[0]))
        print(img.shape)
        print(128*96*3)
        record_count += 1
    print(record_count)'''
