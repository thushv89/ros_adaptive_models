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


def dump_to_tfrecord_suffled(data_folder, drive_direct_dict, image_ids,image_fname_prefix, max_instances_per_file):
    """
    Converts a dataset to tfrecords.
    :param data_folder:
    :param name:
    :return:
    """
    #print(os.getcwd()) # use to get the current working directory
    print('Running shuffled save')
    items_written = 0


    # create 3 tf records each for each direction

    tfrecords_filename = data_folder + os.sep + 'image-direction-shuffled.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    # create example for each image write with the writer
    np.random.shuffle(image_ids)

    for img_id in image_ids:
        im = Image.open(data_folder+os.sep+ image_fname_prefix + '_%d.png'%img_id)
        im_mat = np.array(im,dtype=np.float32)
        (rows,cols,ch) = im_mat.shape
        im_raw = im_mat.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            config.FEAT_IMG_ID: _int64_feature(img_id),
            config.FEAT_IMG_HEIGHT: _int64_feature(rows),
            config.FEAT_IMG_WIDTH: _int64_feature(cols),
            config.FEAT_IMG_CH: _int64_feature(ch),
            config.FEAT_IMG_RAW: _bytes_feature(im_raw),
            config.FEAT_LABEL: _int64_feature(drive_direct_dict[img_id])
        }))
        writer.write(example.SerializeToString())
        items_written += 1


    writer.close()

def dump_to_tfrecord_in_chunks(data_folder, save_dir, drive_direct_dict, image_ids,image_fname_prefix, max_instances_per_file):
    """
    Converts a dataset to tfrecords. Write several tfrecords by breaking the dataset into number of chunks
    :param data_folder:
    :param name:
    :return:
    """
    #print(os.getcwd()) # use to get the current working directory
    print('Running shuffled save')
    items_written_per_chunk = 0

    chunk_index = 0

    tfrecords_filename = save_dir + os.sep + 'data-chunk-%d.tfrecords'%chunk_index
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for img_id in image_ids:
        im = Image.open(data_folder+os.sep+ image_fname_prefix + '_%d.png'%img_id)
        im_mat = np.array(im,dtype=np.float32)
        (rows,cols,ch) = im_mat.shape
        im_raw = im_mat.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            config.FEAT_IMG_ID: _int64_feature(img_id),
            config.FEAT_IMG_HEIGHT: _int64_feature(rows),
            config.FEAT_IMG_WIDTH: _int64_feature(cols),
            config.FEAT_IMG_CH: _int64_feature(ch),
            config.FEAT_IMG_RAW: _bytes_feature(im_raw),
            config.FEAT_LABEL: _int64_feature(drive_direct_dict[img_id])
        }))
        writer.write(example.SerializeToString())
        items_written_per_chunk += 1

        if items_written_per_chunk>=max_instances_per_file:
            writer.close()
            items_written_per_chunk = 0
            chunk_index += 1
            tfrecords_filename = save_dir + os.sep + 'data-chunk-%d.tfrecords' % chunk_index
            writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    writer.close()

def dump_to_tfrecord_sorted_by_direction(data_folder, drive_direct_dict, image_ids,image_fname_prefix, max_instances_per_file):
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

    # create 3 tf records each for each direction
    for di in range(3):
        tfrecords_filenames.append(data_folder + os.sep + 'image-direction-%d-%d.tfrecords'%(file_indices[di],di))
        writers.append(tf.python_io.TFRecordWriter(tfrecords_filenames[di]))

    # create example for each image write with the writer
    for img_id in image_ids:
        direction = int(drive_direct_dict[img_id])
        im = Image.open(data_folder+os.sep+ image_fname_prefix + '_%d.png'%img_id)
        im_mat = np.array(im,dtype=np.float32)
        (rows,cols,ch) = im_mat.shape
        im_raw = im_mat.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            config.FEAT_IMG_ID: _int64_feature(img_id),
            config.FEAT_IMG_HEIGHT: _int64_feature(rows),
            config.FEAT_IMG_WIDTH: _int64_feature(cols),
            config.FEAT_IMG_CH: _int64_feature(ch),
            config.FEAT_IMG_RAW: _bytes_feature(im_raw),
            config.FEAT_LABEL: _int64_feature(drive_direct_dict[img_id])
        }))
        writers[direction].write(example.SerializeToString())
        items_for_direction[direction] += 1

        # update tf record filename, writer
        if items_for_direction[direction]>=max_instances_per_file:
            print('Items for %d direction exceeds max'%direction)
            items_for_direction[direction] = 0
            file_indices[direction] += 1
            writers[direction].close()
            tfrecords_filenames[direction] = (data_folder + os.sep + 'image-direction-%d-%d.tfrecords' % (file_indices[direction], direction))
            writers[direction]=tf.python_io.TFRecordWriter(tfrecords_filenames[direction])

    for di in range(3):
        writers[di].close()

if __name__ == '__main__':
    is_bump_list = [False, True, False, True, False, True]

    data_folders_list = ['.'+os.sep+ '..'+ os.sep+'data_indoor_1_1000', '.'+os.sep+ '..'+ os.sep+'data_indoor_1_bump_200',
                    '.' + os.sep + '..' + os.sep + 'data_sandbox_1000',
                    '.' + os.sep + '..' + os.sep + 'data_sandbox_bump_200',
                    '.' + os.sep + '..' + os.sep + 'data_grande_salle_1000',
                    '.' + os.sep + '..' + os.sep + 'data_grande_salle_bump_200']



    for is_bump_data,data_folder in zip(is_bump_list,data_folders_list):

        direction_frequency_stats = [0 for _ in range(3)]
        save_dir = data_folder + os.sep + 'data-chunks-chronological'

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

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
                    direction_frequency_stats[int(txt_tokens[2])] += 1
        except FileNotFoundError as e:
            print(e)

        print(data_folder)
        print('Right','Straight','Left')
        print(direction_frequency_stats)
        print()
        continue
        if not is_bump_data:
            data_indices = img_indices
            image_fname_prefix = 'img'
            max_instances_per_file = 250
        else:
            data_indices = img_indices
            image_fname_prefix = 'bump_img'
            max_instances_per_file = 50

        dump_to_tfrecord_in_chunks(data_folder, save_dir, direction_dict,data_indices,image_fname_prefix,max_instances_per_file=max_instances_per_file)

        '''record_iterator = tf.python_io.tf_record_iterator(path=data_folder + os.sep + 'image-direction-0-0.tfrecords')

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
        print(record_count)'''
