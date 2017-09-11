from PIL import Image
import numpy as np
import tensorflow as tf
import os
import config
import logging
import sys
import scipy.misc

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


def dump_to_tfrecord_suffled(data_folder, data_save_dir, drive_direct_dict, image_ids,image_fname_prefix):
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

    tfrecords_filename = data_save_dir + os.sep + 'image-shuffled.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    # create example for each image write with the writer
    np.random.shuffle(image_ids)

    file_size_stat_dict = {}
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

    file_size_stat_dict[tfrecords_filename]=items_written-1

    writer.close()

    return file_size_stat_dict

def dump_to_tfrecord_in_chunks(data_folder, save_dir, drive_direct_dict,
                               image_ids,image_fname_prefix, n_direction, max_instances_per_file=None,
                               shuffle=False, augment_data=False, save_images_for_testing=False):
    '''
    Converts a dataset to tfrecords. Write several tfrecords by breaking the dataset into number of chunks
    :param data_folder:
    :param save_dir:
    :param drive_direct_dict:
    :param image_ids:
    :param image_fname_prefix:
    :param max_instances_per_file:
    :param augment_data: Augment data by flipping left right
    :return:
    '''
    #print(os.getcwd()) # use to get the current working directory
    print('Running shuffled save')
    items_written_per_chunk = 0

    chunk_index = 0

    tfrecords_filename = save_dir + os.sep + 'data-chunk-%d.tfrecords'%chunk_index
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    file_size_stat_dict = {}
    if save_images_for_testing:
        for di in range(n_direction):
            direct_sub_dir =save_dir + os.sep + 'direction-'+str(di)
            if not os.path.exists(direct_sub_dir):
                os.mkdir(direct_sub_dir)

        for img_id in image_ids:
            selected_dir = save_dir + os.sep + 'direction-' + str(drive_direct_dict[img_id])
            im = Image.open(data_folder + os.sep + image_fname_prefix + '_%d.png' % img_id)
            im_mat = np.array(im, dtype=np.float32)
            scipy.misc.imsave(selected_dir+os.sep+'image-%d.jpg'%img_id, im_mat)

    if shuffle:
        np.random.shuffle(image_ids)
    for img_id in image_ids:
        im = Image.open(data_folder+os.sep+ image_fname_prefix + '_%d.png'%img_id)
        im_mat = np.array(im,dtype=np.float32)
        drive_direction = drive_direct_dict[img_id]

        (rows,cols,ch) = im_mat.shape
        im_raw = im_mat.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            config.FEAT_IMG_ID: _int64_feature(img_id),
            config.FEAT_IMG_HEIGHT: _int64_feature(rows),
            config.FEAT_IMG_WIDTH: _int64_feature(cols),
            config.FEAT_IMG_CH: _int64_feature(ch),
            config.FEAT_IMG_RAW: _bytes_feature(im_raw),
            config.FEAT_LABEL: _int64_feature(drive_direction)
        }))
        writer.write(example.SerializeToString())
        items_written_per_chunk += 1

        # save a flipped version of the image
        if augment_data:
            flip_direction = drive_direct_dict[img_id]
            im_mat_flip = np.fliplr(im_mat)
            if n_direction == 5:
                if drive_direct_dict[img_id] == 0:
                    flip_direction = 4
                elif drive_direct_dict[img_id] == 1:
                    flip_direction = 3
                elif drive_direct_dict[img_id] == 4:
                    flip_direction = 0
                elif drive_direct_dict[img_id] == 3:
                    flip_direction = 1

            elif n_direction == 3:
                if drive_direct_dict[img_id] == 0:
                    flip_direction = 2
                elif drive_direct_dict[img_id] == 2:
                    flip_direction = 0


            im_raw_flip = im_mat_flip.tostring()

            flip_example = tf.train.Example(features=tf.train.Features(feature={
                config.FEAT_IMG_ID: _int64_feature(img_id),
                config.FEAT_IMG_HEIGHT: _int64_feature(rows),
                config.FEAT_IMG_WIDTH: _int64_feature(cols),
                config.FEAT_IMG_CH: _int64_feature(ch),
                config.FEAT_IMG_RAW: _bytes_feature(im_raw_flip),
                config.FEAT_LABEL: _int64_feature(flip_direction)
            }))
            writer.write(flip_example.SerializeToString())
            items_written_per_chunk += 1

        if max_instances_per_file is not None and items_written_per_chunk>=max_instances_per_file:
            logger.info('image-shuffled-part-%d.tfrecords (Size): %d', chunk_index,
                        items_written_per_chunk)
            file_size_stat_dict['image-shuffled-part-%d.tfrecords (Size): %d'%(chunk_index,
                        items_written_per_chunk)] = items_written_per_chunk
            writer.close()
            items_written_per_chunk = 0
            chunk_index += 1
            tfrecords_filename = save_dir + os.sep + 'image-shuffled-part-%d.tfrecords' % chunk_index
            writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    logger.info('image-shuffled-part-%d.tfrecords (Size): %d', chunk_index,
                items_written_per_chunk)
    file_size_stat_dict['image-shuffled-part-%d.tfrecords (Size): %d' % (chunk_index,
                                                                         items_written_per_chunk)] = items_written_per_chunk
    writer.close()

    return file_size_stat_dict

def dump_to_tfrecord_sorted_by_direction(
        data_folder, save_dir, img_id_to_direction_dict, image_ids, image_fname_prefix,
        n_direction, max_instances_per_file, augment_data):

    """
    Converts a dataset to tfrecords. Seperate data by direction
    :param data_folder:
    :param name:
    :return:
    """
    #print(os.getcwd()) # use to get the current working directory

    file_size_stat_dict = {}
    tfrecords_filenames = []
    writers = []
    items_for_direction = [0 for _ in range(n_direction)]
    file_indices = [0 for _ in range(n_direction)]
    size_per_file = [0 for _ in range(n_direction)]
    # create 3 tf records each for each direction
    for di in range(n_direction):
        tfrecords_filenames.append(save_dir + os.sep + 'image-%d-part-%d.tfrecords'%(di, file_indices[di]))
        writers.append(tf.python_io.TFRecordWriter(tfrecords_filenames[di]))

    # create example for each image write with the writer
    for img_id in image_ids:
        direction = int(img_id_to_direction_dict[img_id])
        im = Image.open(data_folder + os.sep + image_fname_prefix + '_%d.png'%img_id)
        #print(image_fname_prefix + '_%d.png'%img_id)
        im_mat = np.asarray(im,dtype=np.float32)

        (rows,cols,ch) = im_mat.shape
        im_raw = im_mat.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            config.FEAT_IMG_ID: _int64_feature(img_id),
            config.FEAT_IMG_HEIGHT: _int64_feature(rows),
            config.FEAT_IMG_WIDTH: _int64_feature(cols),
            config.FEAT_IMG_CH: _int64_feature(ch),
            config.FEAT_IMG_RAW: _bytes_feature(im_raw),
            config.FEAT_LABEL: _int64_feature(img_id_to_direction_dict[img_id])
        }))
        writers[direction].write(example.SerializeToString())
        items_for_direction[direction] += 1
        size_per_file[direction] += 1

        # save a flipped version of the image
        if augment_data:
            flip_direction = img_id_to_direction_dict[img_id]
            im_mat_flip = np.fliplr(im_mat)

            if n_direction == 5:
                if img_id_to_direction_dict[img_id] == 0:
                    flip_direction = 4
                elif img_id_to_direction_dict[img_id] == 1:
                    flip_direction = 3
                elif img_id_to_direction_dict[img_id] == 4:
                    flip_direction = 0
                elif img_id_to_direction_dict[img_id] == 3:
                    flip_direction = 1

            elif n_direction == 3:
                if img_id_to_direction_dict[img_id] == 0:
                    flip_direction = 2
                elif img_id_to_direction_dict[img_id] == 2:
                    flip_direction = 0

            else:
                raise NotImplementedError

            im_raw_flip = im_mat_flip.tostring()

            flip_example = tf.train.Example(features=tf.train.Features(feature={
                config.FEAT_IMG_ID: _int64_feature(img_id),
                config.FEAT_IMG_HEIGHT: _int64_feature(rows),
                config.FEAT_IMG_WIDTH: _int64_feature(cols),
                config.FEAT_IMG_CH: _int64_feature(ch),
                config.FEAT_IMG_RAW: _bytes_feature(im_raw_flip),
                config.FEAT_LABEL: _int64_feature(flip_direction)
            }))
            writers[flip_direction].write(flip_example.SerializeToString())
            items_for_direction[flip_direction] += 1
            size_per_file[flip_direction] += 1

        # update tf record filename, writer
        if items_for_direction[direction]>=max_instances_per_file:
            print('Items for %d direction exceeds max'%direction)
            logger.info( 'image-%d-part-%d.tfrecords (Size): %d',direction, file_indices[direction],size_per_file[direction])
            file_size_stat_dict['image-%d-part-%d.tfrecords'%(direction, file_indices[direction])] = size_per_file[direction]

            items_for_direction[direction] = 0
            file_indices[direction] += 1
            writers[direction].close()
            tfrecords_filenames[direction] = (save_dir + os.sep + 'image-%d-part-%d.tfrecords' % (direction,file_indices[direction]))
            writers[direction]=tf.python_io.TFRecordWriter(tfrecords_filenames[direction])

            size_per_file[direction] = 0

        if augment_data and items_for_direction[flip_direction]>=max_instances_per_file:
            print('Items for %d direction exceeds max'%flip_direction)
            items_for_direction[flip_direction] = 0
            file_indices[flip_direction] += 1
            writers[flip_direction].close()
            tfrecords_filenames[flip_direction] = (save_dir + os.sep + 'image-%d-part-%d.tfrecords' % (flip_direction,file_indices[flip_direction]))
            writers[flip_direction]=tf.python_io.TFRecordWriter(tfrecords_filenames[flip_direction])
            logger.info( 'image-%d-part-%d.tfrecords (Size): %d',flip_direction, file_indices[flip_direction],size_per_file[flip_direction])
            size_per_file[flip_direction] = 0

    for di in range(3):
        logger.info('image-%d-part-%d.tfrecords (Size): %d', di, file_indices[di],
                    size_per_file[di])
        file_size_stat_dict['image-%d-part-%d.tfrecords'%(direction, file_indices[direction])] = size_per_file[di]
        writers[di].close()

    return file_size_stat_dict


def get_image_indices_with_uniform_distribution(direction_to_img_id_dict):
    global logger
    min_count = 100000000
    all_img_indices = []
    for k,v in direction_to_img_id_dict.items():
        if len(v) < min_count:
            min_count = len(v)

    for k,v in direction_to_img_id_dict.items():
        np.random.shuffle(v)
        all_img_indices.extend(v[:min_count])

    logger.info('Keeping %d samples from each directio (uniform)',min_count)
    return all_img_indices


def save_training_data():
    # Used as Training Data
    # ==========================================
    is_bump_list = [False, False, False, False, False, False, False, False]

    data_folders_list = [
        '.' + os.sep + '..' + os.sep + 'apartment-my1-2000',
        '.' + os.sep + '..' + os.sep + 'apartment-my2-2000',
        '.' + os.sep + '..' + os.sep + 'apartment-my3-2000',
        '.' + os.sep + '..' + os.sep + 'indoor-1-2000',
        '.' + os.sep + '..' + os.sep + 'indoor-1-my1-2000',
        '.' + os.sep + '..' + os.sep + 'grande_salle-my1-2000',
        '.' + os.sep + '..' + os.sep + 'grande_salle-my2-2000',
        '.' + os.sep + '..' + os.sep + 'sandbox-2000'
    ]

    assert len(is_bump_list) == len(data_folders_list), 'Bump List length and Data Folder lenght do not match'

    for is_bump_data, data_folder in zip(is_bump_list, data_folders_list):

        direction_frequency_stats = [0 for _ in range(3)]
        direction_save_dir = data_folder + os.sep + 'data-separated-by-direction-augmented'

        if not os.path.exists(direction_save_dir):
            os.mkdir(direction_save_dir)

        angle_dict = {}
        img_id_to_direction_dict = {}
        direction_to_img_id_dict = {0: [], 1: [], 2: []}
        img_indices = []

        drive_angle_filename = config.DRIVE_ANGLE_LOG if not is_bump_data else config.BUMP_DRIVE_ANGLE_LOG
        try:
            with open(data_folder + os.sep + drive_angle_filename) as f:
                f = f.readlines()
                for line in f:
                    txt_tokens = line.split(':')
                    angle_dict[int(txt_tokens[0])] = float(txt_tokens[1])
                    img_indices.append(int(txt_tokens[0]))
                    img_id_to_direction_dict[int(txt_tokens[0])] = int(txt_tokens[2])
                    direction_to_img_id_dict[int(txt_tokens[2])].append(int(txt_tokens[0]))
                    direction_frequency_stats[int(txt_tokens[2])] += 1
        except FileNotFoundError as e:
            print(e)

        print(data_folder)
        print('Right', 'Straight', 'Left')
        print(direction_frequency_stats)
        print()

        if not is_bump_data:
            image_fname_prefix = 'img'
            max_instances_per_file = 50
        else:
            image_fname_prefix = 'bump_img'
            max_instances_per_file = 50

        logger.info('=' * 80)
        logger.info(data_folder)
        logger.info('=' * 80)

        file_size_stat = dump_to_tfrecord_sorted_by_direction(data_folder, direction_save_dir, img_id_to_direction_dict, img_indices,
                                             image_fname_prefix, 3, max_instances_per_file, augment_data=True)

        with open(direction_save_dir+os.sep+'dataset_sizes.txt','w') as f:
            for k,v in file_size_stat.items():
                f.write(str(k)+':'+str(v))
                f.write('\n')


def save_training_data_5_way():
    # Used as Training Data
    # ==========================================
    is_bump_list = [False, False, False]

    data_folders_list = [
        '.' + os.sep + '..' + os.sep + 'indoor-1-5-way-3000',
        '.' + os.sep + '..' + os.sep + 'indoor-1-my1-5-way-3000',
        '.' + os.sep + '..' + os.sep + 'indoor-1-my2-5-way-3000',
    ]

    assert len(is_bump_list) == len(data_folders_list), 'Bump List length and Data Folder lenght do not match'

    for is_bump_data, data_folder in zip(is_bump_list, data_folders_list):

        direction_frequency_stats = [0 for _ in range(5)]
        direction_save_dir = data_folder + os.sep + 'data-separated-by-direction-augmented-5-way'

        if not os.path.exists(direction_save_dir):
            os.mkdir(direction_save_dir)

        angle_dict = {}
        img_id_to_direction_dict = {}
        direction_to_img_id_dict = {0: [], 1: [], 2: [], 3:[], 4:[]}
        img_indices = []

        drive_angle_filename = config.DRIVE_ANGLE_LOG if not is_bump_data else config.BUMP_DRIVE_ANGLE_LOG
        try:
            with open(data_folder + os.sep + drive_angle_filename) as f:
                f = f.readlines()
                for line in f:
                    txt_tokens = line.split(':')
                    angle_dict[int(txt_tokens[0])] = float(txt_tokens[1])
                    img_indices.append(int(txt_tokens[0]))
                    img_id_to_direction_dict[int(txt_tokens[0])] = int(txt_tokens[2])
                    direction_to_img_id_dict[int(txt_tokens[2])].append(int(txt_tokens[0]))
                    direction_frequency_stats[int(txt_tokens[2])] += 1
        except FileNotFoundError as e:
            print(e)

        print(data_folder)
        print('Hard-Left','Soft-Left', 'Straight', 'Soft-Right','Hard-Right')
        print(direction_frequency_stats)
        print()

        if not is_bump_data:
            image_fname_prefix = 'img'
            max_instances_per_file = 50
        else:
            image_fname_prefix = 'bump_img'
            max_instances_per_file = 50

        logger.info('=' * 80)
        logger.info(data_folder)
        logger.info('=' * 80)

        file_size_stat = dump_to_tfrecord_sorted_by_direction(data_folder, direction_save_dir, img_id_to_direction_dict, img_indices,
                                             image_fname_prefix, 5, max_instances_per_file, augment_data=True)

        with open(direction_save_dir+os.sep+'dataset_sizes.txt','w') as f:
            for k,v in file_size_stat.items():
                f.write(str(k)+':'+str(v))
                f.write('\n')


def save_testing_data():
    # Used as Testing Data
    # =================================================

    is_bump_list = [False, False, False, False, False, False, False, False,
                    True, True, True, True, True, True, True, True]

    data_folders_list = [
        '.' + os.sep + '..' + os.sep + 'apartment-my1-2000',
        '.' + os.sep + '..' + os.sep + 'apartment-my2-2000',
        '.' + os.sep + '..' + os.sep + 'apartment-my3-2000',
        '.' + os.sep + '..' + os.sep + 'indoor-1-2000',
        '.' + os.sep + '..' + os.sep + 'indoor-1-my1-2000',
        '.' + os.sep + '..' + os.sep + 'grande_salle-my1-2000',
        '.' + os.sep + '..' + os.sep + 'grande_salle-my2-2000',
        '.' + os.sep + '..' + os.sep + 'sandbox-2000',
        '.' + os.sep + '..' + os.sep + 'apartment-my1-bump-200',
        '.' + os.sep + '..' + os.sep + 'apartment-my2-bump-200',
        '.' + os.sep + '..' + os.sep + 'apartment-my3-bump-200',
        '.' + os.sep + '..' + os.sep + 'indoor-1-bump-200',
        '.' + os.sep + '..' + os.sep + 'indoor-1-my1-bump-200',
        '.' + os.sep + '..' + os.sep + 'grande_salle-my1-bump-200',
        '.' + os.sep + '..' + os.sep + 'grande_salle-my2-bump-200',
        '.' + os.sep + '..' + os.sep + 'sandbox-bump-200'
    ]

    assert len(is_bump_list) == len(data_folders_list), 'Bump List length and Data Folder lenght do not match'

    for is_bump_data, data_folder in zip(is_bump_list, data_folders_list):

        direction_frequency_stats = [0 for _ in range(3)]
        equal_save_dir = data_folder + os.sep + 'data-equal'

        if not os.path.exists(equal_save_dir):
            os.mkdir(equal_save_dir)


        angle_dict = {}
        img_id_to_direction_dict = {}
        direction_to_img_id_dict = {0: [], 1: [], 2: []}
        img_indices = []

        drive_angle_filename = config.DRIVE_ANGLE_LOG if not is_bump_data else config.BUMP_DRIVE_ANGLE_LOG
        try:
            with open(data_folder + os.sep + drive_angle_filename) as f:
                f = f.readlines()
                for line in f:
                    txt_tokens = line.split(':')
                    angle_dict[int(txt_tokens[0])] = float(txt_tokens[1])
                    img_indices.append(int(txt_tokens[0]))
                    img_id_to_direction_dict[int(txt_tokens[0])] = int(txt_tokens[2])
                    direction_to_img_id_dict[int(txt_tokens[2])].append(int(txt_tokens[0]))
                    direction_frequency_stats[int(txt_tokens[2])] += 1
        except FileNotFoundError as e:
            print(e)

        print(data_folder)
        print('Right', 'Straight', 'Left')
        print(direction_frequency_stats)
        print()

        if not is_bump_data:
            data_indices = img_indices
            image_fname_prefix = 'img'
            max_instances_per_file = 50
        else:
            data_indices = img_indices
            image_fname_prefix = 'bump_img'
            max_instances_per_file = 50

        logger.info('=' * 80)
        logger.info(data_folder)
        logger.info('=' * 80)


        equal_img_indices = get_image_indices_with_uniform_distribution(direction_to_img_id_dict)
        #file_size_stat = dump_to_tfrecord_suffled(data_folder, equal_save_dir, img_id_to_direction_dict,equal_img_indices,image_fname_prefix)
        file_size_stat = dump_to_tfrecord_in_chunks(data_folder,equal_save_dir,img_id_to_direction_dict,equal_img_indices,image_fname_prefix,
                                                    3, max_instances_per_file=None,augment_data=True,shuffle=True,save_images_for_testing=True)

        with open(equal_save_dir+os.sep+'dataset_sizes.txt','w') as f:
            for k,v in file_size_stat.items():
                f.write(str(k)+':'+str(v))
                f.write('\n')


def save_testing_data_5_way():
    # Used as Testing Data
    # =================================================

    is_bump_list = [False, False, False]

    data_folders_list = [
        '.' + os.sep + '..' + os.sep + 'indoor-1-5-way-3000',
        '.' + os.sep + '..' + os.sep + 'indoor-1-my1-5-way-3000',
        '.' + os.sep + '..' + os.sep + 'indoor-1-my2-5-way-3000',
    ]

    assert len(is_bump_list) == len(data_folders_list), 'Bump List length and Data Folder lenght do not match'

    for is_bump_data, data_folder in zip(is_bump_list, data_folders_list):

        direction_frequency_stats = [0 for _ in range(5)]
        equal_save_dir = data_folder + os.sep + 'data-equal-5-way'

        if not os.path.exists(equal_save_dir):
            os.mkdir(equal_save_dir)

        angle_dict = {}
        img_id_to_direction_dict = {}
        direction_to_img_id_dict = {0: [], 1: [], 2: [], 3:[], 4:[]}
        img_indices = []

        drive_angle_filename = config.DRIVE_ANGLE_LOG if not is_bump_data else config.BUMP_DRIVE_ANGLE_LOG
        try:
            with open(data_folder + os.sep + drive_angle_filename) as f:
                f = f.readlines()
                for line in f:
                    txt_tokens = line.split(':')
                    angle_dict[int(txt_tokens[0])] = float(txt_tokens[1])
                    img_indices.append(int(txt_tokens[0]))
                    img_id_to_direction_dict[int(txt_tokens[0])] = int(txt_tokens[2])
                    direction_to_img_id_dict[int(txt_tokens[2])].append(int(txt_tokens[0]))
                    direction_frequency_stats[int(txt_tokens[2])] += 1
        except FileNotFoundError as e:
            print(e)

        print(data_folder)
        print('Hard-Left', 'Soft-Left', 'Straight', 'Soft-Right', 'Hard-Right')
        print(direction_frequency_stats)
        print()

        if not is_bump_data:
            data_indices = img_indices
            image_fname_prefix = 'img'
            max_instances_per_file = 50
        else:
            data_indices = img_indices
            image_fname_prefix = 'bump_img'
            max_instances_per_file = 50

        logger.info('=' * 80)
        logger.info(data_folder)
        logger.info('=' * 80)


        equal_img_indices = get_image_indices_with_uniform_distribution(direction_to_img_id_dict)
        #file_size_stat = dump_to_tfrecord_suffled(data_folder, equal_save_dir, img_id_to_direction_dict,equal_img_indices,image_fname_prefix)
        file_size_stat = dump_to_tfrecord_in_chunks(data_folder,equal_save_dir,img_id_to_direction_dict,
                                                    equal_img_indices,image_fname_prefix,5,
                                                    max_instances_per_file=None,augment_data=True,shuffle=True,save_images_for_testing=True)

        with open(equal_save_dir+os.sep+'dataset_sizes.txt','w') as f:
            for k,v in file_size_stat.items():
                f.write(str(k)+':'+str(v))
                f.write('\n')


logger = None
if __name__ == '__main__':

    logger = logging.getLogger('Logger')
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(message)s'))
    console.setLevel(logging.INFO)
    fileHandler = logging.FileHandler('data-sizes-5-way.log', mode='w')
    fileHandler.setFormatter(logging.Formatter('%(message)s'))
    fileHandler.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(fileHandler)

    save_testing_data_5_way()
    #save_training_data()
    # Used as Test Data (Should have equal amounts for each direction)
    # =======================================
    '''is_bump_list = [False, True, False, True, False, True]

    data_folders_list = ['.'+os.sep+ '..'+ os.sep+'data_indoor_1_1000', '.'+os.sep+ '..'+ os.sep+'data_indoor_1_bump_200',
                    '.' + os.sep + '..' + os.sep + 'data_sandbox_1000',
                    '.' + os.sep + '..' + os.sep + 'data_sandbox_bump_200',
                    '.' + os.sep + '..' + os.sep + 'data_grande_salle_1000',
                    '.' + os.sep + '..' + os.sep + 'data_grande_salle_bump_200']'''



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
