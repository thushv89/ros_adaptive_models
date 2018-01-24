from PIL import Image
import numpy as np
import tensorflow as tf
import os
import config
import logging
import sys
import scipy.misc
import h5py

'''
======================== convert_data_to_tfrecords ===============================
This script will take the data in a given folder (.png images, log file) and create tf records
'''
RESIZE_W, RESIZE_H = 160,96

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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

    print(image_ids)
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

    hdf5_file = h5py.File(save_dir + os.sep + 'image-shuffled.hdf5', "w")

    img_id_arr = None
    images_arr = None
    labels_arr = None

    for img_id in image_ids:
        im = Image.open(data_folder+os.sep+ image_fname_prefix + '_%d.png'%img_id)
        im.thumbnail((RESIZE_W, RESIZE_H), Image.ANTIALIAS)
        im_mat = np.array(im, dtype=np.float32)
        drive_direction = drive_direct_dict[img_id]

        if img_id_arr is None or images_arr is None or labels_arr is None:
            img_id_arr = np.array([img_id], dtype=np.int32)
            images_arr = np.array(np.expand_dims(im_mat, axis=0), dtype=np.float32)
            labels_arr = np.array([[drive_direction]], dtype=np.int32)
        else:
            img_id_arr = np.append(img_id_arr, [img_id], axis=0)
            images_arr = np.append(images_arr, np.expand_dims(im_mat,axis=0), axis=0)
            labels_arr = np.append(labels_arr, [[drive_direction]], axis=0)

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

            img_id_arr = np.append(img_id_arr, [img_id], axis=0)
            images_arr = np.append(images_arr, np.expand_dims(im_mat_flip, axis=0), axis=0)
            labels_arr = np.append(labels_arr, [[flip_direction]], axis=0)

            items_written_per_chunk += 1
            if items_written_per_chunk%250==0:
                logger.info('Items read: %d',items_written_per_chunk)

    hdf5_img_id_data = hdf5_file.create_dataset('image_ids', img_id_arr.shape, dtype='f')
    hdf5_img_data = hdf5_file.create_dataset('images', images_arr.shape, dtype='f')
    hdf5_label_data = hdf5_file.create_dataset('labels', labels_arr.shape, dtype='f')

    hdf5_img_id_data[:] = img_id_arr
    hdf5_img_data[:, :, :, :] = images_arr
    hdf5_label_data[:, 0] = labels_arr[:, 0]

    logger.info('image-shuffled-part-%d.tfrecords (Size): %d', chunk_index,
                items_written_per_chunk)
    file_size_stat_dict['image-shuffled-part-%d.tfrecords (Size): %d' % (chunk_index,
                                                                         items_written_per_chunk)] = items_written_per_chunk

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

    logger.info('Keeping %d samples from each direction (uniform)',min_count)
    return all_img_indices


def save_training_data(data_folders_list, is_bump_list, test_indices):
    # Used as Training Data
    # ==========================================
    '''is_bump_list = [False, False, False, False, False, False, False, False]

    data_folders_list = [
        '.' + os.sep + '..' + os.sep + 'apartment-my1-2000',
        '.' + os.sep + '..' + os.sep + 'apartment-my2-2000',
        '.' + os.sep + '..' + os.sep + 'apartment-my3-2000',
        '.' + os.sep + '..' + os.sep + 'indoor-1-2000',
        '.' + os.sep + '..' + os.sep + 'indoor-1-my1-2000',
        '.' + os.sep + '..' + os.sep + 'grande_salle-my1-2000',
        '.' + os.sep + '..' + os.sep + 'grande_salle-my2-2000',
        '.' + os.sep + '..' + os.sep + 'sandbox-2000'
    ]'''

    print('Saving Training Data')

    assert len(is_bump_list) == len(data_folders_list), 'Bump List length and Data Folder lenght do not match'

    for fold_i, (is_bump_data, data_folder) in enumerate(zip(is_bump_list, data_folders_list)):

        direction_frequency_stats = [0 for _ in range(3)]
        train_save_dir = data_folder + os.sep + 'train'

        if not os.path.exists(train_save_dir):
            os.mkdir(train_save_dir)

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
                    if int(txt_tokens[0]) not in test_indices[fold_i]:
                        angle_dict[int(txt_tokens[0])] = float(txt_tokens[1])
                        img_indices.append(int(txt_tokens[0]))
                        img_id_to_direction_dict[int(txt_tokens[0])] = int(txt_tokens[2])
                        direction_to_img_id_dict[int(txt_tokens[2])].append(int(txt_tokens[0]))
                        direction_frequency_stats[int(txt_tokens[2])] += 1
        except FileNotFoundError as e:
            print(e)

        print(data_folder)
        print('(Train) Right', ' Straight', ' Left')
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

        equal_img_indices = get_image_indices_with_uniform_distribution(direction_to_img_id_dict)

        # file_size_stat = dump_to_tfrecord_suffled(data_folder, equal_save_dir, img_id_to_direction_dict,equal_img_indices,image_fname_prefix)
        file_size_stat = dump_to_tfrecord_in_chunks(data_folder, train_save_dir, img_id_to_direction_dict,
                                                    equal_img_indices, image_fname_prefix,
                                                    3, max_instances_per_file=None, augment_data=True, shuffle=True,
                                                    save_images_for_testing=True)

        with open(train_save_dir+os.sep+'dataset_sizes.txt','w') as f:
            for k,v in file_size_stat.items():
                f.write(str(k)+':'+str(v))
                f.write('\n')



def save_testing_data(data_folders_list, is_bump_list, test_indices):
    # Used as Testing Data
    # =================================================

    '''is_bump_list = [False, False, False, False, False, False, False, False,
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
    ]'''

    assert len(is_bump_list) == len(data_folders_list), 'Bump List length and Data Folder lenght do not match'

    for fold_i, (is_bump_data, data_folder) in enumerate(zip(is_bump_list, data_folders_list)):

        direction_frequency_stats = [0 for _ in range(3)]
        equal_save_dir = data_folder + os.sep + 'test'

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

                    if int(txt_tokens[0]) in test_indices[fold_i]:

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

        print('Direction to IMG_ID')
        print([len(direction_to_img_id_dict[k]) for k in direction_to_img_id_dict.keys()])

        equal_img_indices = get_image_indices_with_uniform_distribution(direction_to_img_id_dict)
        #file_size_stat = dump_to_tfrecord_suffled(data_folder, equal_save_dir, img_id_to_direction_dict,equal_img_indices,image_fname_prefix)
        file_size_stat = dump_to_tfrecord_in_chunks(data_folder,equal_save_dir,img_id_to_direction_dict,equal_img_indices,image_fname_prefix,
                                                    3, max_instances_per_file=None,augment_data=True,shuffle=True,save_images_for_testing=True)

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
    fileHandler = logging.FileHandler('data-sizes-test-sizes-new.log', mode='w')
    fileHandler.setFormatter(logging.Formatter('%(message)s'))
    fileHandler.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(fileHandler)

    is_bump_list = [False, False, False, False]

    data_folders_list = [
        '.' + os.sep + '..' + os.sep + 'wombot-sit-front-evening',
        '.' + os.sep + '..' + os.sep + 'wombot-level1-courtyad-afternoon',
        '.' + os.sep + '..' + os.sep + 'wombot-lab-level5-afternoon',
        '.' + os.sep + '..' + os.sep + 'wombot-acfr-front-afternoon-2'
    ]


    test_indices = [
        list(range(1112,1162))+ list(range(1632,1732))+ list(range(3283,3383)),
        list(range(527,827)),
        list(range(606,656))+ list(range(1055,1115)) + list(range(1823,1903)) + list(range(2126,2186)),
        list(range(1007,1097)) + list(range(1252,1292)) + list(range(1465,1565)) + list(range(1688,1708))
    ]

    #is_bump_list = [False, False, False]

    #data_folders_list = [
    #    '.' + os.sep + '..' + os.sep + 'wombot-level1-courtyad-afternoon',
    #    '.' + os.sep + '..' + os.sep + 'wombot-lab-level5-afternoon',
    #    '.' + os.sep + '..' + os.sep + 'wombot-acfr-front-afternoon-2'
    #]

    #test_range = [(527,977),(472,922),(452,902)]

    #save_training_data(data_folders_list, is_bump_list, test_indices)
    save_testing_data(data_folders_list, is_bump_list, test_indices)

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
