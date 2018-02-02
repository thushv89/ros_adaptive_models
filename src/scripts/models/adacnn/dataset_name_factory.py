import os
import numpy as np


def old_get_col_noncol_train_and_col_noncol_test_data():
    dataset_filenames = {'train_dataset': ['..' + os.sep + 'data_indoor_1_1000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(3)] +
                                          ['..' + os.sep + 'data_sandbox_1000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(3)] +
                                          ['..' + os.sep + 'data_grande_salle_1000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(3)],
                         'train_bump_dataset': ['..' + os.sep + 'data_indoor_1_bump_200' + os.sep +
                                                'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for
                                                i in
                                                range(3)] +
                                               ['..' + os.sep + 'data_sandbox_bump_200' + os.sep +
                                                'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for
                                                i in
                                                range(3)] +
                                               ['..' + os.sep + 'data_grande_salle_bump_200' + os.sep +
                                                'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for
                                                i in
                                                range(3)],
                         'test_dataset': ['..' + os.sep + 'data_indoor_1_1000' + os.sep +
                                          'data-chunks-chronological' + os.sep + 'data-chunk-3.tfrecords'] +
                                         ['..' + os.sep + 'data_sandbox_1000' + os.sep +
                                          'data-chunks-chronological' + os.sep + 'data-chunk-3.tfrecords'] +
                                         ['..' + os.sep + 'data_grande_salle_1000' + os.sep +
                                          'data-chunks-chronological' + os.sep + 'data-chunk-3.tfrecords'],
                         'test_bump_dataset': ['..' + os.sep + 'data_indoor_1_bump_200' + os.sep +
                                               'data-chunks-chronological' + os.sep + 'data-chunk-3.tfrecords'] +
                                              ['..' + os.sep + 'data_sandbox_bump_200' + os.sep +
                                               'data-chunks-chronological' + os.sep + 'data-chunk-3.tfrecords'] +
                                              ['..' + os.sep + 'data_grande_salle_bump_200' + os.sep +
                                               'data-chunks-chronological' + os.sep + 'data-chunk-3.tfrecords']
                         }

    dataset_sizes = {'train_dataset': [500 for _ in range(9)],
                     'train_bump_dataset': [50 for _ in range(9)],
                     'test_dataset': [250, 250, 250],
                     'test_bump_dataset': [50, 50, 50]}

    return dataset_filenames, dataset_sizes


def new_get_noncol_train_data_col_noncol_test_data():
    dataset_filenames = {'train_dataset': ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(9)] +
                                          ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(9)] +
                                          ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(9)] +
                                          ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(9)] +
                                          ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(9)] +
                                          ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(9)],
                         'train_bump_dataset': [],
                         'valid_dataset': ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(9,10)] +
                                          ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(9,10)] +
                                          ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(9,10)] +
                                          ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(9,10)] +
                                          ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(9,10)] +
                                          ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                           'data-chunks-chronological' + os.sep + 'data-chunk-%d.tfrecords' % i for i in
                                           range(9,10)],
                         'test_dataset': ['..' + os.sep + 'data_indoor_1_1000' + os.sep +
                                          'data-equal' + os.sep + 'image-direction-shuffled.tfrecords'] +
                                         ['..' + os.sep + 'data_sandbox_1000' + os.sep +
                                          'data-equal' + os.sep + 'image-direction-shuffled.tfrecords'] +
                                         ['..' + os.sep + 'data_grande_salle_1000' + os.sep +
                                          'data-equal' + os.sep + 'image-direction-shuffled.tfrecords'],
                         'test_bump_dataset': ['..' + os.sep + 'data_indoor_1_bump_200' + os.sep +
                                               'data-equal' + os.sep + 'image-direction-shuffled.tfrecords'] +
                                              ['..' + os.sep + 'data_sandbox_bump_200' + os.sep +
                                               'data-equal' + os.sep + 'image-direction-shuffled.tfrecords'] +
                                              ['..' + os.sep + 'data_grande_salle_bump_200' + os.sep +
                                               'data-equal' + os.sep + 'image-direction-shuffled.tfrecords']
                         }

    dataset_sizes = {'train_dataset': [200 for _ in range(9*6)],
                     'train_bump_dataset': [],
                     'valid_dataset': [200 for _ in range(1*6)],
                     'test_dataset': [184, 247, 146],
                     'test_bump_dataset': [40, 47, 40]}

    return dataset_filenames, dataset_sizes


def new_get_noncol_train_data_sorted_by_direction_noncol_test_data():

    sub_dir = 'train'
    dataset_filenames = {'train_dataset':['..' + os.sep + '..' + os.sep + 'wombot-sit-front-evening' +
                                          os.sep + 'train' + os.sep + 'image-shuffled.hdf5',
                                          #'..' + os.sep + '..' + os.sep + 'wombot-level1-courtyad-afternoon' +
                                          #os.sep + 'train' + os.sep + 'image-shuffled.hdf5',
                                          '..' + os.sep + '..' + os.sep + 'wombot-lab-level5-afternoon' +
                                          os.sep + 'train' + os.sep + 'image-shuffled.hdf5',
                                          '..' + os.sep + '..' + os.sep + 'wombot-acfr-front-afternoon-2' +
                                          os.sep + 'train' + os.sep + 'image-shuffled.hdf5'
                                          ],

                         'valid_dataset':
                             None,

                         'test_dataset': ['..' + os.sep + '..' + os.sep + 'wombot-sit-front-evening' + os.sep +
                                          'test' + os.sep + 'image-shuffled.hdf5',
                                          #'..' + os.sep + '..' + os.sep + 'wombot-level1-courtyad-afternoon' + os.sep +
                                          #'test' + os.sep + 'image-shuffled.hdf5',
                                          '..' + os.sep + '..' + os.sep + 'wombot-lab-level5-afternoon' +
                                          os.sep + 'test' + os.sep + 'image-shuffled.hdf5',
                                          '..' + os.sep + '..' + os.sep + 'wombot-acfr-front-afternoon-2' +
                                          os.sep + 'test' + os.sep + 'image-shuffled.hdf5'
                                          ]
                         }

    dataset_sizes = {'train_dataset': [3714, 5448, 3300],
                     'valid_dataset': None,
                     'test_dataset': [500, 500, 500]}



    return dataset_filenames, dataset_sizes

def new_get_noncol_train_data_sorted_by_direction_augmented_col_noncol_test_data():

    sub_dir = 'data-separated-by-direction-augmented'
    dataset_filenames = {'train_dataset':
                             {'left':
                                       ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(18)] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(16)] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(17)] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(15)] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(15)] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(19)],

                               'straight':
                                   ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                    sub_dir + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(45)] +
                                   ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                    sub_dir + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(49)] +
                                   ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                    sub_dir + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(46)] +
                                   ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                    sub_dir + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(49)] +
                                   ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                    sub_dir + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(51)] +
                                   ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                    sub_dir + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(43)],

                                'right':
                                  ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(18)] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(16)] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(17)] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(15)] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(15)] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(19)]

                                           },
                         'train_bump_dataset': [],

                         'valid_dataset':
                             {'left':
                                  ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-18.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-16.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-17.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-15.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-15.tfrecords'] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-19.tfrecords'],

                              'straight':
                                  ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-1-part-45.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   sub_dir + os.sep + 'image-1-part-49.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   sub_dir + os.sep + 'image-1-part-46.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-1-part-49.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-1-part-51.tfrecords'] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-1-part-43.tfrecords'],

                              'right':
                                  ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-18.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-16.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-17.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-15.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-15.tfrecords'] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-19.tfrecords']

                              }
        ,
                         'test_dataset': ['..' + os.sep + 'data_indoor_1_1000' + os.sep +
                                          'data-equal' + os.sep + 'image-direction-shuffled.tfrecords']
                                         +
                                         ['..' + os.sep + 'data_grande_salle_1000' + os.sep +
                                          'data-equal' + os.sep + 'image-direction-shuffled.tfrecords'],
                         'test_bump_dataset': ['..' + os.sep + 'data_indoor_1_bump_200' + os.sep +
                                               'data-equal' + os.sep + 'image-direction-shuffled.tfrecords']
                                               +
                                              ['..' + os.sep + 'data_grande_salle_bump_200' + os.sep +
                                               'data-equal' + os.sep + 'image-direction-shuffled.tfrecords']
                         }

    dataset_sizes = {'train_dataset': sum([50 for _ in range(483)]),
                     'train_bump_dataset': [],
                     'valid_dataset': sum([50 for _ in range(15)]),
                     'test_dataset': sum([184, 146]), # size of sandbox_1000: 247
                     'test_bump_dataset': sum([40, 40])} # size of sandbox_bump_200: 47

    return dataset_filenames, dataset_sizes

