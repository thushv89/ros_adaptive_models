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
    dataset_filenames = {'train_dataset':['..' + os.sep + '..' + os.sep + 'wombot-sit-front-jan-19-daytime' +
                                          os.sep + 'train' + os.sep + 'image-shuffled.hdf5',
                                          '..' + os.sep + '..' + os.sep + 'wombot-level1-courtyad-afternoon' +
                                          os.sep + 'train' + os.sep + 'image-shuffled.hdf5',
                                          '..' + os.sep + '..' + os.sep + 'wombot-lab-level5-afternoon' +
                                          os.sep + 'train' + os.sep + 'image-shuffled.hdf5',
                                          '..' + os.sep + '..' + os.sep + 'wombot-acfr-front-afternoon-2' +
                                          os.sep + 'train' + os.sep + 'image-shuffled.hdf5'
                                          ],

                         'valid_dataset':
                             None,

                         'test_dataset': ['..' + os.sep + '..' + os.sep + 'wombot-sit-front-jan-19-daytime' + os.sep +
                                          'test' + os.sep + 'image-shuffled.hdf5',
                                          '..' + os.sep + '..' + os.sep + 'wombot-level1-courtyad-afternoon' + os.sep +
                                          'test' + os.sep + 'image-shuffled.hdf5',
                                          '..' + os.sep + '..' + os.sep + 'wombot-lab-level5-afternoon' +
                                          os.sep + 'test' + os.sep + 'image-shuffled.hdf5',
                                          '..' + os.sep + '..' + os.sep + 'wombot-acfr-front-afternoon-2' +
                                          os.sep + 'test' + os.sep + 'image-shuffled.hdf5'
                                          ]
                         }

    dataset_sizes = {'train_dataset': [315*3, 418*3, 490*3, 522*3],
                     'valid_dataset': None,
                     'test_dataset': [54*3, 67*3, 66*3, 89*3]}



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


def new_get_train_test_data_with_holdout(hold_out_index):

    dataset_names = ['apartment-my1-2000','apartment-my2-2000','apartment-my3-2000',
                     'indoor-1-2000','indoor-1-my1-2000','indoor-1-my2-2000','grande_salle-my1-2000',
                     'grande_salle-my2-2000','grande_salle-my3-2000','sandbox-2000']

    assert hold_out_index < len(dataset_names), 'Hold out index should be less than length of dataset folders'

    train_valid_dataset_chunk_count = {
        'left':[18, 16, 17, 15, 15, 18, 18, 15,15,12],
        'straight': [44, 49, 46, 50, 51,45, 43, 50,50,56],
        'right': [18, 16, 17, 15, 15, 18, 18, 15,15, 12]
                           }
    test_dataset_sizes_per_dir = [2406,2214,2532,2274,2064,2172,2454,2322,2352,1662]
    bump_test_dataset_sizes_per_dir = [396,144,288,252,360,216,288,288,216,240]

    bump_dataset_names = ['apartment-my1-bump-200','apartment-my2-bump-200','apartment-my3-bump-200',
                     'indoor-1-bump-200','indoor-1-my1-bump-200','indoor-1-my2-bump-200','grande_salle-my1-bump-200',
                     'grande_salle-my2-bump-200','grande_salle-my3-bump-200','sandbox-bump-200']

    assert len(dataset_names)==len(bump_dataset_names),"length of bump and non-bump dataset names do not match"

    train_sub_dir = 'data-separated-by-direction-augmented'
    test_sub_dir = 'data-equal'
    test_dataset_filename = 'data-chunk-0.tfrecords' # (old) 'image-shuffled.tfrecords'

    dataset_filenames = {
        'test_dataset': ['..' + os.sep + dataset_names[hold_out_index] + os.sep +
                         test_sub_dir + os.sep + test_dataset_filename],
        'test_bump_dataset': ['..' + os.sep + bump_dataset_names[hold_out_index] + os.sep +
                              test_sub_dir + os.sep + test_dataset_filename],
        'train_dataset': {'left':[], 'straight':[], 'right':[]},
        'valid_dataset': []
                         }

    train_size,valid_size = 0,0
    for data_index in range(len(dataset_names)):

        if data_index == hold_out_index:
            continue

        for di,direct in enumerate(['left','straight','right']):
            dataset_filenames['train_dataset'][direct].extend(
                [
                    '..'+os.sep+dataset_names[data_index] + os.sep + train_sub_dir +
                    os.sep + 'image-%d-part-%d.tfrecords' % (di,i) for i in range(train_valid_dataset_chunk_count[direct][data_index])
                 ]
            )
            train_size += 50 * train_valid_dataset_chunk_count[direct][data_index]
            dataset_filenames['valid_dataset'].append(
                '..' + os.sep + dataset_names[data_index] + os.sep + train_sub_dir +
                os.sep + 'image-%d-part-%d.tfrecords' % (di,train_valid_dataset_chunk_count[direct][data_index])
            )
            valid_size += 45

    dataset_sizes = {'train_dataset': train_size,
                     'valid_dataset': valid_size,
                     'test_dataset': test_dataset_sizes_per_dir[hold_out_index],
                     'test_bump_dataset': bump_test_dataset_sizes_per_dir[hold_out_index]}

    print(dataset_filenames)
    print(dataset_sizes)
    return dataset_filenames, dataset_sizes


def new_get_train_test_data_with_holdout_5_way(hold_out_index):

    dataset_names = ['indoor-1-5-way-3000','indoor-1-my1-5-way-3000','indoor-1-my2-5-way-3000']

    assert hold_out_index < len(dataset_names), 'Hold out index should be less than length of dataset folders'

    train_valid_dataset_chunk_count = {
        'hard-left':[12, 11, 10],
        'soft-left': [33, 30, 28],
        'straight': [25, 35, 41],
        'soft-right': [32, 31, 28],
        'hard-right': [11, 10, 10]
                           }
    test_dataset_sizes_per_dir = [2990,2670,2670]

    train_sub_dir = 'data-separated-by-direction-augmented-5-way'
    test_sub_dir = 'data-equal-5-way'
    test_dataset_filename = 'data-chunk-0.tfrecords' # (old) 'image-shuffled.tfrecords'

    dataset_filenames = {
        'test_dataset': ['..' + os.sep + dataset_names[hold_out_index] + os.sep +
                         test_sub_dir + os.sep + test_dataset_filename],
        'test_bump_dataset': [],
        'train_dataset': {'hard-left':[], 'soft-left':[], 'straight':[], 'soft-right':[], 'hard-right':[]},
        'valid_dataset': []
                         }

    train_size,valid_size = 0,0
    for data_index in range(len(dataset_names)):

        if data_index == hold_out_index:
            continue

        for di,direct in enumerate(['hard-left','soft-left','straight','soft-right','hard-right']):
            dataset_filenames['train_dataset'][direct].extend(
                [
                    '..'+os.sep+dataset_names[data_index] + os.sep + train_sub_dir +
                    os.sep + 'image-%d-part-%d.tfrecords' % (di,i) for i in range(train_valid_dataset_chunk_count[direct][data_index])
                 ]
            )
            train_size += 50 * train_valid_dataset_chunk_count[direct][data_index]
            dataset_filenames['valid_dataset'].append(
                '..' + os.sep + dataset_names[data_index] + os.sep + train_sub_dir +
                os.sep + 'image-%d-part-%d.tfrecords' % (di,train_valid_dataset_chunk_count[direct][data_index])
            )
            valid_size += 45

    dataset_sizes = {'train_dataset': train_size,
                     'valid_dataset': valid_size,
                     'test_dataset': test_dataset_sizes_per_dir[hold_out_index],
                     'test_bump_dataset': -1}

    print(dataset_filenames)
    print(dataset_sizes)
    return dataset_filenames, dataset_sizes


def new_get_train_test_data_with_holdout_5_way_half_dataset(hold_out_index):

    dataset_names = ['indoor-1-5-way-3000','indoor-1-my1-5-way-3000','indoor-1-my2-5-way-3000']

    assert hold_out_index < len(dataset_names), 'Hold out index should be less than length of dataset folders'

    train_valid_dataset_chunk_count = {
        'hard-left':[12, 11, 10],
        'soft-left': [33, 30, 28],
        'straight': [25, 35, 41],
        'soft-right': [32, 31, 28],
        'hard-right': [11, 10, 10]
                           }
    test_dataset_sizes_per_dir = [2990,2670,2670]

    train_sub_dir = 'data-separated-by-direction-augmented-5-way'
    test_sub_dir = 'data-equal-5-way'
    test_dataset_filename = 'data-chunk-0.tfrecords' # (old) 'image-shuffled.tfrecords'

    dataset_filenames = {
        'test_dataset': [],
        'test_bump_dataset': [],
        'train_dataset': {'hard-left':[], 'soft-left':[], 'straight':[], 'soft-right':[], 'hard-right':[]},
        'valid_dataset': []
                         }

    train_size,valid_size,test_size = 0,0,0
    for data_index in range(len(dataset_names)):

        if data_index == hold_out_index:
            for di, direct in enumerate(['hard-left', 'soft-left', 'straight', 'soft-right', 'hard-right']):
                dataset_filenames['train_dataset'][direct].extend(
                    [
                    '..' + os.sep + dataset_names[data_index] + os.sep + train_sub_dir +
                    os.sep + 'image-%d-part-%d.tfrecords' % (di,i) for i in range(train_valid_dataset_chunk_count[direct][data_index]//2)
                    ]
                )
                train_size += 50 * train_valid_dataset_chunk_count[direct][data_index]//2
                dataset_filenames['test_dataset'].extend(
                    [
                        '..' + os.sep + dataset_names[data_index] + os.sep + train_sub_dir +
                        os.sep + 'image-%d-part-%d.tfrecords' % (di, i) for i in
                        range(train_valid_dataset_chunk_count[direct][data_index] // 2,train_valid_dataset_chunk_count[direct][data_index])
                        ]
                )
                test_size += 50 * train_valid_dataset_chunk_count[direct][data_index] // 2

        for di,direct in enumerate(['hard-left','soft-left','straight','soft-right','hard-right']):
            dataset_filenames['train_dataset'][direct].extend(
                [
                    '..'+os.sep+dataset_names[data_index] + os.sep + train_sub_dir +
                    os.sep + 'image-%d-part-%d.tfrecords' % (di,i) for i in range(train_valid_dataset_chunk_count[direct][data_index])
                 ]
            )
            train_size += 50 * train_valid_dataset_chunk_count[direct][data_index]
            dataset_filenames['valid_dataset'].append(
                '..' + os.sep + dataset_names[data_index] + os.sep + train_sub_dir +
                os.sep + 'image-%d-part-%d.tfrecords' % (di,train_valid_dataset_chunk_count[direct][data_index])
            )
            valid_size += 40

    dataset_sizes = {'train_dataset': train_size,
                     'valid_dataset': valid_size,
                     'test_dataset': test_size,
                     'test_bump_dataset': -1}

    print(dataset_filenames)
    print(dataset_sizes)
    return dataset_filenames, dataset_sizes