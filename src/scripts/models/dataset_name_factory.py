import os

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


def new_get_noncol_train_data_sorted_by_direction_col_noncol_test_data():

    sub_dir = 'data-seperated-by-direction'
    dataset_filenames = {'train_dataset':
                             {'left':
                                       ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(8)] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(7)] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(8)] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(7)] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(7)] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(10)],

                               'straight':
                                   ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                    sub_dir + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(22)] +
                                   ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                    sub_dir + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(24)] +
                                   ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                    sub_dir + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(21)] +
                                   ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                    sub_dir + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(25)] +
                                   ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                    sub_dir + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(25)] +
                                   ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                    sub_dir + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(21)],

                                'right':
                                  ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(10)] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(8)] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(9)] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(8)] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(8)] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(8)]

                                           },
                         'train_bump_dataset': [],

                         'valid_dataset':
                             {'left':
                                  ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-8.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-7.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-8.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-7.tfrecords'] +

                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-0-part-10.tfrecords'],

                              'straight':
                                  ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-1-part-22.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   sub_dir + os.sep + 'image-1-part-24.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   sub_dir + os.sep + 'image-1-part-22.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-1-part-25.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-1-part-25.tfrecords'] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-1-part-21.tfrecords'],

                              'right':
                                  ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-10.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-8.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-9.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-8.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-8.tfrecords'] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   sub_dir + os.sep + 'image-2-part-8.tfrecords']

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

    dataset_sizes = {'train_dataset': sum([50 for _ in range(236)]),
                     'train_bump_dataset': [],
                     'valid_dataset': sum([50 for _ in range(15)]),
                     'test_dataset': sum([184, 146]), # size of sandbox_1000: 247
                     'test_bump_dataset': sum([40, 40])} # size of sandbox_bump_200: 47

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
                     'indoor-1-2000','indoor-1-my1-2000','grande_salle-my1-2000',
                     'grande_salle-my2-2000','sandbox-2000']

    assert hold_out_index < len(dataset_names), 'Hold out index should be less than length of dataset folders'

    train_valid_dataset_chunk_count = {
        'left':[18, 16, 17, 15, 15, 18, 15,12],
        'straight': [44, 49, 46, 50, 51, 43, 50,56],
        'right': [18, 16, 17, 15, 15, 18, 15, 12]
                           }
    test_dataset_sizes_per_dir = [401,369,422,379,344,409,387,277]
    bump_test_dataset_sizes_per_dir = [66,24,48,42,60,48,48,40]

    bump_dataset_names = ['apartment-my1-bump-200','apartment-my2-bump-200','apartment-my3-bump-200',
                     'indoor-1-bump-200','indoor-1-my1-bump-200','grande_salle-my1-bump-200',
                     'grande_salle-my2-bump-200','sandbox-bump-200']

    train_sub_dir = 'data-separated-by-direction-augmented'
    test_sub_dir = 'data-equal'
    test_dataset_filename = 'image-shuffled.tfrecords'

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
                     'test_dataset': test_dataset_sizes_per_dir[hold_out_index], # size of sandbox_1000: 247
                     'test_bump_dataset': bump_test_dataset_sizes_per_dir[hold_out_index]} # size of sandbox_bump_200: 47

    print(dataset_filenames)
    print(dataset_sizes)
    return dataset_filenames, dataset_sizes