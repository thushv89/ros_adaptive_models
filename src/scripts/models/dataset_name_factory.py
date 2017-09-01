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
    dataset_filenames = {'train_dataset':
                             {'left':
                                       ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(8)] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(7)] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(8)] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(7)] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(7)] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-0-part-%d.tfrecords' % i for i in
                                   range(10)],

                               'straight':
                                   ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                    'data-separated-by-direction' + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(22)] +
                                   ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                    'data-separated-by-direction' + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(24)] +
                                   ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                    'data-separated-by-direction' + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(21)] +
                                   ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                    'data-separated-by-direction' + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(25)] +
                                   ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                    'data-separated-by-direction' + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(25)] +
                                   ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                    'data-separated-by-direction' + os.sep + 'image-1-part-%d.tfrecords' % i
                                    for i in
                                    range(21)],

                                'right':
                                  ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(10)] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(8)] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(9)] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(8)] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(8)] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-2-part-%d.tfrecords' % i
                                   for i in
                                   range(8)]

                                           },
                         'train_bump_dataset': [],

                         'valid_dataset':
                             {'left':
                                  ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-0-part-8.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-0-part-7.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-0-part-8.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-0-part-7.tfrecords'] +

                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-0-part-10.tfrecords'],

                              'straight':
                                  ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-1-part-22.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-1-part-24.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-1-part-22.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-1-part-25.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-1-part-25.tfrecords'] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-1-part-21.tfrecords'],

                              'right':
                                  ['..' + os.sep + 'apartment-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-2-part-10.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my2-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-2-part-8.tfrecords'] +
                                  ['..' + os.sep + 'apartment-my3-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-2-part-9.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-2-part-8.tfrecords'] +
                                  ['..' + os.sep + 'indoor-1-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-2-part-8.tfrecords'] +
                                  ['..' + os.sep + 'grande_salle-my1-2000' + os.sep +
                                   'data-separated-by-direction' + os.sep + 'image-2-part-8.tfrecords']

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