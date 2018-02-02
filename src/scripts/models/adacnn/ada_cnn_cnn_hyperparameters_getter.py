import logging
import os
import config


def get_research_hyperparameters(adapt, use_pooling,logging_level):
    '''
    These are various research hyperparameters used in AdaCNN.
    Some hyperparameters can be sent as arguments for convenience
    :param dataset_name:
    :param adapt: Whether use AdaCNN or Rigid-CNN
    :param use_pooling: Use Rigid-CNN-B or Rigid-CNN
    :return: Research hyperparamters as a dictionary
    '''
    research_parameters = {
        'save_train_test_images': False, # If true will save train and test images randomly (to make sure images are correctly read)
        'log_class_distribution': True, 'log_distribution_every': 24, # log distribution of data (useful for generating data distribution over time curves)
        'adapt_structure': adapt,  # Enable AdaCNN behavior
        'hard_pool_acceptance_rate': 0.1,  # Probability with which data is accepted in to the pool
        'optimizer': 'Momentum', 'momentum': 0.0, 'pool_momentum': 0.9,  # Two momentums one for data one for pool
        'use_custom_momentum_opt': True, # Use a custom implemented momentum (Tensorflow builtin optimizer doesnot support variable size tensors
        'remove_filters_by': 'Activation', # The criteria for removing filters (AdaCNN) set of minimum maximum mean activations
        'optimize_end_to_end': True, # if true functions such as add and finetune will optimize the network from starting layer to end (fulcon_out)
        'loss_diff_threshold': 0.02, # This is used to check if the loss reduction has stabalized
        'start_adapting_after': 500, # Acts as a warming up phase, adapting from the very begining can make CNNs unstable
        'debugging': True if logging_level == logging.DEBUG else False,
        'stop_training_at': 11000,  # If needed to truncate training earlier
        'train_min_activation': False,
        'use_weighted_loss': True,  # Weight the loss by the class distribution at the time
        'whiten_images': True,  # Whiten images using batch mean and batch std for each batch
        'finetune_rate': 0.5,  # amount of data from data pool used for finetuning
        'pool_randomize': True,  # randomize the pool data when training with it
        'pool_randomize_rate': 0.25,  # frequency the pool data randomized for
        'pooling_for_nonadapt': use_pooling,
        'hard_pool_max_threshold': 0.5,  # when there's not much data use a higher pool accumulation rate
    }

    research_parameters['start_adapting_after'] = 1000

    if adapt:
        # quickly accumulate data at the beginning
        research_parameters['hard_pool_acceptance_rate'] *= 2.0

    return research_parameters


def get_interval_related_hyperparameters():

    interval_parameters = {
        'history_dump_interval': 500,
        'policy_interval': 0,  # number of batches to process for each policy iteration
        'finetune_interval': 0,
        'orig_finetune_interval':0,
        'test_interval': 100
    }

    interval_parameters['policy_interval'] = 10
    interval_parameters['finetune_interval'] = 10
    interval_parameters['orig_finetune_interval'] = 10

    return interval_parameters


def get_model_specific_hyperparameters(adapt_structure, use_pooling, use_fse_capacity, num_labels):

    model_hyperparameters = {}
    model_hyperparameters['adapt_structure'] = adapt_structure
    model_hyperparameters['batch_size'] = 25  # number of datapoints in a single batch
    model_hyperparameters['start_lr'] = 0.0001
    model_hyperparameters['min_learning_rate'] = 0.000001
    model_hyperparameters['decay_learning_rate'] = True
    model_hyperparameters['decay_rate'] = 0.75
    model_hyperparameters['adapt_decay_rate'] = 0.9 # decay rate used for adaptation related optimziations

    model_hyperparameters['dropout_rate'] = 0.5
    model_hyperparameters['in_dropout_rate'] = 0.0

    model_hyperparameters['use_dropout'] = True
    model_hyperparameters['check_early_stopping_from'] = 5
    model_hyperparameters['accuracy_drop_cap'] = 3
    model_hyperparameters['iterations_per_batch'] = 1

    model_hyperparameters['epochs'] = 5
    model_hyperparameters['num_env'] = 3
    model_hyperparameters['start_eps'] = 0.5
    model_hyperparameters['eps_decay'] = 0.9
    model_hyperparameters['validation_set_accumulation_decay'] = 0.9

    if not (adapt_structure and use_pooling):
        model_hyperparameters['iterations_per_batch'] = 2

    model_hyperparameters['include_l2_loss'] = False
    model_hyperparameters['beta'] = 0.0005

    model_hyperparameters['top_k_accuracy'] = 1.0

    pool_size = model_hyperparameters['batch_size'] * 10* num_labels

    if not adapt_structure:

        cnn_string = "C,2,4,2,2,32#C,4,8,2,2,32#C,2,4,2,2,64#C,2,2,1,1,64" + \
                     "#FC,100,0,0,0,0#Terminate,0,0,0,0,0"
    else:
        cnn_string = "C,2,4,2,2,16#C,4,8,2,2,16#C,2,4,2,2,16#C,2,2,1,1,16" + \
                     "#FC,100,0,0,0,0#Terminate,0,0,0,0,0"



        start_filter_vector = [16,16,16,16,0]
        filter_vector = [32, 32, 64, 64, 0]
        add_amount, remove_amount, add_fulcon_amount = 4, 2, -1
        filter_min_threshold = 12


    model_hyperparameters['n_tasks'] = 2
    model_hyperparameters['binned_data_dist_length'] = 3

    model_hyperparameters['cnn_string'] = cnn_string

    if adapt_structure or use_pooling:
        model_hyperparameters['pool_size'] = pool_size

    if adapt_structure:
        model_hyperparameters['filter_vector'] = filter_vector
        model_hyperparameters['start_filter_vector'] = start_filter_vector
        model_hyperparameters['add_amount'] = add_amount
        model_hyperparameters['add_fulcon_amount'] = add_fulcon_amount
        model_hyperparameters['remove_amount'] = remove_amount
        model_hyperparameters['filter_min_threshold'] = filter_min_threshold

    return model_hyperparameters


def get_data_specific_hyperparameters():
    global research_parameters, interval_parameters
    data_hyperparameters,model_hyperparameters = {},{}

    resize_to = 0

    image_size = config.TF_INPUT_AFTER_RESIZE[:-1]
    num_labels = config.TF_NUM_CLASSES
    num_channels = 3  # rgb

    data_hyperparameters['image_size'] = image_size
    data_hyperparameters['original_image_size'] = config.TF_INPUT_SIZE
    data_hyperparameters['resize_to'] = resize_to
    data_hyperparameters['n_labels'] = num_labels
    data_hyperparameters['n_channels'] = num_channels

    return data_hyperparameters