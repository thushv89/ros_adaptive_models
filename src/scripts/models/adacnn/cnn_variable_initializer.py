import tensorflow as tf
import config
import logging
import sys

sess = None
graph = None
logger = None

def set_from_main(session):
    global sess, logger
    sess = session

    logger = logging.getLogger('InitLogger')
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('[%(name)s] [%(funcName)s] %(message)s'))
    console.setLevel(logging.INFO)
    logger.addHandler(console)


def build_tensorflw_variables_detached():
    '''
    Build the required tensorflow variables to populate the VGG-16 model
    All are initialized with zeros (initialization doesn't matter in this case as we assign exact values later)
    :param variable_shapes: Shapes of the variables
    :return:
    '''
    global logger,sess

    logger.info("Building Tensorflow Variables (Tensorflow)...")
    #with sess.as_default:
    for si,scope in enumerate(config.TF_ANG_SCOPES):
        with tf.variable_scope(scope) as sc:

            # Try Except because if you try get_variable with an intializer and
            # the variable exists, you will get a ValueError saying the variable exists

            try:
                    if not (scope=='fc1' or scope=='out' or 'pool' in scope):
                        tf.get_variable(config.TF_WEIGHTS_STR,shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope],
                                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
                        tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_DETACHED[scope][-1],
                                               initializer = tf.random_uniform_initializer(minval=-0.01,maxval=0.01,dtype=tf.float32))
                    else:

                        direction = ['left','straight','right']
                        for di in direction:
                            with tf.variable_scope(di):
                                tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope],
                                                initializer=    tf.contrib.layers.xavier_initializer())
                                tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_DETACHED[scope][-1],
                                                initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01, dtype=tf.float32))

            except ValueError as e:
                logger.critical(e)
                logger.debug('Variables in scope %s already initialized\n'%scope)

    print([v.name for v in tf.global_variables()])


def build_tensorflw_variables_naive():
    '''
    Build the required tensorflow variables to populate the VGG-16 model
    All are initialized with zeros (initialization doesn't matter in this case as we assign exact values later)
    :param variable_shapes: Shapes of the variables
    :return:
    '''
    global logger,sess

    logger.info("Building Tensorflow Variables (Tensorflow)...")
    with sess.as_default():
        for si,scope in enumerate(config.TF_ANG_SCOPES):
            with tf.variable_scope(scope) as sc:

                # Try Except because if you try get_variable with an intializer and
                # the variable exists, you will get a ValueError saying the variable exists

                try:

                    if 'pool' not in scope and 'out' not in scope:
                        tf.get_variable(config.TF_WEIGHTS_STR,shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope],
                                                  initializer=tf.contrib.layers.xavier_initializer())
                        tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_NAIVE[scope][-1],
                                               initializer = tf.random_uniform_initializer(minval=-0.01,maxval=+0.01,dtype=tf.float32))

                        with tf.variable_scope(config.TF_WEIGHTS_STR):
                            tf.get_variable(config.TF_MOMENTUM_STR,shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope],
                                            initializer = tf.constant_initializer(0,dtype=tf.float32))
                        with tf.variable_scope(config.TF_BIAS_STR):
                            tf.get_variable(config.TF_MOMENTUM_STR,shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope][-1],
                                            initializer = tf.constant_initializer(0,dtype=tf.float32))

                    if 'out' in scope:
                        for di in ['left','straight','right']:
                            with tf.variable_scope(di):
                                tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope],
                                                initializer=tf.contrib.layers.xavier_initializer())
                                tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_NAIVE[scope][-1],
                                                initializer=tf.random_uniform_initializer(minval=-0.01, maxval=+0.01,
                                                                                          dtype=tf.float32))

                                with tf.variable_scope(config.TF_WEIGHTS_STR):
                                    tf.get_variable(config.TF_MOMENTUM_STR, shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope],
                                                    initializer=tf.constant_initializer(0, dtype=tf.float32))
                                with tf.variable_scope(config.TF_BIAS_STR):
                                    tf.get_variable(config.TF_MOMENTUM_STR,
                                                    shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope][-1],
                                                    initializer=tf.constant_initializer(0, dtype=tf.float32))


                except ValueError as e:
                    logger.critical(e)
                    logger.debug('Variables in scope %s already initialized\n'%scope)

        print([v.name for v in tf.global_variables()])


