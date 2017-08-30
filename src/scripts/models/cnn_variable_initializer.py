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
    with sess.as_default:
        for si,scope in enumerate(config.TF_ANG_SCOPES):
            with tf.variable_scope(scope) as sc:

                # Try Except because if you try get_variable with an intializer and
                # the variable exists, you will get a ValueError saying the variable exists

                try:
                        if not (scope=='fc1' or scope=='out' or 'pool' in scope):
                            tf.get_variable(config.TF_WEIGHTS_STR,shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope],
                                                      initializer=tf.truncated_normal_initializer(stddev=0.02,dtype=tf.float32))
                            tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_DETACHED[scope][-1],
                                                   initializer = tf.random_uniform_initializer(minval=-0.01,maxval=0.01,dtype=tf.float32))

                            with tf.variable_scope(config.TF_WEIGHTS_STR):
                                tf.get_variable(config.TF_MOMENTUM_STR,shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope],
                                                initializer = tf.constant_initializer(0,dtype=tf.float32))
                            with tf.variable_scope(config.TF_BIAS_STR):
                                tf.get_variable(config.TF_MOMENTUM_STR,shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope][-1],
                                                initializer = tf.constant_initializer(0,dtype=tf.float32))
                        else:
                            direction = ['left','straight','right']
                            for di in direction:
                                with tf.variable_scope(di):
                                    tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.02,
                                                                                                dtype=tf.float32))
                                    tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_DETACHED[scope][-1],
                                                    initializer=tf.constant_initializer(0.001, dtype=tf.float32))

                                    with tf.variable_scope(config.TF_WEIGHTS_STR):
                                        tf.get_variable(config.TF_MOMENTUM_STR, shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope],
                                                        initializer=tf.constant_initializer(0, dtype=tf.float32))
                                    with tf.variable_scope(config.TF_BIAS_STR):
                                        tf.get_variable(config.TF_MOMENTUM_STR,
                                                        shape=config.TF_ANG_VAR_SHAPES_DETACHED[scope][-1],
                                                        initializer=tf.constant_initializer(0, dtype=tf.float32))

                except ValueError as e:
                    logger.critical(e)
                    logger.debug('Variables in scope %s already initialized\n'%scope)

        print([v.name for v in tf.global_variables()])


def build_tensorflw_variables_naive(separate_collision_momentum):
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
                    if 'pool' not in scope:
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
                        if separate_collision_momentum:
                            with tf.variable_scope(config.TF_WEIGHTS_STR):
                                tf.get_variable(config.TF_COL_MOMENTUM_STR, shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope],
                                                initializer=tf.constant_initializer(0, dtype=tf.float32))
                            with tf.variable_scope(config.TF_BIAS_STR):
                                tf.get_variable(config.TF_COL_MOMENTUM_STR, shape=config.TF_ANG_VAR_SHAPES_NAIVE[scope][-1],
                                                initializer=tf.constant_initializer(0, dtype=tf.float32))

                except ValueError as e:
                    logger.critical(e)
                    logger.debug('Variables in scope %s already initialized\n'%scope)

        print([v.name for v in tf.global_variables()])


def build_tensorflw_variables_multiple():
    '''
    Build the required tensorflow variables to populate the VGG-16 model
    All are initialized with zeros (initialization doesn't matter in this case as we assign exact values later)
    :param variable_shapes: Shapes of the variables
    :return:
    '''
    global logger,sess

    logger.info("Building Tensorflow Variables (Tensorflow)...")
    with sess.as_default:
        for di in ['left','straight','right']:
            with tf.variable_scope(di):
                for si,scope in enumerate(config.TF_ANG_SCOPES):
                    with tf.variable_scope(scope) as sc:

                        # Try Except because if you try get_variable with an intializer and
                        # the variable exists, you will get a ValueError saying the variable exists

                        try:
                            if 'pool' not in scope:
                                tf.get_variable(config.TF_WEIGHTS_STR,shape=config.TF_ANG_VAR_SHAPES_MULTIPLE[scope],
                                                          initializer=tf.contrib.layers.xavier_initializer())
                                tf.get_variable(config.TF_BIAS_STR, config.TF_ANG_VAR_SHAPES_MULTIPLE[scope][-1],
                                                       initializer = tf.random_uniform_initializer(minval=-0.01,maxval=+0.01,dtype=tf.float32))

                                with tf.variable_scope(config.TF_WEIGHTS_STR):
                                    tf.get_variable(config.TF_MOMENTUM_STR,shape=config.TF_ANG_VAR_SHAPES_MULTIPLE[scope],
                                                    initializer = tf.constant_initializer(0,dtype=tf.float32))
                                with tf.variable_scope(config.TF_BIAS_STR):
                                    tf.get_variable(config.TF_MOMENTUM_STR,shape=config.TF_ANG_VAR_SHAPES_MULTIPLE[scope][-1],
                                                    initializer = tf.constant_initializer(0,dtype=tf.float32))

                        except ValueError as e:
                            logger.critical(e)
                            logger.debug('Variables in scope %s already initialized\n'%scope)

            print([v.name for v in tf.global_variables()])


def build_tensorflw_variables_dual_naive():
    '''
    The dual model has dedicated variables for collision and non-collision learning as below
    input -> conv1(noncol) + conv1(col) -> conv2(noncol) + conv2(col) -> ... -> out
    :param variable_shapes: Shapes of the variables
    :return:
    '''
    global logger, sess

    logger.info("Building Tensorflow Variables (Tensorflow)...")
    with sess.as_default():
        for si, scope in enumerate(config.TF_ANG_SCOPES):
                # Try Except because if you try get_variable with an intializer and
                # the variable exists, you will get a ValueError saying the variable exists

                try:

                    if 'conv' in scope:
                        with tf.variable_scope(config.TF_NONCOL_STR):
                            logger.info('\tCreating non-collision path variables')

                            with tf.variable_scope(scope) as sc:
                                logger.info('\t\tCreating  %s variables',scope)
                                w = tf.get_variable(config.TF_WEIGHTS_STR, shape=config.TF_VAR_SHAPES_DUAL_NAIVE_NONCOL[scope],
                                                initializer=tf.contrib.layers.xavier_initializer())
                                tf.get_variable(config.TF_BIAS_STR, config.TF_VAR_SHAPES_DUAL_NAIVE_NONCOL[scope][-1],
                                                initializer=tf.random_uniform_initializer(minval=-0.01,maxval=+0.01,dtype=tf.float32))

                                logger.info('\t\t\tVariable: %s',w.name)

                        with tf.variable_scope(config.TF_COL_STR):

                            logger.info('\tCreating collision path variables')
                            logger.info('\t\tCreating  %s variables', scope)

                            with tf.variable_scope(scope) as sc:
                                w = tf.get_variable(config.TF_WEIGHTS_STR,
                                                shape=config.TF_VAR_SHAPES_DUAL_NAIVE_COL[scope],
                                                initializer=tf.contrib.layers.xavier_initializer())
                                tf.get_variable(config.TF_BIAS_STR,
                                                config.TF_VAR_SHAPES_DUAL_NAIVE_COL[scope][-1],
                                                initializer=tf.random_uniform_initializer(minval=-0.01,maxval=+0.01,dtype=tf.float32))
                                logger.info('\t\t\tVariable: %s (%s)',w.name,config.TF_VAR_SHAPES_DUAL_NAIVE_COL[scope])
                                logger.info('\t\t\tCreating  %s variables (momentum)', scope)
                                with tf.variable_scope(config.TF_WEIGHTS_STR):
                                    tf.get_variable(config.TF_MOMENTUM_STR,
                                                    shape=config.TF_VAR_SHAPES_DUAL_NAIVE_COL[scope],
                                                    initializer=tf.constant_initializer(0, dtype=tf.float32))
                                with tf.variable_scope(config.TF_BIAS_STR):
                                    tf.get_variable(config.TF_MOMENTUM_STR,
                                                    shape=config.TF_VAR_SHAPES_DUAL_NAIVE_COL[scope][-1],
                                                    initializer=tf.constant_initializer(0, dtype=tf.float32))

                    elif scope == 'out' or 'fc' in scope:
                        logger.info('\t\tCreating  %s variables', scope)

                        with tf.variable_scope(scope) as sc:

                            w = tf.get_variable(config.TF_WEIGHTS_STR,
                                                shape=config.TF_VAR_SHAPES_DUAL_NAIVE_NONCOL[scope],
                                                initializer=tf.contrib.layers.xavier_initializer())
                            tf.get_variable(config.TF_BIAS_STR,
                                            config.TF_VAR_SHAPES_DUAL_NAIVE_NONCOL[scope][-1],
                                            initializer=tf.random_uniform_initializer(minval=-0.01,maxval=+0.01,dtype=tf.float32))
                            logger.info('\t\t\tVariable: %s (%s)', w.name,
                                        config.TF_VAR_SHAPES_DUAL_NAIVE_NONCOL[scope])
                            with tf.variable_scope(config.TF_WEIGHTS_STR):
                                tf.get_variable(config.TF_MOMENTUM_STR,
                                                shape=config.TF_VAR_SHAPES_DUAL_NAIVE_NONCOL[scope],
                                                initializer=tf.constant_initializer(0,
                                                                                    dtype=tf.float32))
                            with tf.variable_scope(config.TF_BIAS_STR):
                                tf.get_variable(config.TF_MOMENTUM_STR,
                                                shape=config.TF_VAR_SHAPES_DUAL_NAIVE_NONCOL[scope][-1],
                                                initializer=tf.constant_initializer(0,
                                                                                    dtype=tf.float32))

                except ValueError as e:
                    logger.critical(e)
                    logger.debug('Variables in scope %s already initialized\n' % scope)

        print([v.name for v in tf.global_variables()])
