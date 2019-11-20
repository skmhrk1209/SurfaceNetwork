import tensorflow as tf


def model(inputs, num_units, num_layers):
    for i in range(num_layers):
        with tf.variable_scope(f'layer_{i}'):
            inputs = tf.layers.dense(inputs, num_units)
            inputs = tf.layers.batch_normalization(inputs, training=True)
            inputs = tf.nn.tanh(inputs)
    return inputs


with tf.Graph().as_default():

    x = tf.placeholder(tf.float32, (None, 3), name='x')
    y = tf.identity(model(x, 128, 8), name='y')

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        graph = tf.graph_util.convert_variables_to_constants(session, session.graph_def, ['y'])
        tf.train.write_graph(graph, '.', 'graph.pb', as_text=False)
