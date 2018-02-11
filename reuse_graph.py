import tensorflow as tf
from tensorflow.python.platform import gfile
import helper


def load_graph(graph_file, use_xla=False):
    jit_level = 0
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        n_ops = len(ops)
        return sess, ops


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs_reuse'

    # Create a TensorFlow configuration object. This will be 
    # passed as an argument to the session.
    config = tf.ConfigProto()

    # JIT level, this can be set to ON_1 or ON_2 
    jit_level = tf.OptimizerOptions.ON_2
    config.graph_options.optimizer_options.global_jit_level = jit_level
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0


    with tf.Session(config=config, graph=tf.Graph()) as sess:
        #sess, _ = load_graph('./runs_reuse/frozen_graph.pb')
        #sess, _ = load_graph('./runs_reuse/optimized_graph.pb')
        sess, _ = load_graph('./runs_reuse/eightbit_graph.pb')

        graph = sess.graph
        adam_angst = graph.get_tensor_by_name('adam_logit:0')
        image_input = graph.get_tensor_by_name('image_input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        print('Adam Angst = ',adam_angst)
        logits = graph.get_tensor_by_name('adam_logit:0')

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)
    
        #print("%d ops in the final graph." % len(output_graph_def.node))


        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

