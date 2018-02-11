import tensorflow as tf
from tensorflow.python.platform import gfile
import helper

import scipy.misc

import cv2
import numpy as np
from moviepy.editor import VideoFileClip


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

    # avoid Out of GPU memory Error
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

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

        flist = [
            #['../../../Shelby/dat/LegacyVideo_05_40_20170619_100415.mp4', 'LegacyVideo_05_40_20170619_100415_fcn8_100.mp4'],
            #['../../../Shelby/dat/LegacyVideo_05_40_20170622_144203.mp4', 'LegacyVideo_05_40_20170622_144203_fcn8_100.mp4'],
            #['../../../Shelby/dat/LegacyVideo_05_40_20170623_113915.mp4', 'LegacyVideo_05_40_20170623_113915_fcn8_100.mp4'],
            #['../../../Shelby/dat/LegacyVideo_05_40_20170809_132418.mp4', 'LegacyVideo_05_40_20170809_132418_fcn8_100.mp4'],
            ['project_video.mp4', 'project_video_fcn8_100.mp4'],
            ]

        for files in (flist):

            print('file: ' + files[0] + ' -> ' + files[1], flush=True)
            clip1 = VideoFileClip(files[0])

            fps = 30
            fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
            video_out = cv2.VideoWriter(files[1], int(fourcc), fps, (576, 160))

            frameno = 0
            for frame in clip1.iter_frames():
                if frameno % 1 == 0:

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    image = scipy.misc.imresize(frame, image_shape)

                    im_softmax = sess.run([tf.nn.softmax(logits)],
                                          {keep_prob: 1.0,
                                           image_input: [image]})
                    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
                    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
                    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
                    mask = scipy.misc.toimage(mask, mode="RGBA")
                    street_im = scipy.misc.toimage(image)
                    street_im.paste(mask, box=None, mask=mask)

                    result = np.array(street_im)
                    #cv2.imshow('fcn8s', result)
                    video_out.write(result)
 
                frameno += 1
                if 100 < frameno:
                    break

                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break

            video_out = None

        #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)
        #print("%d ops in the final graph." % len(output_graph_def.node))

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
