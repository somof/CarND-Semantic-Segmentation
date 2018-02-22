import tensorflow as tf
from tensorflow.python.platform import gfile
import helper

import scipy.misc
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    graph_name = 'frozen'

    graph_path = runs_dir + '/' + graph_name + '_graph.pb'
    print('** name:  ' + graph_path, flush=True)

    # Create a TensorFlow configuration object. This will be 
    # passed as an argument to the session.
    config = tf.ConfigProto()
    # JIT level, this can be set to ON_1 or ON_2 
    jit_level = tf.OptimizerOptions.ON_2
    config.graph_options.optimizer_options.global_jit_level = jit_level
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    with tf.Session(config=config) as sess:
        # Load graphDef
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_path, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')

        # get Network Layers from the graph
        graph = sess.graph
        image_input = graph.get_tensor_by_name('image_input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        logits = graph.get_tensor_by_name('adam_logit:0')

        # TODO: Save inference data using helper.save_inference_samples
        # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video
        flist = [
            #['../../../Shelby/dat/LegacyVideo_05_40_20170619_100415.mp4', 'LegacyVideo_05_40_20170619_100415_fcn8_100.mp4'],
            ['../../../Shelby/dat/LegacyVideo_05_40_20170622_144203.mp4', 'LegacyVideo_05_40_20170622_144203_fcn8_100.mp4'],
            #['../../../Shelby/dat/LegacyVideo_05_40_20170623_113915.mp4', 'LegacyVideo_05_40_20170623_113915_fcn8_100.mp4'],
            #['../../../Shelby/dat/LegacyVideo_05_40_20170809_132418.mp4', 'LegacyVideo_05_40_20170809_132418_fcn8_100.mp4'],
            #['project_video.mp4', 'project_video_fcn8_100.mp4'],
            ]

        for files in (flist):

            print('file: ' + files[0] + ' -> ' + files[1], flush=True)
            clip1 = VideoFileClip(files[0])

            fps = 30
            fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
            video_out = cv2.VideoWriter(files[1], int(fourcc), fps, (576, 160))

            frameno = 0
            for frame in clip1.iter_frames():
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                image = scipy.misc.imresize(frame, image_shape)

                im_softmax = sess.run([tf.nn.softmax(logits)],
                                      {keep_prob: 1.0, image_input: [image]})
                im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
                segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
                mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
                mask = scipy.misc.toimage(mask, mode="RGBA")
                street_im = scipy.misc.toimage(image)
                street_im.paste(mask, box=None, mask=mask)

                result = np.array(street_im)
                video_out.write(result)

                #cv2.imshow('fcn8s', result)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break

                frameno += 1
                if 100 < frameno:
                    break

            video_out = None




if __name__ == '__main__':
    run()
