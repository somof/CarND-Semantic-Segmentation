import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import time

import scipy.misc
import cv2
import numpy as np
from moviepy.editor import VideoFileClip


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    with tf.name_scope(vgg_tag):
        image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # 1x1 Convolution of VGG Layer 7
    fcn_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  name='fcn_layer7')

    # 2x2 Upsampled FCN Layer 7
    fcn_dconv_layer7 = tf.layers.conv2d_transpose(fcn_layer7, num_classes, 4,
                                                  strides=(2, 2),
                                                  padding='same',
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                  name='fcn_dconv_layer7')

    # 1x1 Convolution of VGG Layer 4
    fcn_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  name='fcn_layer4')

    # Skip Connection between Layer 4 and 7
    fcn_skip_layer4 = tf.add(fcn_dconv_layer7, fcn_layer4)

    # 2x2 Upsampled FCN Skip Connection above
    fcn_dconv_layer4 = tf.layers.conv2d_transpose(fcn_skip_layer4, num_classes, 4,
                                                  strides=(2, 2),
                                                  padding='same',
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                  name='fcn_dconv_layer4')
    # 1x1 Convolution of VGG layer 3
    fcn_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  name='fcn_layer3')

    # additional Skip Connection with Layer 3
    fcn_skip_layer3 = tf.add(fcn_dconv_layer4, fcn_layer3)

    # 8x8 Upsampled FCN Layer 3
    nn_last_layer = tf.layers.conv2d_transpose(fcn_skip_layer3, num_classes, 16,
                                               strides=(8, 8),
                                               padding='same',
                                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                               name='fcn_dconv_layer_3')

    # tf.Print(fcn_dconv_layer7, [tf.shape(fcn_dconv_layer7)[1:3]])
    return nn_last_layer


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # transform the 4D output tensor into 2D tensor
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="adam_logit")
    correct_labels = tf.reshape(correct_label, (-1, num_classes))

    # loss function and optimizer
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=correct_labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    sess.run(tf.global_variables_initializer())
    print("Training...", flush=True)

    start_time = time.clock()
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image,
                                          correct_label: label,
                                          keep_prob: 0.5,
                                          learning_rate: 0.001})
        end_time = time.clock()
        train_time = end_time - start_time
        print("Epoch: {}/{}, {:.3f} sec, Loss: {:.3f}".format(epoch + 1, epochs, train_time, loss), flush=True)
        if loss <= 0.010:
            return


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs_video'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        epochs = 1000
        batch_size = 20
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # export the graph for inference programs
        output_node_names = 'adam_logit'
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(","))  # The output node names are used to select the usefull nodes
        print("output_node: {}".format(output_node_names.split(",")))
        saver = tf.train.Saver()
        saver.save(sess, runs_dir + '/fcn')
        # tf.train.write_graph(tf.get_default_graph().as_graph_def(), '', './runs/base_graph.pb', False)
        tf.train.write_graph(output_graph_def, '', runs_dir + '/frozen_graph.pb', False)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        flist = [
            ['../../../Shelby/dat/LegacyVideo_05_40_20170622_144203.mp4', 'LegacyVideo_05_40_20170622_144203_fcn8_1000.mp4', (612, 512)],
            ['../../../Shelby/dat/LegacyVideo_05_40_20170623_113915.mp4', 'LegacyVideo_05_40_20170623_113915_fcn8_1000.mp4', (612, 512)],
            ['../../../Shelby/dat/LegacyVideo_05_40_20170809_132418.mp4', 'LegacyVideo_05_40_20170809_132418_fcn8_1000.mp4', (612, 512)],
            ['../../../Shelby/dat/LegacyVideo_05_40_20170626_133711.mp4', 'LegacyVideo_05_40_20170626_133711_fcn8_1000.mp4', (612, 512)],
            ['../../../Shelby/dat/LegacyVideo_05_40_20170809_132654.mp4', 'LegacyVideo_05_40_20170809_132654_fcn8_1000.mp4', (612, 512)],
            ['../../../Shelby/dat/LegacyVideo_05_40_20170809_133009.mp4', 'LegacyVideo_05_40_20170809_133009_fcn8_1000.mp4', (612, 512)],
            ['../../../Shelby/dat/LegacyVideo_05_40_20170619_100415.mp4', 'LegacyVideo_05_40_20170619_100415_fcn8_1000.mp4', (612, 512)],
            ['project_video.mp4', 'project_video_fcn8_100.mp4', (640, 360)]]

        for files in (flist):

            print('file: ' + files[0] + ' -> ' + files[1], flush=True)
            clip1 = VideoFileClip(files[0])

            save_size = files[2]

            fps = 30
            fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
            video_out = cv2.VideoWriter(files[1], int(fourcc), fps, save_size)

            frameno = 0
            for frame in clip1.iter_frames():
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                image = scipy.misc.imresize(frame, image_shape)

                im_softmax = sess.run([tf.nn.softmax(logits)],
                                      {keep_prob: 1.0, input_image: [image]})
                im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
                segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
                mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
                mask = scipy.misc.toimage(mask, mode="RGBA")
                street_im = scipy.misc.toimage(image)
                street_im.paste(mask, box=None, mask=mask)

                result = np.array(street_im)
                result2 = cv2.resize(result, (save_size[0], save_size[1]))
                video_out.write(result2)
 
                frameno += 1
                if 3000 < frameno:
                    break

                #cv2.imshow('fcn8s', result2)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break

            video_out = None


if __name__ == '__main__':
    run()
