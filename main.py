#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


num_classes = 2
image_shape = (160, 576)

EPOCHS = 40
BATCH_SIZE = 16
DROPOUT = 0.7

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
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    vgg_model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

    vgg_graph = tf.get_default_graph()
    image_input = vgg_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = vgg_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = vgg_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = vgg_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = vgg_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3, layer4, layer7       
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
    # Encoder
    # vgg16
    
    # Decoder
    fcn8_layer8 = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, kernel_size=1, name="fcn8_8")
    # Upsample 
    fcn8_layer9 = tf.layers.conv2d_transpose(fcn8_layer8, filters=vgg_layer4_out.get_shape().as_list()[-1], kernel_size=4, strides=(2, 2), padding='SAME', name="fcn8_9")
    # skip connection 
    fcn8_layer9_skip = tf.add(fcn8_layer9, vgg_layer4_out, name="fcn8_9ANDvgg_layer4")
    # Upsample 
    fcn8_layer10 = tf.layers.conv2d_transpose(fcn8_layer9_skip , filters=vgg_layer3_out.get_shape().as_list()[-1], kernel_size=4, strides=(2, 2), padding='SAME', name="fcn8_10")
    # skip connection
    fcn8_layer10_skip = tf.add(fcn8_layer10, vgg_layer3_out, name="fcn8_10ANDvgg_layer3")
    # Upsample
    fcn8_layer11 = tf.layers.conv2d_transpose(fcn8_layer10_skip, filters=num_classes, kernel_size=16, strides=(8, 8), padding='SAME', name="fcn8_11")
    
    return fcn8_layer11
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
    
    # reshape tensors to 2D (row = pixel, column = class)
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="FCN8_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))
    # get distance from labels using cross entropy
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    # total loss
    loss_op = tf.reduce_mean(cross_entropy_loss, name="FCN8_loss")
    # find the weights which yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="FCN8_train_op")

    return logits, train_op, loss_op
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
    for epoch in range(epochs):
        for i, (image, label) in enumerate(get_batches_fn(batch_size)):
            _, loss = sess.run([train_op, cross_entropy_loss], 
            feed_dict={input_image:image, correct_label:label, keep_prob:DROPOUT, learning_rate:1e-4}) # 

            print("epoch: {}, batch: {}, loss: {}".format(epoch+1, i, loss))

tests.test_train_nn(train_nn)

def run():
    data_dir = '/data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        final_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        label = tf.placeholder(tf.int32, shape=[None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, loss = optimize(final_layer, label, learning_rate, num_classes)

        saver = tf.train.Saver()
        # saver.restore(sess, './runs/fcn8.model')
        
        sess.run(tf.global_variables_initializer())
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, loss, 
                input_image, label, keep_prob, learning_rate)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        saver.save(sess, './runs/fcn8.model')
        
if __name__ == '__main__':
    run()
