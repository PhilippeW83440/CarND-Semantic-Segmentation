import os.path
import tensorflow as tf
import helper
import helper_cityscapes
import warnings
from distutils.version import LooseVersion
import project_tests as tests

from timeit import default_timer as timer


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
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
        
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    init = tf.truncated_normal_initializer(stddev = 0.01)
    reg = tf.contrib.layers.l2_regularizer(1e-3)
    
    # reduce dimensions with conv1x1 filters
    conv1x1_l3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, padding='same', 
                                  kernel_initializer=init, kernel_regularizer=reg)
    conv1x1_l4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, padding='same', 
                                  kernel_initializer=init, kernel_regularizer=reg)
    conv1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, padding='same', 
                                  kernel_initializer=init, kernel_regularizer=reg)
    
    # upsample output of encoder by 2
    deconv_1 = tf.layers.conv2d_transpose(conv1x1, num_classes, 4, 2, padding='same',
                                          kernel_initializer=init, kernel_regularizer=reg)
    # add skip connection from layer 4
    deconv_1 = tf.add(deconv_1, conv1x1_l4)
    # upsample by 2
    deconv_2 = tf.layers.conv2d_transpose(deconv_1, num_classes, 4, 2, padding='same',
                                          kernel_initializer=init, kernel_regularizer=reg)
    # add skip connection from layer 3
    deconv_2 = tf.add(deconv_2, conv1x1_l3)
    # upsample by 8: so we are back to the original image size (that was downsampled by 32 in encoder)
    deconv_3 = tf.layers.conv2d_transpose(deconv_2, num_classes, 16, 8, padding='same',
                                          kernel_initializer=init, kernel_regularizer=reg)

    #tf.Print(deconv_3, [tf.shape](deconv_3)])
    
    return deconv_3


def build_predictor(nn_last_layer):
  softmax_output = tf.nn.softmax(nn_last_layer)
  predictions_argmax = tf.argmax(softmax_output, axis=-1, output_type=tf.int64)
  return softmax_output, predictions_argmax

def build_iou_metric(correct_label, predictions_argmax, num_classes):
  labels_argmax = tf.argmax(correct_label, axis=-1, output_type=tf.int64)
  iou, iou_op = tf.metrics.mean_iou(labels_argmax, predictions_argmax, num_classes)
  return iou, iou_op


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # l2_reg does not help here apparently (on kitti) ...
    # regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # #regularization_loss = tf.add_n(regularization_losses, name='reg_loss') # Scalar
    # cross_entropy_loss = cross_entropy_loss + sum(regularization_losses)
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_train_batches_fn, get_valid_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, iou, iou_op, saver):
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
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    best_iou = 0
    for epoch in range(epochs):
        start = timer()
        losses = []
        ious = []
        for image, label in get_train_batches_fn(batch_size):
            _, loss, _ = sess.run([train_op, cross_entropy_loss, iou_op], feed_dict={input_image: image, correct_label: label, keep_prob: 0.8})
            print(loss)
            losses.append(loss)
            ious.append(sess.run(iou))
        end = timer()
        print("EPOCH {} ...".format(epoch+1))
        print("  time {} ...".format(end-start))
        print("  Train Xentloss = {:.4f}".format(sum(losses) / len(losses))) 
        print("  Train IOU = {:.4f}".format(sum(ious) / len(ious))) 

        losses = []
        ious = []
        for image, label in get_valid_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, iou_op], feed_dict={input_image: image, correct_label: label, keep_prob: 1})
            losses.append(loss)
            ious.append(sess.run(iou))
        print("  Valid Xentloss = {:.4f}".format(sum(losses) / len(losses))) 
        valid_iou = sum(ious) / len(ious)
        print("  Valid IOU = {:.4f}".format(valid_iou)) 

        if (valid_iou > best_iou):
            saver.save(sess, './fcn8s')
            print("  model saved")
            best_iou = valid_iou


def run():
    #num_classes = 2
    #image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './city_runs'
    city_data_dir = './data/cityscapes'

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    train_images, valid_images, test_images, num_classes, label_colors, image_shape = helper_cityscapes.load_data(city_data_dir)
    print("len: train_images {}, valid_images {}, test_images {}".format(len(train_images), len(valid_images), len(test_images)))

    epochs = 2 # XXX temp for testing purposes
    batch_size = 128
    learning_rate = 1e-4 # 1e-4
    correct_label = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], num_classes))

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches

        get_train_batches_fn = helper_cityscapes.gen_batch_function(train_images, image_shape)
        get_valid_batches_fn = helper_cityscapes.gen_batch_function(valid_images, image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        fcn8s_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(fcn8s_output, correct_label, learning_rate, num_classes)

        softmax_output, predictions_argmax = build_predictor(fcn8s_output)
        iou, iou_op = build_iou_metric(correct_label, predictions_argmax, num_classes)

        saver = tf.train.Saver()

        train_nn(sess, epochs, batch_size, get_train_batches_fn, get_valid_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, iou, iou_op, saver)

        saver.restore(sess, tf.train.latest_checkpoint('.'))

        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper_cityscapes.save_inference_samples(runs_dir, test_images, sess, image_shape, logits, keep_prob, input_image, label_colors)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
