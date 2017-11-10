import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

from collections import namedtuple

#-------------------------------------------------------------------------------
# Data
#-------------------------------------------------------------------------------

data_dir="/home/philippew/Udacity-term3/CarND-Semantic-Segmentation/data/cityscapes"

#-------------------------------------------------------------------------------
# Labels
#-------------------------------------------------------------------------------
Label = namedtuple('Label', ['name', 'color'])

def rgb2bgr(tpl):
    return (tpl[2], tpl[1], tpl[0])

label_defs = [
    Label('unlabeled',     rgb2bgr((0,     0,   0))),
    Label('dynamic',       rgb2bgr((111,  74,   0))),
    Label('ground',        rgb2bgr(( 81,   0,  81))),
    Label('road',          rgb2bgr((128,  64, 128))),
    Label('sidewalk',      rgb2bgr((244,  35, 232))),
    Label('parking',       rgb2bgr((250, 170, 160))),
    Label('rail track',    rgb2bgr((230, 150, 140))),
    Label('building',      rgb2bgr(( 70,  70,  70))),
    Label('wall',          rgb2bgr((102, 102, 156))),
    Label('fence',         rgb2bgr((190, 153, 153))),
    Label('guard rail',    rgb2bgr((180, 165, 180))),
    Label('bridge',        rgb2bgr((150, 100, 100))),
    Label('tunnel',        rgb2bgr((150, 120,  90))),
    Label('pole',          rgb2bgr((153, 153, 153))),
    Label('traffic light', rgb2bgr((250, 170,  30))),
    Label('traffic sign',  rgb2bgr((220, 220,   0))),
    Label('vegetation',    rgb2bgr((107, 142,  35))),
    Label('terrain',       rgb2bgr((152, 251, 152))),
    Label('sky',           rgb2bgr(( 70, 130, 180))),
    Label('person',        rgb2bgr((220,  20,  60))),
    Label('rider',         rgb2bgr((255,   0,   0))),
    Label('car',           rgb2bgr((  0,   0, 142))),
    Label('truck',         rgb2bgr((  0,   0,  70))),
    Label('bus',           rgb2bgr((  0,  60, 100))),
    Label('caravan',       rgb2bgr((  0,   0,  90))),
    Label('trailer',       rgb2bgr((  0,   0, 110))),
    Label('train',         rgb2bgr((  0,  80, 100))),
    Label('motorcycle',    rgb2bgr((  0,   0, 230))),
    Label('bicycle', rgb2bgr((119, 11, 32)))]


def build_file_list(images_root, labels_root, sample_name):
    image_sample_root = images_root + '/' + sample_name
    image_root_len    = len(image_sample_root)
    label_sample_root = labels_root + '/' + sample_name
    image_files       = glob(image_sample_root + '/**/*png')
    file_list         = []
    for f in image_files:
        f_relative      = f[image_root_len:]
        f_dir           = os.path.dirname(f_relative)
        f_base          = os.path.basename(f_relative)
        f_base_gt = f_base.replace('leftImg8bit', 'gtFine_color')
        f_label   = label_sample_root + f_dir + '/' + f_base_gt
        if os.path.exists(f_label):
            file_list.append((f, f_label))
    return file_list


def load_data_cityscapes(data_folder):
    images_root = data_folder + '/leftImg8bit'
    labels_root = data_folder + '/gtFine'

    train_images = build_file_list(images_root, labels_root, 'train')
    valid_images = build_file_list(images_root, labels_root, 'val')
    test_images = build_file_list(images_root, labels_root, 'test')
    num_classes = len(label_defs)
    label_colors = {i: np.array(l.color) for i, l in enumerate(label_defs)}
    image_shape = (512, 256)

    return train_images, valid_images, test_images, num_classes, label_colors, image_shape

    #num_images = len(image_paths)
    #random.shuffle(image_paths)
    #valid_images = image_paths[:(int)(valid_frac*num_images)]
    #train_images = image_paths[(int)(valid_frac*num_images):]
    #return train_images, valid_images, label_paths

def gen_batch_function_cityscapes(image_paths, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn_cityscapes(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        #background_color = np.array([255, 0, 0])
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            image_files = image_paths[batch_i:batch_i+batch_size]

            images = []
            labels = []

            for f in image_files:
                image_file = f[0]
                gt_image_file = f[1]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                label_bg = np.zeros([image_shape[0], image_shape[1]], dtype=bool)
                label_list = []
                for ldef in label_defs[1:]:
                    label_current = np.all(gt_image == ldef.color, axis=2)
                    label_bg |= label_current
                    label_list.append(label_current)

                label_bg = ~label_bg
                label_all = np.dstack([label_bg, *label_list])
                label_all = label_al.astype(np.float32)

                images.append(image)
                labels.append(label_all)

            yield np.array(images), np.array(labels)
    return get_batches_fn_cityscapes


def gen_test_output_cityscapes(sess, logits, keep_prob, image_pl, image_files, image_shape, label_colors):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    #for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')): ### TODO XXX
    for f in image_files:
        image_file = f[0]
        gt_image_file = f[1]

        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        # labels: 2D shape of floats
        labels = sess.run(
            [tf.argmax(tf.nn.softmax(logits), axis=-1)],
            {keep_prob: 1.0, image_pl: [image]})

        labels_colored = np.zeros_like(image)
        for label in label_colors:
            label_mask = labels == label
            labels_colored[label_mask] = label_colors[label]

        #im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        #segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        #mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(labels_colored, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples_cityscapes(runs_dir, image_files, sess, image_shape, logits, keep_prob, input_image, label_colors):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, image_files, image_shape, label_colors)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)




train_images, valid_images, test_images, num_classes, label_colors, image_shape = load_data_cityscapes(data_dir)
train_generator = gen_batch_function_cityscapes(train_images, image_shape)
valid_generator = gen_batch_function_cityscapes(valid_images, image_shape)
