import tensorflow as tf
import scipy.misc
from graph_utils import load_graph
from timeit import default_timer as timer
import numpy as np
from collections import namedtuple
from moviepy.editor import VideoFileClip

import argparse

# GTX 1080 TI with 11 GB:
# -----------------------
# base model: 100 ms
# frozen model: 63 ms
# optimized model: 62 ms (bof...)


# ------------------------------------------------------------------------------------------------


# config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 1})
# #config.gpu_options.allow_growth = True
# #config.gpu_options.per_process_gpu_memory_fraction = 0.9
# # JIT level, this can be set to ON_1 or ON_2 
# # well ... does not help here ...
# jit_level = tf.OptimizerOptions.ON_1
# #jit_level = tf.OptimizerOptions.ON_2
# config.graph_options.optimizer_options.global_jit_level = jit_level

# ... does not help so far ...
# ------------------------------------------------------------------------------------------------

# cf https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

# num_classes 20
# background (unlabeled) + 19 classes as per official benchmark
# cf "The Cityscapes Dataset for Semantic Urban Scene Understanding"



Label = namedtuple('Label', ['name', 'color'])

label_defs = [
    Label('unlabeled',     (0,     0,   0)),
    #Label('dynamic',       (111,  74,   0)),
    #Label('ground',        ( 81,   0,  81)),
    Label('road',          (128,  64, 128)),
    Label('sidewalk',      (244,  35, 232)),
    #Label('parking',       (250, 170, 160)),
    #Label('rail track',    (230, 150, 140)),
    Label('building',      ( 70,  70,  70)),
    Label('wall',          (102, 102, 156)),
    Label('fence',         (190, 153, 153)),
    #Label('guard rail',    (180, 165, 180)),
    #Label('bridge',        (150, 100, 100)),
    #Label('tunnel',        (150, 120,  90)),
    Label('pole',          (153, 153, 153)),
    Label('traffic light', (250, 170,  30)),
    Label('traffic sign',  (220, 220,   0)),
    Label('vegetation',    (107, 142,  35)),
    Label('terrain',       (152, 251, 152)),
    Label('sky',           ( 70, 130, 180)),
    Label('person',        (220,  20,  60)),
    Label('rider',         (255,   0,   0)),
    Label('car',           (  0,   0, 142)),
    Label('truck',         (  0,   0,  70)),
    Label('bus',           (  0,  60, 100)),
    #Label('caravan',       (  0,   0,  90)),
    #Label('trailer',       (  0,   0, 110)),
    Label('train',         (  0,  80, 100)),
    Label('motorcycle',    (  0,   0, 230)),
    Label('bicycle',       (119, 11, 32))]

parser = argparse.ArgumentParser(description='Test graph')
parser.add_argument('--video', type=str, default=None, help='video')
parser.add_argument('--graph', type=str, default='models/transformed_graph.pb', help='graph for inference')
args = parser.parse_args()

video = args.video
name_graph = args.graph

#sess, _ = load_graph('models/base_graph.pb')
#sess, _ = load_graph('models/frozen_graph.pb')
#sess, _ = load_graph('models/frozen_model.pb')
#sess, _ = load_graph('models/optimized_graph.pb')
#sess, _ = load_graph('models/transformed_graph.pb')

sess, _ = load_graph(name_graph)

graph = sess.graph
input_image = graph.get_tensor_by_name('image_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
softmax = graph.get_tensor_by_name('Softmax:0')
predictions_argmax = tf.argmax(softmax, axis=-1, output_type=tf.int64)

num_classes = len(label_defs)
label_colors = {i: np.array(l.color) for i, l in enumerate(label_defs)}

test_image = scipy.misc.imread("data/test_image.png")
image_shape = (256, 512)



def predict_image(image):
    start = timer()
    image = scipy.misc.imresize(image, image_shape)

    start_inference = timer()
    labels = sess.run([predictions_argmax], feed_dict={input_image: [image], keep_prob: 1})
    end_inference = timer()

    labels = labels[0].reshape(image_shape[0], image_shape[1])
    # create an overlay
    labels_colored = np.zeros((*image_shape, 4)) # 4 for RGBA
    for label in label_colors:
        label_mask = labels == label
        labels_colored[label_mask] = np.array((*label_colors[label], 127))

    mask = scipy.misc.toimage(labels_colored, mode="RGBA")
    pred_image = scipy.misc.toimage(image)
    pred_image.paste(mask, box=None, mask=mask)
    res_image = np.array(pred_image)
    end = timer()

    if dump_time:
        time_inference = end_inference - start_inference
        time_img_processing = (end - start) - time_inference
        print("time: inference {:.6f} overlay {:.6f}".format(time_inference, time_img_processing))
    return res_image

dump_time = True

for i in range(10):
    pred_image = predict_image(test_image)
scipy.misc.imsave("data/pred_image.png", pred_image)

dump_time = False

if video is not None:
    # ffmpeg -f image2 -i stuttgart_00_000000_000%03d_leftImg8bit.png -c:v libx264 -r 30/1.8 -pix_fmt yuv420p out.mp4
    output = 'data/output.mp4'
    clip1 = VideoFileClip(video)
    output_clip = clip1.fl_image(predict_image)
    #%time 
    output_clip.write_videofile(output, audio=False)

