import tensorflow as tf
import scipy.misc
from graph_utils import load_graph
from timeit import default_timer as timer

# GTX 1080 TI with 11 GB:
# -----------------------
# base model: 100 ms
# frozen model: 63 ms
# optimized model: 62 ms (bof...)

#sess, _ = load_graph('frozen_model.pb')
sess, _ = load_graph('optimized_graph.pb')

graph = sess.graph
input_image = graph.get_tensor_by_name('image_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
softmax = graph.get_tensor_by_name('Softmax:0')

test_image = scipy.misc.imread("test_image.png")
image_shape = (256, 512)
img = scipy.misc.imresize(test_image, image_shape)

for i in range(20):
    start = timer()
    probs = sess.run([softmax], {input_image: [img], keep_prob: 1.0})
    end = timer()
    print("predict time {}".format(end-start))

#print(probs)
