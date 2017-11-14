import tensorflow as tf
import scipy.misc
from graph_utils import load_graph
from timeit import default_timer as timer

# GTX 1080 TI with 11 GB:
# -----------------------
# base model: 100 ms
# frozen model: 63 ms
# optimized model: 62 ms (bof...)


# ------------------------------------------------------------------------------------------------

# TensorFlow supports both JIT (just in time) and AOT (ahead of time) compilation.
# 
# AOT compilation is the kind used in a C or C++ program; it compiles the program “ahead” of the actual use. A really cool aspect of AOT compilation is you can potentially create a static binary file, meaning it’s entirely self contained. You can deploy it by simply downloading the file and executing it, without concern for downloading extra software, besides necessary hardware drivers, i.e. for GPU use.
# 
# JIT compilation doesn’t compile code until it’s actually run. You can imagine as a piece of code is being interpreted machine instructions are concurrently generated. Nothing will change during the initial interpretation, the JIT might as well not exist. However, on the second and all future uses that piece of code will no longer be interpreted. Instead the compiled machine instructions will be used.
# 
# Under the hood AOT and JIT compilation make use of XLA (Accelerated Linear Algebra). XLA is a compiler and runtime for linear algebra. XLA takes a TensorFlow graph and uses LLVM to generate machine code instructions. LLVM is itself a compiler which generates machine code from its IR (intermediate representation). So, in a nutshell:
# 
# TensorFlow -> XLA -> LLVM IR -> Machine Code
# 
# which means TensorFlow can potentially be used on any architecture LLVM generates code for.
# 
# Both AOT and JIT compilation are experimental in TensorFlow. However, JIT compilation is fairly straightforward to apply. Note JIT compilation is NOT limited to inference but can be used during training as well.

config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 1})
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
# JIT level, this can be set to ON_1 or ON_2 
# well ... does not help here ...
jit_level = tf.OptimizerOptions.ON_1
#jit_level = tf.OptimizerOptions.ON_2
config.graph_options.optimizer_options.global_jit_level = jit_level

# ... does not help so far ...
# ------------------------------------------------------------------------------------------------


#sess, _ = load_graph('frozen_model.pb')
sess, _ = load_graph('optimized_graph.pb')

graph = sess.graph
input_image = graph.get_tensor_by_name('image_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
softmax = graph.get_tensor_by_name('Softmax:0')

test_image = scipy.misc.imread("test_image.png")
image_shape = (256, 512)
img = scipy.misc.imresize(test_image, image_shape)

for i in range(30):
    start = timer()
    probs = sess.run([softmax], {input_image: [img], keep_prob: 1.0})
    end = timer()
    print("predict time {}".format(end-start))

#print(probs)
